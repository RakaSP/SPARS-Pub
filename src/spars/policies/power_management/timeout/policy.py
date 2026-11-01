from __future__ import annotations

import math
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from spars.core.decision import SchedulingDecision

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.trigger import PlanContext


EPS = 1e-9


class TimeoutPowerManager:
    """Switches a node off after it has been idle for the timeout; pair with a PSAS scheduler."""

    # timeout: idle seconds before a node is switched off; None never switches off.
    DEFAULTS = {"timeout": None}

    def __init__(
        self,
        machines: "Machine",
        start_time: float,
        config: dict[str, Any],
    ) -> None:
        unexpected_config = set(config) - {"timeout"}
        if unexpected_config:
            raise ValueError(
                "Invalid timeout power-manager config: "
                f"unexpected keys: {sorted(unexpected_config)}"
            )

        self.machines = machines
        self.state = machines.nodes
        self.transitions = machines.transition_map

        self.current_time = float(start_time)
        self.timeout = config.get("timeout")

        self.timeout_list: dict[int, float] = {}
        self.next_timeout_at: float | None = None
        self.call_me_laters: list[float] = []
        self._call_me_later_set: set[float] = set()
        self.events: list[dict[str, Any]] = []

        # Platform transition definitions are immutable during a simulation.
        # Cache the scalar values used on every power-management decision.
        self._transition_time_cache: dict[tuple[int, str, str], float] = {}
        self._wake_lead_time_cache: dict[int, float] = {}

    def _push_event(
        self,
        timestamp: float,
        event: dict[str, Any],
    ) -> None:
        event_bucket = next(
            (
                item
                for item in self.events
                if item["timestamp"] == timestamp
            ),
            None,
        )

        if event_bucket is not None:
            event_bucket["events"].append(event)
            return

        self.events.append({
            "timestamp": timestamp,
            "events": [event],
        })
        self.events.sort(key=itemgetter("timestamp"))

    def _transition_time(
        self,
        node_id: int,
        from_state: str,
        to_state: str,
    ) -> float:
        key = (node_id, from_state, to_state)
        cached = self._transition_time_cache.get(key)
        if cached is not None:
            return cached

        node_name = self.state[node_id]["node_name"]
        transition = self.transitions[node_name][
            (from_state, to_state)
        ]
        value = float(transition["transition_time"])
        self._transition_time_cache[key] = value
        return value

    def _wake_lead_time(self, node_id: int) -> float:
        cached = self._wake_lead_time_cache.get(node_id)
        if cached is not None:
            return cached

        value = float(
            self._transition_time(
                node_id,
                "sleeping",
                "switching_on",
            )
            + self._transition_time(
                node_id,
                "switching_on",
                "active",
            )
        )
        self._wake_lead_time_cache[node_id] = value
        return value

    def _idle_nodes(self) -> list[int]:
        return [
            node_id
            for node_id, node in self.state.items()
            if (
                node.get("state") == "active"
                and node.get("job_id") is None
            )
        ]

    def _sleeping_nodes(self) -> list[int]:
        return [
            node_id
            for node_id, node in self.state.items()
            if node.get("state") == "sleeping"
        ]

    @staticmethod
    def _allocated_nodes(
        scheduling_decision: SchedulingDecision,
    ) -> set[int]:
        allocated_nodes: set[int] = set()

        for event_bucket in scheduling_decision.events:
            for event in event_bucket["events"]:
                if event.get("type") != "execution_start":
                    continue

                allocated_nodes.update(
                    event.get("nodes", [])
                )

        allocated_nodes.update(scheduling_decision.reserved_nodes)
        return allocated_nodes

    def prepare(self, current_time: float) -> None:
        self.current_time = float(current_time)

        if self.timeout is None:
            self.timeout_list = {}
            self.next_timeout_at = None
            return

        idle_nodes = self._idle_nodes()
        idle_node_set = set(idle_nodes)
        expires_at = (
            self.current_time
            + float(self.timeout)
        )

        self.timeout_list = {
            node_id: deadline
            for node_id, deadline in self.timeout_list.items()
            if node_id in idle_node_set
        }

        for node_id in idle_nodes:
            self.timeout_list.setdefault(
                node_id,
                expires_at,
            )

    def needs_replan(self, ctx: "PlanContext") -> bool:
        """Replan when decide() must act on a timeout deadline."""
        if self.timeout is None or not self.timeout_list:
            return False

        earliest = min(self.timeout_list.values())
        if ctx.now >= earliest:
            return True
        return self.next_timeout_at != earliest

    def remaining_idle_timeout(
        self,
        node_id: int,
        current_time: float,
    ) -> float:
        if self.timeout is None:
            return math.inf

        deadline = self.timeout_list.get(node_id)
        if deadline is None:
            return math.inf

        return float(
            deadline - float(current_time)
        )

    def _emit_wake_triggers(
        self,
        scheduling_decision: SchedulingDecision,
    ) -> list[int]:
        current_time = self.current_time
        sleeping_nodes = self._sleeping_nodes()
        sleeping_node_set = set(sleeping_nodes)
        earliest_wake_by_node: dict[int, float] = {}

        for planned_job in scheduling_decision.future_plan:
            start_time = float(planned_job.start_time)

            if start_time <= current_time + EPS:
                continue

            for node_id in planned_job.nodes:
                if node_id not in sleeping_node_set:
                    continue

                wake_time = (
                    start_time
                    - self._wake_lead_time(node_id)
                )
                previous_wake = earliest_wake_by_node.get(
                    node_id
                )

                if (
                    previous_wake is None
                    or wake_time < previous_wake
                ):
                    earliest_wake_by_node[node_id] = wake_time

        immediate_wake_nodes = [
            node_id
            for node_id, wake_time
            in earliest_wake_by_node.items()
            if wake_time <= current_time + EPS
        ]

        if immediate_wake_nodes:
            self._push_event(
                current_time,
                {
                    "type": "switch_on",
                    "nodes": immediate_wake_nodes,
                },
            )

            immediate_wake_set = set(
                immediate_wake_nodes
            )
            sleeping_nodes = [
                node_id
                for node_id in sleeping_nodes
                if node_id not in immediate_wake_set
            ]

        future_wake_times = sorted({
            wake_time
            for wake_time in earliest_wake_by_node.values()
            if wake_time > current_time + EPS
        })

        for wake_time in future_wake_times:
            self._push_event(
                float(wake_time),
                {"type": "call_me_later_so"},
            )

        return sleeping_nodes

    def _emit_timeout_policy(
        self,
        scheduling_decision: SchedulingDecision,
    ) -> None:
        if self.timeout is None:
            return

        current_time = self.current_time
        allocated_nodes = self._allocated_nodes(
            scheduling_decision
        )
        idle_nodes = set(self._idle_nodes()) - allocated_nodes

        retained_timeouts: dict[int, float] = {}
        nodes_to_switch_off: list[int] = []
        next_earliest_timeout: float | None = None

        for node_id, deadline in self.timeout_list.items():
            if node_id not in idle_nodes:
                continue

            if current_time >= deadline:
                selected_plan = next(
                    (
                        planned_job
                        for planned_job
                        in scheduling_decision.future_plan
                        if node_id in planned_job.nodes
                    ),
                    None,
                )

                if selected_plan is None:
                    nodes_to_switch_off.append(node_id)
                    continue

                transition_time = float(
                    self._transition_time(
                        node_id,
                        "switching_off",
                        "sleeping",
                    )
                    + self._wake_lead_time(node_id)
                )

                if (
                    current_time + transition_time
                    <= float(selected_plan.start_time)
                ):
                    nodes_to_switch_off.append(node_id)
                else:
                    retained_timeouts[node_id] = deadline
            else:
                retained_timeouts[node_id] = deadline
                next_earliest_timeout = (
                    deadline
                    if next_earliest_timeout is None
                    else min(
                        next_earliest_timeout,
                        deadline,
                    )
                )

        self.timeout_list = retained_timeouts

        if nodes_to_switch_off:
            self._push_event(
                current_time,
                {
                    "type": "switch_off",
                    "nodes": nodes_to_switch_off,
                },
            )

        if (
            next_earliest_timeout is not None
            and self.next_timeout_at != next_earliest_timeout
        ):
            self._push_event(
                next_earliest_timeout,
                {"type": "call_me_later_to"},
            )
            self.next_timeout_at = next_earliest_timeout

    def _emit_compute_callbacks(
        self,
        scheduling_decision: SchedulingDecision,
        sleeping_nodes: list[int],
    ) -> None:
        for node_id in sleeping_nodes:
            lead_time = self._wake_lead_time(node_id)

            for finish_time in scheduling_decision.compute_finish_times:
                callback_time = (
                    float(finish_time)
                    - lead_time
                )

                if callback_time < self.current_time:
                    continue

                if callback_time in self._call_me_later_set:
                    continue

                self._push_event(
                    callback_time,
                    {"type": "CALL_ME_LATER"},
                )
                self.call_me_laters.append(callback_time)
                self._call_me_later_set.add(callback_time)

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(
            scheduling_decision,
            SchedulingDecision,
        ):
            raise TypeError(
                "scheduling_decision must be "
                "a SchedulingDecision"
            )

        self.current_time = float(current_time)
        self.events = []

        sleeping_nodes = self._emit_wake_triggers(
            scheduling_decision
        )
        self._emit_timeout_policy(
            scheduling_decision
        )
        self._emit_compute_callbacks(
            scheduling_decision,
            sleeping_nodes,
        )

        return tuple(self.events)
