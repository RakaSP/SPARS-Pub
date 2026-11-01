from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from spars.core.decision import SchedulingDecision

from spars.policies.power_management.timeout.policy import TimeoutPowerManager

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine


class RLPowerManager(TimeoutPowerManager):
    """Combine scheduler-required power actions with the RL target."""
    DEFAULTS = {}  # no settings

    def __init__(
        self,
        machines: "Machine",
        start_time: float,
        config: dict[str, Any],
    ) -> None:
        if config:
            raise ValueError(
                "Invalid RL controller power-manager config: "
                f"unexpected keys: {sorted(config)}"
            )

        super().__init__(
            machines=machines,
            start_time=start_time,
            config={"timeout": None},
        )
        self._latest_decision: SchedulingDecision | None = None
        self._mandatory_wake_nodes: set[int] = set()
        # Nodes the agent chose to keep switched off. A scheduler that wakes
        # nodes on demand is refused these until the agent decides otherwise.
        self._held_off: set[int] = set()

    def prepare(self, current_time: float) -> None:
        super().prepare(current_time)
        self._held_off = {
            node_id
            for node_id in self._held_off
            if self.state[node_id]["state"] in ("switching_off", "sleeping")
        }

    def allow_wake(self, node_ids: Sequence[int], current_time: float) -> list[int]:
        return [node_id for node_id in node_ids if node_id not in self._held_off]

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> tuple[dict[str, Any], ...]:
        self._latest_decision = scheduling_decision
        events = super().decide(current_time, scheduling_decision)
        self._mandatory_wake_nodes = {
            node_id
            for event_bucket in events
            for event in event_bucket["events"]
            if event["type"] == "switch_on"
            for node_id in event["nodes"]
        }
        return events

    def _state_power(self, node_id: int, state_name: str) -> float:
        node = self.state[node_id]
        node_name = node["node_name"]
        state = self.machines.node_specs[node_name]["states"][state_name]
        power = state["power"]
        if power == "from_dvfs":
            profile = self.machines.node_specs[node_name]["dvfs_profiles"][
                node["dvfs_mode"]
            ]
            power = profile["power_idle"]
        return float(power)

    def _shutdown_savings(
        self,
        node_id: int,
        next_start: float | None,
        current_time: float,
    ) -> float | None:
        if next_start is None:
            # The current scheduler plan has no future use for this node.
            # Let the policy decide whether to keep it active; rank it ahead
            # of nodes with a finite planned idle interval when the policy
            # requests a lower active-node target.
            return math.inf

        idle_interval = next_start - current_time
        if idle_interval <= 0:
            return None

        off_duration = self._transition_time(
            node_id, "active", "switching_off"
        ) + self._transition_time(node_id, "switching_off", "sleeping")
        wake_duration = self._wake_lead_time(node_id)
        if off_duration + wake_duration > idle_interval:
            return None

        sleeping_duration = idle_interval - off_duration - wake_duration
        shutdown_energy = (
            self._state_power(node_id, "switching_off") * off_duration
            + self._state_power(node_id, "sleeping") * sleeping_duration
            + self._state_power(node_id, "switching_on") * wake_duration
        )
        idle_energy = self._state_power(node_id, "active") * idle_interval
        if shutdown_energy < idle_energy:
            return idle_energy - shutdown_energy
        return None

    def get_node_action_mask(
        self,
        current_time: float,
    ) -> dict[str, set[int]]:
        if self._latest_decision is None:
            raise RuntimeError(
                "RL controller cannot mask actions before the scheduler has planned"
            )

        planned_starts: dict[int, float] = {}
        for planned_job in self._latest_decision.future_plan:
            for node_id in planned_job.nodes:
                planned_starts[node_id] = min(
                    planned_starts.get(node_id, math.inf),
                    float(planned_job.start_time),
                )

        reserved_nodes = self._allocated_nodes(self._latest_decision)
        switch_off: set[int] = set()
        switch_on: set[int] = set()
        for node_id, node in self.state.items():
            if node["state"] == "sleeping":
                if node_id not in self._mandatory_wake_nodes:
                    switch_on.add(node_id)
                continue

            if node["state"] != "active" or node["job_id"] is not None:
                continue

            next_start = planned_starts.get(node_id)
            if node_id in reserved_nodes:
                continue

            if (
                self._shutdown_savings(
                    node_id,
                    next_start,
                    current_time,
                )
                is not None
            ):
                switch_off.add(node_id)

        return {
            "switch_off": switch_off,
            "switch_on": switch_on,
        }

    def active_target_bounds(self, current_time: float) -> tuple[int, int]:
        mask = self.get_node_action_mask(current_time)
        eventual = sum(
            node["state"] in ("active", "switching_on") for node in self.state.values()
        ) + sum(
            self.state[n]["state"] == "sleeping" for n in self._mandatory_wake_nodes
        )
        return (eventual - len(mask["switch_off"]), eventual + len(mask["switch_on"]))

    def apply_rl_action(
        self,
        action: Any,
        current_time: float,
    ) -> tuple[dict[str, Any], ...]:
        if self._latest_decision is None:
            raise RuntimeError(
                "RL controller cannot apply an action before the scheduler has planned"
            )

        if isinstance(action, Mapping):
            return self._apply_node_actions(action, current_time)

        try:
            target_active = int(action.item())
        except AttributeError:
            target_active = int(action)

        if target_active < 0:
            raise ValueError("RL controller active-node target cannot be negative")
        if target_active > len(self.state):
            raise ValueError(
                "RL controller active-node target exceeds the machine node count"
            )

        planned_starts: dict[int, float] = {}
        reserved_nodes = self._allocated_nodes(self._latest_decision)
        for planned_job in self._latest_decision.future_plan:
            for node_id in planned_job.nodes:
                planned_starts[node_id] = min(
                    planned_starts.get(node_id, math.inf),
                    float(planned_job.start_time),
                )

        active_nodes = [
            node_id for node_id, node in self.state.items() if node["state"] == "active"
        ]
        eligible_to_power_off: list[tuple[float, int]] = []
        for node_id in active_nodes:
            node = self.state[node_id]
            if node["job_id"] is not None:
                continue

            next_start = planned_starts.get(node_id)
            if node_id in reserved_nodes:
                continue
            savings = self._shutdown_savings(
                node_id,
                next_start,
                current_time,
            )
            if savings is None:
                continue
            eligible_to_power_off.append((savings, node_id))

        eligible_to_power_off.sort(key=lambda item: (-item[0], item[1]))
        switching_on_count = sum(
            node["state"] == "switching_on" for node in self.state.values()
        )
        mandatory_sleeping_count = sum(
            self.state[node_id]["state"] == "sleeping"
            for node_id in self._mandatory_wake_nodes
        )
        minimum_active_target = (
            len(active_nodes)
            - len(eligible_to_power_off)
            + mandatory_sleeping_count
            + switching_on_count
        )
        effective_target = max(target_active, minimum_active_target)

        events: list[dict[str, Any]] = []
        nodes_to_power_off = min(
            max(
                0,
                len(active_nodes)
                + switching_on_count
                + mandatory_sleeping_count
                - effective_target,
            ),
            len(eligible_to_power_off),
        )
        if nodes_to_power_off:
            events.append(
                {
                    "type": "switch_off",
                    "nodes": [
                        node_id
                        for _, node_id in eligible_to_power_off[:nodes_to_power_off]
                    ],
                }
            )

        expected_active = (
            len(active_nodes) + switching_on_count + mandatory_sleeping_count
        )
        nodes_to_power_on = max(0, effective_target - expected_active)
        sleeping_candidates = [
            node_id
            for node_id, node in self.state.items()
            if node["state"] == "sleeping" and node_id not in self._mandatory_wake_nodes
        ]
        if nodes_to_power_on and sleeping_candidates:
            events.append(
                {
                    "type": "switch_on",
                    "nodes": sleeping_candidates[:nodes_to_power_on],
                }
            )

        if not events:
            return ()

        return (
            {
                "timestamp": float(current_time),
                "events": events,
            },
        )

    def _apply_node_actions(
        self,
        action: Mapping[str, Any],
        current_time: float,
    ) -> tuple[dict[str, Any], ...]:
        allowed_keys = {"switch_off", "switch_on"}
        unexpected_keys = set(action) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Invalid RL controller node-action keys: {sorted(unexpected_keys)}"
            )

        node_actions: dict[str, set[int]] = {}
        for action_name in allowed_keys:
            node_ids = action.get(action_name, ())
            if isinstance(node_ids, (str, bytes)) or not isinstance(node_ids, Sequence):
                raise TypeError(
                    f"RL controller {action_name} action must be a sequence of node IDs"
                )

            selected_nodes = set(node_ids)
            unknown_nodes = selected_nodes - set(self.state)
            if unknown_nodes:
                raise ValueError(
                    f"RL controller {action_name} action contains unknown nodes: "
                    f"{sorted(unknown_nodes)}"
                )
            node_actions[action_name] = selected_nodes

        switch_off = node_actions["switch_off"]
        switch_on = node_actions["switch_on"]
        conflicting_nodes = switch_off & switch_on
        if conflicting_nodes:
            raise ValueError(
                "RL controller action cannot switch nodes both on and off: "
                f"{sorted(conflicting_nodes)}"
            )

        action_mask = self.get_node_action_mask(current_time)
        nodes_to_power_off = sorted(switch_off & action_mask["switch_off"])
        nodes_to_power_on = sorted(switch_on & action_mask["switch_on"])

        # Sleeping nodes the agent could have woken but did not, and the nodes
        # it switches off now, stay off until its next decision.
        self._held_off = (
            action_mask["switch_on"] - set(nodes_to_power_on)
        ) | set(nodes_to_power_off)

        events: list[dict[str, Any]] = []
        if nodes_to_power_off:
            events.append(
                {
                    "type": "switch_off",
                    "nodes": nodes_to_power_off,
                }
            )
        if nodes_to_power_on:
            events.append(
                {
                    "type": "switch_on",
                    "nodes": nodes_to_power_on,
                }
            )

        if not events:
            return ()

        return (
            {
                "timestamp": float(current_time),
                "events": events,
            },
        )
