from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spars.core.decision import SchedulingDecision
from spars.core.registry import validate_component_config

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.platform_control import PlatformControl
    from spars.core.trigger import PlanContext


@dataclass(frozen=True)
class NodeUse:
    start: float
    finish: float


@dataclass(frozen=True)
class IdleDecision:
    mode: str
    energy: float
    switch_on_at: float | None = None
    off_duration: float | None = None
    on_duration: float | None = None


class OraclePowerManager:
    """Sleep every idle gap that saves energy, knowing the complete plan.

    Pair with the SNFOraclePSAS scheduler: it plans every job up front and
    asks idle_decision() for the energy of each candidate node's idle gap.
    The plan in its first decision, from that decision's time on, becomes one
    timeline of every job start and power transition, which the simulator
    queues when it processes the timeline's "oracle" event.

    Transition durations are sampled once per node and idle episode, so the
    plan uses the durations the simulation will see.
    """
    DEFAULTS = {}  # no settings

    def __init__(
        self,
        machines: "Machine",
        start_time: float,
        platform_control: "PlatformControl",
        config: dict[str, Any],
    ) -> None:
        validate_component_config(config, type(self).__name__, set())
        if platform_control is None:
            raise ValueError("OraclePowerManager requires platform_control")

        self.machines = machines
        self.state = machines.nodes
        self.platform_control = platform_control
        self.current_time = float(start_time)
        self._plan_emitted = False
        self._off_duration_cache: dict[tuple[int, int], float] = {}
        self._on_duration_cache: dict[tuple[int, int], float] = {}

    def prepare(self, current_time: float) -> None:
        self.current_time = float(current_time)

    def needs_replan(self, ctx: "PlanContext") -> bool:
        return False

    def remaining_idle_timeout(
        self,
        node_id: int,
        current_time: float,
    ) -> float:
        del node_id, current_time
        return math.inf

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> tuple[dict[str, Any], ...]:
        self.current_time = float(current_time)
        if not isinstance(scheduling_decision, SchedulingDecision):
            raise TypeError(
                "scheduling_decision must be a SchedulingDecision"
            )
        if self._plan_emitted:
            return ()

        self._plan_emitted = True
        node_uses: dict[int, list[NodeUse]] = {
            int(node_id): [] for node_id in self.state
        }
        for planned in scheduling_decision.future_plan:
            for node_id in planned.nodes:
                node_uses[int(node_id)].append(
                    NodeUse(
                        start=float(planned.start_time),
                        finish=float(planned.finish_time),
                    )
                )

        plan = [
            {
                "timestamp": float(planned.start_time),
                "event": {
                    **planned.job,
                    "type": "execution_start",
                    "nodes": list(planned.nodes),
                },
            }
            for planned in scheduling_decision.future_plan
        ]
        plan += self._build_power_events(node_uses, self.current_time)
        plan.sort(key=self._planned_event_key)
        return (
            {
                "timestamp": float(self.current_time),
                "events": [{"type": "oracle", "plan": plan}],
            },
        )

    def _build_power_events(
        self,
        node_uses,
        planning_start,
    ):
        events = []

        for node_id in sorted(node_uses):
            uses = sorted(
                node_uses[node_id],
                key=lambda use: use.start,
            )

            idle_at = float(planning_start)
            episode_id = 0

            for use in uses:
                decision = self.idle_decision(
                    node_id=node_id,
                    idle_at=idle_at,
                    next_use_at=float(use.start),
                    episode_id=episode_id,
                )

                if decision.mode == "sleep":
                    events.append(
                        self._power_event(
                            timestamp=idle_at,
                            event_type="switch_off",
                            node_id=node_id,
                            duration=(
                                decision.off_duration
                            ),
                        )
                    )

                    events.append(
                        self._power_event(
                            timestamp=(
                                decision.switch_on_at
                            ),
                            event_type="switch_on",
                            node_id=node_id,
                            duration=(
                                decision.on_duration
                            ),
                        )
                    )

                idle_at = float(use.finish)
                episode_id += 1

            events.append(
                self._power_event(
                    timestamp=idle_at,
                    event_type="switch_off",
                    node_id=node_id,
                    duration=self._off_duration(
                        node_id,
                        episode_id,
                    ),
                )
            )

        return events

    def _sample_transition(
        self,
        node_id,
        from_state,
        to_state,
    ):
        node_name = self.state[node_id][
            "node_name"
        ]

        transition = (
            self.machines.transition_map[
                node_name
            ][
                (from_state, to_state)
            ]
        )

        return float(
            self.platform_control
            .sample_transition_time(transition)
        )

    def _off_duration(
        self,
        node_id,
        episode_id,
    ):
        key = (
            int(node_id),
            int(episode_id),
        )

        if key not in self._off_duration_cache:
            self._off_duration_cache[key] = (
                self._sample_transition(
                    node_id,
                    "active",
                    "switching_off",
                )
                + self._sample_transition(
                    node_id,
                    "switching_off",
                    "sleeping",
                )
            )

        return self._off_duration_cache[key]

    def _on_duration(
        self,
        node_id,
        episode_id,
    ):
        key = (
            int(node_id),
            int(episode_id),
        )

        if key not in self._on_duration_cache:
            self._on_duration_cache[key] = (
                self._sample_transition(
                    node_id,
                    "sleeping",
                    "switching_on",
                )
                + self._sample_transition(
                    node_id,
                    "switching_on",
                    "active",
                )
            )

        return self._on_duration_cache[key]

    def idle_decision(self, node_id, idle_at, next_use_at, episode_id):
        """Keep the node idle or sleep it until next_use_at, whichever costs less."""
        gap = max(0.0, float(next_use_at) - float(idle_at))
        idle_power = self._state_power(node_id, "active")
        keep_energy = idle_power * gap

        if gap <= 0.0:
            return IdleDecision(mode="idle", energy=keep_energy)

        off_duration = self._off_duration(node_id, episode_id)
        on_duration = self._on_duration(node_id, episode_id)

        if off_duration + on_duration > gap:
            return IdleDecision(mode="idle", energy=keep_energy)

        sleeping_duration = max(0.0, gap - off_duration - on_duration)
        sleep_energy = (
            self._state_power(node_id, "switching_off") * off_duration
            + self._state_power(node_id, "sleeping") * sleeping_duration
            + self._state_power(node_id, "switching_on") * on_duration
        )

        if sleep_energy >= keep_energy:
            return IdleDecision(mode="idle", energy=keep_energy)

        switch_on_at = float(next_use_at) - float(on_duration)
        while switch_on_at + float(on_duration) > float(next_use_at):
            switch_on_at = math.nextafter(switch_on_at, -math.inf)

        if switch_on_at < float(idle_at) + float(off_duration):
            return IdleDecision(mode="idle", energy=keep_energy)

        return IdleDecision(
            mode="sleep",
            energy=float(sleep_energy),
            switch_on_at=float(switch_on_at),
            off_duration=float(off_duration),
            on_duration=float(on_duration),
        )

    @staticmethod
    def _power_event(
        timestamp,
        event_type,
        node_id,
        duration,
    ):
        return {
            "timestamp": float(timestamp),
            "event": {
                "type": event_type,
                "nodes": [int(node_id)],
                "oracle_durations": {
                    int(node_id): float(duration),
                },
            },
        }

    def _state_power(
        self,
        node_id,
        state_name,
    ):
        node = self.state[node_id]

        machine = self.machines.node_specs[
            node["node_name"]
        ]

        power = machine["states"][
            state_name
        ]["power"]

        if power != "from_dvfs":
            return float(power)

        profile = machine["dvfs_profiles"][
            node["dvfs_mode"]
        ]

        if "power_idle" in profile:
            return float(
                profile["power_idle"]
            )

        if "power" in profile:
            return float(profile["power"])

        raise KeyError(
            f"DVFS profile "
            f"{node['dvfs_mode']!r} "
            f"lacks power_idle"
        )

    @staticmethod
    def _planned_event_key(item):
        event_type = item["event"]["type"]

        priority = {
            "execution_start": 0,
            "switch_off": 1,
            "switch_on": 2,
        }.get(event_type, 100)

        return (
            float(item["timestamp"]),
            priority,
        )
