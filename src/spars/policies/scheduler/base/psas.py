from __future__ import annotations

import heapq
import math
import re
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from spars.policies.shared.power import NodePower

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager
    from spars.core.trigger import PlanContext
    from collections.abc import Mapping
    from spars.core.power_manager import PowerManager


def _is_compute_phase(phase: Any) -> bool:
    value = str(phase)
    return (
        value.startswith("compute(job=")
        and value.endswith(")")
        and value[12:-1].isdigit()
    )
EPS = 1e-9
_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


class BasePSAS(NodePower):
    # Tolerance (in standard deviations) for switch on/off transitions that
    # finish earlier or later than planned. 0 = must match exactly (keeps
    # results identical). A value > 0 can change results.
    PLAN_TOLERANCE_SIGMA = 0.0

    # A scheduler that exports its plan through SchedulingDecision.future_plan
    # and has no side effects on repeated calls may enable this.
    # Default False: always replan.
    supports_plan_skip = False

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        start_time: float,
    ) -> None:
        self.machines = machines
        self.jobs_manager = jobs_manager

        self.state = machines.nodes
        self.machines_transitions = machines.transition_map
        self.waiting_queue = jobs_manager.waiting_queue

        self.scheduled_queue: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.current_time = float(start_time)
        self.power_state: PowerManager | None = None

        self.computing: list[int] = []
        self.idle: list[int] = []
        self.sleeping: list[int] = []
        self.switching_on: list[int] = []
        self.switching_off: list[int] = []
        self.selected_list: list[int] = []

        self.next_releases = {
            node_id: {
                "queue": [],
                "release_time": self.current_time,
            }
            for node_id in self.state
        }

    def set_power_state(self, power_state: PowerManager) -> None:
        self.power_state = power_state

    # ------------------------------------------------------------------
    # Plan trigger support
    # ------------------------------------------------------------------

    def needs_replan(self, ctx: "PlanContext") -> bool:
        return not self.supports_plan_skip

    def observe(self) -> None:
        """Sync the bookkeeping with the actual state without making a plan."""
        BasePSAS.prep_schedule(self)

    def _transition_std(
        self,
        node_id: int,
        from_state: str,
        to_state: str,
    ) -> float:
        node_name = self.state[node_id]["node_name"]
        transition = self.machines_transitions[node_name][
            (from_state, to_state)
        ]
        return float(transition.get("std", 0.0))

    def _within_transition_tolerance(
        self,
        node_id: int,
        phase: str,
        gap: float,
    ) -> bool:
        if phase == "switching_on":
            std = self._transition_std(node_id, "switching_on", "active")
        else:
            std = self._transition_std(node_id, "switching_off", "sleeping")
        return abs(gap) <= self.PLAN_TOLERANCE_SIGMA * std + EPS

    def _node_deviates(
        self,
        node_id: int,
        node: dict[str, Any],
        queue: list[dict[str, Any]],
        now: float,
        ignored_label: str | None,
    ) -> bool:
        actual = node["state"]
        job_id = node.get("job_id")
        if actual == "active":
            actual_label = "computing" if job_id is not None else "idle"
        else:
            actual_label = actual

        if ignored_label is not None and actual_label == ignored_label:
            return False

        active_phase = None
        for segment in queue:
            start = float(segment["start_time"])
            finish = float(segment["finish_time"])
            if start <= now + EPS and now < finish - EPS:
                active_phase = segment
                break

        if active_phase is not None:
            phase = str(active_phase["phase"])
            finish = float(active_phase["finish_time"])

            if _is_compute_phase(phase):
                return not (
                    actual_label == "computing"
                    and str(job_id) == phase[12:-1]
                )

            if actual_label == phase:
                return False

            # Transition finished earlier than planned, within tolerance.
            done_label = "idle" if phase == "switching_on" else "sleeping"
            if actual_label == done_label:
                return not self._within_transition_tolerance(
                    node_id, phase, finish - now
                )
            return True

        # No active phase: the node is expected idle or sleeping.
        if actual_label in ("idle", "sleeping"):
            return False

        # Transition finished later than planned, within tolerance.
        if actual_label in ("switching_on", "switching_off"):
            for segment in queue:
                if str(segment["phase"]) != actual_label:
                    continue
                finish = float(segment["finish_time"])
                if finish <= now + EPS:
                    return not self._within_transition_tolerance(
                        node_id, actual_label, now - finish
                    )
        return True

    def plan_deviation(
        self,
        current_time: float,
        ignored_nodes: "Mapping[int, str] | None" = None,
    ) -> bool:
        """True when the actual node state deviates from the last plan."""
        now = float(current_time)
        for node_id, node in self.state.items():
            entry = self.next_releases.get(node_id)
            queue = entry["queue"] if entry else []
            ignored_label = (
                ignored_nodes.get(node_id) if ignored_nodes else None
            )
            if self._node_deviates(
                node_id, node, queue, now, ignored_label
            ):
                return True
        return False

    def _compute_finish_times(self) -> tuple[float, ...]:
        finish_times = set()

        for release_entry in self.next_releases.values():
            for segment in release_entry["queue"]:
                if not _is_compute_phase(segment["phase"]):
                    continue

                finish_time = float(segment["finish_time"])
                if math.isinf(finish_time):
                    continue

                finish_times.add(finish_time)

        return tuple(sorted(finish_times))

    def push_event(
        self,
        timestamp: float,
        event: dict[str, Any],
    ) -> None:
        event_bucket = next(
            (
                bucket
                for bucket in self.events
                if bucket["timestamp"] == timestamp
            ),
            None,
        )

        if event_bucket:
            event_bucket["events"].append(event)
            return

        self.events.append({
            "timestamp": timestamp,
            "events": [event],
        })
        self.events.sort(key=itemgetter("timestamp"))

    def set_time(self, current_time: float) -> None:
        self.current_time = float(current_time)

    def _recalculate_release_at(
        self,
        release_entry: dict[str, Any],
    ) -> None:
        queue = release_entry["queue"]
        if not queue:
            release_entry["release_time"] = self.current_time
            return

        finish_time = float(queue[-1]["finish_time"])
        release_entry["release_time"] = (
            float("inf")
            if math.isinf(finish_time)
            else finish_time
        )

    @staticmethod
    def _append_phase_abs(
        release_entry: dict[str, Any],
        phase: str,
        start_time: float,
        duration: float,
    ) -> None:
        phase_start = float(start_time)
        phase_finish = phase_start + float(duration)

        release_entry["queue"].append({
            "phase": phase,
            "start_time": phase_start,
            "finish_time": phase_finish,
        })
        release_entry["release_time"] = phase_finish

    def _transition_time(
        self,
        node_id: int,
        from_state: str,
        to_state: str,
    ) -> float:
        node_name = self.state[node_id]["node_name"]
        transition = self.machines_transitions[node_name][
            (from_state, to_state)
        ]
        return float(transition["transition_time"])

    def _get_release_entry(
        self,
        node_id: int,
    ) -> dict[str, Any]:
        release_entry = self.next_releases.get(node_id)
        if release_entry is not None:
            return release_entry

        release_entry = {
            "queue": [],
            "release_time": self.current_time,
        }
        self.next_releases[node_id] = release_entry
        return release_entry

    def _retain_current_release_segments(
        self,
        release_entry: dict[str, Any],
        node: dict[str, Any],
    ) -> None:
        current_time = self.current_time
        active_segments = []

        for segment in release_entry["queue"]:
            segment_finish = float(segment["finish_time"])

            if _is_compute_phase(segment["phase"]):
                phase_job_id = int(str(segment["phase"])[12:-1])
                if (
                    current_time >= segment_finish
                    and node["state"] == "active"
                    and node.get("job_id") == phase_job_id
                ):
                    # A running job can outlive its planned finish time.
                    # Keep the reservation occupied until the real release.
                    segment["finish_time"] = float("inf")
                    active_segments.append(segment)
                    continue

            if segment_finish > current_time:
                active_segments.append(segment)

        release_entry["queue"] = active_segments

    def _release_queue_matches_node_state(
        self,
        release_entry: dict[str, Any],
        node: dict[str, Any],
    ) -> bool:
        if not release_entry["queue"]:
            return False

        first_phase = release_entry["queue"][0]
        current_time = self.current_time
        first_phase_active = (
            float(first_phase["start_time"])
            <= current_time
            < float(first_phase["finish_time"])
        )
        if not first_phase_active:
            return False

        phase = first_phase["phase"]
        node_state = node["state"]

        return (
            (
                phase == "switching_off"
                and node_state == "switching_off"
            )
            or (
                phase == "switching_on"
                and node_state == "switching_on"
            )
            or (
                _is_compute_phase(phase)
                and node_state == "active"
                and node.get("job_id") is not None
            )
        )

    def _rebuild_transition_queue(
        self,
        node_id: int,
        node: dict[str, Any],
        release_entry: dict[str, Any],
    ) -> None:
        queue = release_entry["queue"]
        current_time = self.current_time
        node_state = node["state"]

        if node_state == "switching_off":
            switching_off_duration = self._transition_time(
                node_id,
                "switching_off",
                "sleeping",
            )
            sleeping_duration = self._transition_time(
                node_id,
                "sleeping",
                "switching_on",
            )
            switching_on_duration = self._transition_time(
                node_id,
                "switching_on",
                "active",
            )
            if not any(
                segment["phase"] == "switching_off"
                for segment in queue
            ):
                queue.insert(0, {
                    "phase": "switching_off",
                    "start_time": current_time,
                    "finish_time": (
                        current_time + switching_off_duration
                    ),
                })

            switching_off_phase = next(
                (
                    segment
                    for segment in queue
                    if segment["phase"] == "switching_off"
                ),
                None,
            )
            if switching_off_phase:
                cursor = float(switching_off_phase["finish_time"])
                if not any(
                    segment["phase"] == "switching_on"
                    for segment in queue
                ):
                    switch_on_start = (
                        cursor + float(sleeping_duration)
                    )
                    self._append_phase_abs(
                        release_entry,
                        "switching_on",
                        switch_on_start,
                        switching_on_duration,
                    )

        elif node_state == "sleeping":
            sleeping_duration = self._transition_time(
                node_id,
                "sleeping",
                "switching_on",
            )
            switching_on_duration = self._transition_time(
                node_id,
                "switching_on",
                "active",
            )
            switch_on_start = (
                current_time + float(sleeping_duration)
            )
            self._append_phase_abs(
                release_entry,
                "switching_on",
                switch_on_start,
                switching_on_duration,
            )

        elif node_state == "switching_on":
            switching_on_duration = self._transition_time(
                node_id,
                "switching_on",
                "active",
            )
            if not any(
                segment["phase"] == "switching_on"
                for segment in queue
            ):
                queue.insert(0, {
                    "phase": "switching_on",
                    "start_time": current_time,
                    "finish_time": (
                        current_time + switching_on_duration
                    ),
                })

        elif node_state == "active" and node.get("job_id") is None:
            release_entry["queue"] = [
                segment
                for segment in queue
                if segment["phase"]
                not in ("switching_off", "switching_on")
            ]

    def _rebuild_next_releases_global(self) -> None:
        for node_id, node in self.state.items():
            release_entry = self._get_release_entry(node_id)
            self._retain_current_release_segments(
                release_entry,
                node,
            )

            if not self._release_queue_matches_node_state(
                release_entry,
                node,
            ):
                release_entry["queue"] = []

            self._rebuild_transition_queue(
                node_id,
                node,
                release_entry,
            )
            self._recalculate_release_at(release_entry)

    def allocate(
        self,
        job: dict[str, Any],
        allocated_nodes: list[int],
    ) -> None:
        if not allocated_nodes:
            return

        for node_id in allocated_nodes:
            if node_id not in self.idle:
                raise RuntimeError("Non-Idle node is allocated")

        allocated_node_set = set(allocated_nodes)
        self.idle = [
            node_id
            for node_id in self.idle
            if node_id not in allocated_node_set
        ]
        self.sleeping = [
            node_id
            for node_id in self.sleeping
            if node_id not in allocated_node_set
        ]
        self.switching_on = [
            node_id
            for node_id in self.switching_on
            if node_id not in allocated_node_set
        ]
        self.switching_off = [
            node_id
            for node_id in self.switching_off
            if node_id not in allocated_node_set
        ]
        self.computing.extend(allocated_nodes)

        compute_speed = min(
            float(self.state[node_id]["compute_speed"])
            for node_id in allocated_nodes
        )
        if compute_speed <= 0.0:
            raise RuntimeError("compute_speed must be > 0")

        walltime = float(job["reqtime"]) / compute_speed

        for node_id in allocated_nodes:
            release_entry = self.next_releases.get(node_id)
            if release_entry is None:
                raise RuntimeError(
                    f"next_releases entry missing for node {node_id}"
                )

            cursor = float(release_entry["release_time"])
            self._append_phase_abs(
                release_entry,
                f'compute(job={job["job_id"]})',
                cursor,
                walltime,
            )

        self.push_event(
            self.current_time,
            {
                **job,
                "type": "execution_start",
                "nodes": allocated_nodes,
            },
        )

    def _build_partitions(self) -> None:
        self.computing = []
        self.idle = []
        self.sleeping = []
        self.switching_on = []
        self.switching_off = []
        self.selected_list = []

        for node_id, node in self.state.items():
            node_state = node.get("state")
            job_id = node.get("job_id")

            if job_id is not None and node_state == "active":
                self.computing.append(node_id)
            elif node_state == "active" and job_id is None:
                self.idle.append(node_id)
            elif node_state == "sleeping":
                self.sleeping.append(node_id)
            elif node_state == "switching_on":
                self.switching_on.append(node_id)
            elif node_state == "switching_off":
                self.switching_off.append(node_id)

    def prep_schedule(self) -> None:
        self.events = []
        self._rebuild_next_releases_global()
        self._build_partitions()

    def _build_node_selection_static_data(
        self,
        candidates,
        release_map,
    ):
        current_time = float(self.current_time)
        static_data = {}

        for node_id in candidates:
            node_release = release_map.get(node_id)
            if node_release is None:
                continue

            release_time = float(node_release["release_time"])
            if math.isinf(release_time):
                continue

            node = self.state[node_id]
            node_name = self.machines.nodes[node_id]["node_name"]
            machine = self.machines.node_specs[node_name]

            if (
                node["state"] == "active"
                and node.get("job_id") is None
            ):
                state_label = "idle"
                state_priority = 0
            elif (
                node["state"] == "active"
                and node.get("job_id") is not None
            ):
                state_label = "computing"
                state_priority = 1
            elif node["state"] == "switching_on":
                state_label = "switching_on"
                state_priority = 2
            else:
                state_label = node["state"]
                state_priority = 3

            base_energy_waste = 0.0
            for queue_entry in node_release["queue"]:
                phase = str(queue_entry["phase"])
                if _COMPUTE_RE.fullmatch(phase):
                    continue

                start_time = float(queue_entry["start_time"])
                finish_time = float(queue_entry["finish_time"])
                if start_time < current_time:
                    duration = finish_time - current_time
                else:
                    duration = finish_time - start_time

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )

                base_energy_waste += (
                    float(energy_rate) * float(duration)
                )

            idle_power = machine["states"]["active"]["power"]
            if idle_power == "from_dvfs":
                idle_power = self._resolve_dvfs_power(
                    machine=machine,
                    node=node,
                    power_type="idle",
                )

            static_data[node_id] = {
                "base": float(base_energy_waste),
                "idle": float(idle_power),
                "state_label": state_label,
                "state_priority": int(state_priority),
                "timeout_priority": (
                    -self._remaining_idle_timeout(node_id)
                    if state_label == "idle"
                    else 0.0
                ),
            }

        return static_data

    def _select_nodes_energy_aware(
        self,
        required_nodes,
        _candidates,
        min_start_time=None,
        release_map=None,
        node_static_data=None,
    ):
        if release_map is None:
            release_map = self.next_releases

        candidates = [
            node_id
            for node_id in _candidates
            if (
                node_id in release_map
                and not math.isinf(
                    float(release_map[node_id]["release_time"])
                )
            )
        ]

        if len(candidates) < required_nodes:
            return None

        if min_start_time is None:
            min_start_time = -math.inf
        else:
            min_start_time = float(min_start_time)

        if node_static_data is None:
            node_static_data = self._build_node_selection_static_data(
                candidates,
                release_map,
            )

        available_nodes = [
            node_id
            for node_id in candidates
            if node_id in node_static_data
        ]
        if len(available_nodes) < required_nodes:
            return None

        earliest_releases = heapq.nsmallest(
            required_nodes,
            (
                float(release_map[node_id]["release_time"])
                for node_id in available_nodes
            ),
        )
        candidate_time = max(
            min_start_time,
            earliest_releases[-1],
        )

        eligible = []
        for node_id in available_nodes:
            release_time = float(
                release_map[node_id]["release_time"]
            )
            if release_time > candidate_time:
                continue

            data = node_static_data[node_id]
            if data["state_label"] in (
                "switching_off",
                "sleeping",
            ):
                cost = data["base"]
            else:
                cost = (
                    data["base"]
                    + data["idle"]
                    * (candidate_time - release_time)
                )

            eligible.append(
                (
                    float(cost),
                    data["state_priority"],
                    data["timeout_priority"],
                    node_id,
                )
            )

        if len(eligible) < required_nodes:
            return None

        selected_nodes = [
            item[3]
            for item in heapq.nsmallest(
                required_nodes,
                eligible,
            )
        ]
        return selected_nodes, float(candidate_time)
