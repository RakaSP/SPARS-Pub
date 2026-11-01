# SNFPSAS
import math
import re
from typing import TYPE_CHECKING, Any

from spars.core.registry import validate_component_config

from ..base.psas import BasePSAS
from spars.core.decision import PlannedJob, SchedulingDecision

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager


_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


class SNFPSAS(BasePSAS):
    """Smallest node request first; plans with the node power states (pair with a power manager that switches nodes)."""
    DEFAULTS = {}  # no settings

    supports_plan_skip = True

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        start_time: float,
        config: dict[str, Any],
    ) -> None:
        validate_component_config(config, type(self).__name__, set())
        super().__init__(
            machines,
            jobs_manager,
            start_time,
        )

        self.selected_list = []

    @staticmethod
    def _job_key(job: dict[str, Any]) -> tuple[int, float]:
        return (
            int(job["res"]),
            float(job["reqtime"]),
        )

    def schedule(self) -> SchedulingDecision:
        super().prep_schedule()

        now = float(self.current_time)

        started_now = (
            self._current_snf_commit()
        )

        remaining = [
            job
            for job in self.waiting_queue
            if job["job_id"]
            not in started_now
        ]

        future_plan = (
            self._future_snf_plan(
                remaining,
                barrier=now,
            )
        )

        self.selected_list = list(
            future_plan
        )

        planned_jobs = tuple(
            PlannedJob(
                job=job,
                nodes=tuple(nodes),
                start_time=float(
                    start_time
                ),
                finish_time=float(
                    finish_time
                ),
            )
            for (
                job,
                nodes,
                start_time,
                finish_time,
            ) in self.selected_list
        )

        return SchedulingDecision(
            events=tuple(self.events),
            future_plan=planned_jobs,
            compute_finish_times=(
                self._compute_finish_times()
            ),
        )
    # ---------------- current SNF ----------------

    def _current_snf_commit(self) -> set[int]:
        now = float(self.current_time)
        started_now = set()
        node_selection_static = self._build_node_selection_static_data(
            list(self.idle),
            self.next_releases,
        )

        for job in sorted(
            self.waiting_queue[:],
            key=self._job_key,
        ):
            required_nodes = int(job["res"])

            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive "
                    f"resource request: {required_nodes}"
                )

            if len(self.idle) < required_nodes:
                break

            result = self._select_nodes_energy_aware(
                required_nodes=required_nodes,
                candidates=list(self.idle),
                min_start_time=now,
                node_static_data=node_selection_static,
            )

            if result is None:
                break

            nodes, start_time = result

            if float(start_time) > now + EPS:
                break

            super().allocate(job, nodes)
            started_now.add(job["job_id"])

        return started_now

    # ---------------- future SNF ----------------

    def _copy_release_plan(self) -> dict[int, dict[str, Any]]:
        return {
            node_id: {
                "queue": [
                    dict(segment)
                    for segment in entry["queue"]
                ],
                "release_time": float(entry["release_time"]),
            }
            for node_id, entry in self.next_releases.items()
        }

    def _append_planned_compute(
        self,
        planned_releases: dict[int, dict[str, Any]],
        job: dict[str, Any],
        selected_node_ids: list[int],
        job_start_time: float,
    ) -> float:
        compute_speed = min(
            float(self.state[node_id]["compute_speed"])
            for node_id in selected_node_ids
        )
        walltime = float(job["reqtime"]) / compute_speed
        finish_time = float(job_start_time) + walltime
        phase = f'compute(job={job["job_id"]})'

        for node_id in selected_node_ids:
            release_entry = planned_releases[node_id]
            release_entry["queue"].append({
                "phase": phase,
                "start_time": float(job_start_time),
                "finish_time": float(finish_time),
            })
            release_entry["release_time"] = float(finish_time)

        return float(finish_time)

    def _future_snf_plan(
        self,
        jobs: list[dict[str, Any]],
        barrier: float,
    ) -> list[tuple[dict[str, Any], list[int], float, float]]:
        planned_releases = self._copy_release_plan()
        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )
        node_selection_static = self._build_node_selection_static_data(
            candidates,
            planned_releases,
        )

        plan = []
        barrier = float(barrier)

        for job in sorted(jobs, key=self._job_key):
            required_nodes = int(job["res"])
            if required_nodes <= 0:
                continue

            result = self._select_nodes_energy_aware(
                required_nodes=required_nodes,
                candidates=candidates,
                min_start_time=barrier,
                release_map=planned_releases,
                node_static_data=node_selection_static,
            )
            if result is None:
                break

            nodes, start_time = result
            finish_time = self._append_planned_compute(
                planned_releases=planned_releases,
                job=job,
                selected_node_ids=nodes,
                job_start_time=float(start_time),
            )
            plan.append((
                job,
                nodes,
                float(start_time),
                float(finish_time),
            ))
            barrier = float(start_time)

        return plan

    # ---------------- internals ----------------
    def _remaining_idle_timeout(
        self,
        node_id: int,
    ) -> float:
        return (
            self.power_state
            .remaining_idle_timeout(
                node_id=node_id,
                current_time=self.current_time,
            )
        )

    def _build_node_selection_static_data(
        self,
        candidates: list[int],
        release_map: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
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
                duration = (
                    finish_time - current_time
                    if start_time < current_time
                    else finish_time - start_time
                )

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )
                base_energy_waste += float(energy_rate) * float(duration)

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
        required_nodes: int,
        candidates: list[int],
        min_start_time: float | None = None,
        release_map: dict[int, dict[str, Any]] | None = None,
        node_static_data: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[list[int], float] | None:
        if release_map is None:
            release_map = self.next_releases

        eligible_candidates = [
            node_id
            for node_id in candidates
            if (
                node_id in release_map
                and not math.isinf(
                    float(release_map[node_id]["release_time"])
                )
            )
        ]
        if len(eligible_candidates) < required_nodes:
            return None

        min_start_time = (
            -math.inf
            if min_start_time is None
            else float(min_start_time)
        )

        if node_static_data is None:
            node_static_data = self._build_node_selection_static_data(
                eligible_candidates,
                release_map,
            )

        available_nodes = [
            node_id
            for node_id in eligible_candidates
            if node_id in node_static_data
        ]
        if len(available_nodes) < required_nodes:
            return None

        release_times = sorted(
            float(release_map[node_id]["release_time"])
            for node_id in available_nodes
        )
        candidate_time = max(
            min_start_time,
            release_times[required_nodes - 1],
        )

        eligible = []
        for node_id in available_nodes:
            release_time = float(
                release_map[node_id]["release_time"]
            )
            if release_time > candidate_time:
                continue

            node_data = node_static_data[node_id]
            if node_data["state_label"] in (
                "switching_off",
                "sleeping",
            ):
                cost = node_data["base"]
            else:
                cost = (
                    node_data["base"]
                    + node_data["idle"]
                    * (candidate_time - release_time)
                )

            eligible.append(
                (
                    float(cost),
                    node_data["state_priority"],
                    node_data["timeout_priority"],
                    node_id,
                )
            )

        if len(eligible) < required_nodes:
            return None

        selected_nodes = [
            item[3]
            for item in sorted(eligible)[:required_nodes]
        ]
        return selected_nodes, float(candidate_time)
