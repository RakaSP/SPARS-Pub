from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Iterable

from spars.core.decision import PlannedJob, SchedulingDecision
from spars.core.registry import validate_component_config

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager
    from spars.core.simulator.platform_control import PlatformControl
    from typing import Any
    from spars.core.power_manager import PowerManager

EPS = 1e-9


class SNFOraclePSAS:
    """Shortest-node-first over the complete workload, known in advance.

    The whole schedule is planned at the first call and returned as the
    decision's future plan. Among free nodes, a job takes those whose idle gap
    before it costs the least energy, as judged by the oracle power manager,
    which then plans every power transition and runs the plan.
    """
    DEFAULTS = {}  # no settings

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        start_time,
        workload: "dict[str, Any]",
        platform_control: "PlatformControl",
        config,
    ):
        validate_component_config(config, type(self).__name__, set())
        if workload is None:
            raise ValueError(
                "SNFOraclePSAS requires the complete workload"
            )

        if platform_control is None:
            raise ValueError(
                "SNFOraclePSAS requires platform_control"
            )

        self.machines = machines
        self.jobs_manager = jobs_manager
        self.state = machines.nodes
        self.start_time = float(start_time)
        self.current_time = float(start_time)
        self.workload = workload
        self.platform_control = platform_control
        self.overrun_policy = platform_control.overrun_policy
        self.power_state: PowerManager | None = None

        self._oracle_emitted = False

    def set_time(self, current_time):
        self.current_time = float(current_time)

    def set_power_state(self, power_state: PowerManager) -> None:
        self.power_state = power_state

    def schedule(self) -> SchedulingDecision:
        if self._oracle_emitted:
            return SchedulingDecision()

        self._oracle_emitted = True
        return SchedulingDecision(
            future_plan=tuple(self._build_complete_plan()),
        )

    def _effective_runtime(self, job):
        runtime = float(job["runtime"])
        reqtime = float(job["reqtime"])

        if self.overrun_policy == "continue":
            return runtime

        if self.overrun_policy == "terminate":
            return min(runtime, reqtime)

        raise ValueError(
            f"Unsupported overrun policy: "
            f"{self.overrun_policy!r}"
        )

    def _job_key(self, job):
        return (
            int(job["res"]),
            self._effective_runtime(job),
            int(job["job_id"]),
        )

    def _build_complete_plan(self):
        planning_start = float(self.current_time)

        if self.jobs_manager.active_jobs:
            raise RuntimeError(
                "SNFOraclePSAS must build its plan "
                "before any job starts"
            )

        jobs = []

        for raw_job in self.workload["jobs"]:
            job = dict(raw_job)

            # Private arrival timestamp used only by oracle planning.
            # The arrival event itself is not added to the oracle plan.
            job["subtime"] = (
                self.start_time
                + float(raw_job["subtime"])
            )

            jobs.append(job)

        jobs.sort(
            key=lambda job: (
                float(job["subtime"]),
                int(job["job_id"]),
            )
        )

        return self._build_compute_schedule(
            jobs=jobs,
            planning_start=planning_start,
        )

    def _build_compute_schedule(
        self,
        jobs,
        planning_start,
    ):
        node_ids = sorted(
            int(node_id)
            for node_id in self.state
        )

        for node_id in node_ids:
            node = self.state[node_id]

            if (
                node["state"] != "active"
                or node.get("job_id") is not None
            ):
                raise RuntimeError(
                    "SNFOraclePSAS must build its plan "
                    "while all nodes are active and idle"
                )

        free_nodes = set(node_ids)

        node_idle_since = {
            node_id: float(planning_start)
            for node_id in node_ids
        }

        node_episode = {
            node_id: 0
            for node_id in node_ids
        }

        planned_jobs = []

        waiting = []
        arrival_index = 0
        active_heap = []
        serial = 0
        now = float(planning_start)

        while (
            arrival_index < len(jobs)
            or waiting
            or active_heap
        ):
            while (
                active_heap
                and active_heap[0][0] <= now + EPS
            ):
                finish_time, _, nodes = heapq.heappop(
                    active_heap
                )

                for node_id in nodes:
                    free_nodes.add(node_id)
                    node_idle_since[node_id] = float(
                        finish_time
                    )
                    node_episode[node_id] += 1

            while (
                arrival_index < len(jobs)
                and float(
                    jobs[arrival_index]["subtime"]
                ) <= now + EPS
            ):
                waiting.append(jobs[arrival_index])
                arrival_index += 1

            started_any = False

            while waiting:
                waiting.sort(key=self._job_key)
                job = waiting[0]
                required_nodes = int(job["res"])

                if required_nodes <= 0:
                    raise RuntimeError(
                        f"Job {job['job_id']} has "
                        f"invalid res={required_nodes}"
                    )

                if required_nodes > len(node_ids):
                    raise RuntimeError(
                        f"Job {job['job_id']} requests "
                        f"{required_nodes} nodes, but only "
                        f"{len(node_ids)} exist"
                    )

                if required_nodes > len(free_nodes):
                    break

                selected_nodes = (
                    self._select_nodes_for_job(
                        job=job,
                        candidate_nodes=free_nodes,
                        job_start=now,
                        node_idle_since=node_idle_since,
                        node_episode=node_episode,
                    )
                )

                compute_speed = min(
                    float(
                        self.state[node_id][
                            "compute_speed"
                        ]
                    )
                    for node_id in selected_nodes
                )

                if compute_speed <= 0:
                    raise RuntimeError(
                        "compute_speed must be "
                        "greater than zero"
                    )

                finish_time = (
                    now
                    + self._effective_runtime(job)
                    / compute_speed
                )

                for node_id in selected_nodes:
                    free_nodes.remove(node_id)

                planned_jobs.append(
                    PlannedJob(
                        job=dict(job),
                        nodes=tuple(selected_nodes),
                        start_time=float(now),
                        finish_time=float(finish_time),
                    )
                )

                serial += 1

                heapq.heappush(
                    active_heap,
                    (
                        float(finish_time),
                        serial,
                        tuple(selected_nodes),
                    ),
                )

                waiting.pop(0)
                started_any = True

            if started_any:
                continue

            next_times = []

            if arrival_index < len(jobs):
                next_times.append(
                    float(
                        jobs[arrival_index]["subtime"]
                    )
                )

            if active_heap:
                next_times.append(
                    float(active_heap[0][0])
                )

            if not next_times:
                if waiting:
                    raise RuntimeError(
                        "Oracle planning stopped "
                        "with unscheduled jobs"
                    )

                break

            next_time = min(next_times)

            if next_time < now - EPS:
                raise RuntimeError(
                    "Oracle virtual time moved backward"
                )

            now = float(next_time)

        return planned_jobs

    def _select_nodes_for_job(
        self,
        job,
        candidate_nodes: Iterable[int],
        job_start,
        node_idle_since,
        node_episode,
    ):
        ranked = []
        power_manager = self.power_state

        for node_id in candidate_nodes:
            decision = power_manager.idle_decision(
                node_id=int(node_id),
                idle_at=float(
                    node_idle_since[node_id]
                ),
                next_use_at=float(job_start),
                episode_id=int(
                    node_episode[node_id]
                ),
            )

            ranked.append(
                (
                    float(decision.energy),
                    int(node_id),
                )
            )

        ranked.sort()

        required_nodes = int(job["res"])

        return [
            node_id
            for _, node_id
            in ranked[:required_nodes]
        ]
