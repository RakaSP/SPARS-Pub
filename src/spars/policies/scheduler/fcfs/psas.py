import heapq
from typing import TYPE_CHECKING

from spars.core.decision import PlannedJob, SchedulingDecision
from spars.core.registry import validate_component_config

from ..base.psas import BasePSAS

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager

EPS = 1e-9


class FCFSPSAS(BasePSAS):
    """First come, first served; plans with the node power states (pair with a power manager that switches nodes)."""
    DEFAULTS = {}  # no settings

    supports_plan_skip = True

    def __init__(self, machines: "Machine", jobs_manager: "JobsManager", start_time, config):
        validate_component_config(config, type(self).__name__, set())
        super().__init__(machines, jobs_manager, start_time)
        self.selected_list = []

    def _decision_from_plan(self, plan) -> SchedulingDecision:
        self.selected_list = list(plan)
        planned_jobs = tuple(
            PlannedJob(
                job=job,
                nodes=tuple(nodes),
                start_time=float(start_time),
                finish_time=float(finish_time),
            )
            for job, nodes, start_time, finish_time in self.selected_list
        )
        return SchedulingDecision(
            events=tuple(self.events),
            future_plan=planned_jobs,
            compute_finish_times=self._compute_finish_times(),
        )

    def schedule(self) -> SchedulingDecision:
        super().prep_schedule()
        now = float(self.current_time)

        started_now = self._current_fcfs_commit()
        remaining = [
            job
            for job in self.waiting_queue
            if job["job_id"] not in started_now
        ]
        future_plan = self._future_fcfs_plan(remaining, barrier=now)
        return self._decision_from_plan(future_plan)

    def _select_nodes_energy_aware_prepared(
        self,
        required_nodes,
        candidates,
        release_times,
        min_start_time,
        node_static_data,
    ):
        """Select nodes using prevalidated candidates and float releases."""
        if len(candidates) < required_nodes:
            return None

        candidate_time = max(
            float(min_start_time),
            heapq.nsmallest(
                required_nodes,
                (release_times[nid] for nid in candidates),
            )[-1],
        )

        eligible = []
        for node_id in candidates:
            release_time = release_times[node_id]
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

            eligible.append((
                float(cost),
                data["state_priority"],
                data["timeout_priority"],
                node_id,
            ))

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

    def _current_fcfs_commit(self):
        now = float(self.current_time)
        started_now = set()
        idle_candidates = list(self.idle)
        node_selection_static = self._build_node_selection_static_data(
            idle_candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in idle_candidates
            if nid in node_selection_static
        }
        idle_candidates = [
            nid for nid in idle_candidates if nid in release_times
        ]

        for job in self.waiting_queue:
            req = int(job["res"])
            if req <= 0:
                continue

            # FCFS: if head cannot start now, stop.
            if len(idle_candidates) < req:
                break

            res = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=idle_candidates,
                release_times=release_times,
                min_start_time=now,
                node_static_data=node_selection_static,
            )
            if res is None:
                break

            nodes, st = res
            if float(st) > now + EPS:
                break

            super().allocate(job, nodes)
            started_now.add(job["job_id"])

            selected_set = set(nodes)
            idle_candidates = [
                nid
                for nid in idle_candidates
                if nid not in selected_set
            ]

        return started_now

    def _future_fcfs_plan(self, jobs, barrier):
        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        node_selection_static = self._build_node_selection_static_data(
            candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in candidates
            if nid in node_selection_static
        }
        candidates = [
            nid for nid in candidates if nid in release_times
        ]

        plan = []
        barrier = float(barrier)
        state = self.state

        for job in jobs:
            req = int(job["res"])
            if req <= 0:
                continue

            res = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=candidates,
                release_times=release_times,
                min_start_time=barrier,
                node_static_data=node_selection_static,
            )
            if res is None:
                break

            nodes, st = res
            st = float(st)
            compute_speed = min(
                float(state[nid]["compute_speed"])
                for nid in nodes
            )
            ft = st + (float(job["reqtime"]) / compute_speed)
            plan.append((job, nodes, st, float(ft)))

            barrier = st

        return plan

    def _remaining_idle_timeout(self, node_id):
        return self.power_state.remaining_idle_timeout(
            node_id=node_id,
            current_time=self.current_time,
        )
