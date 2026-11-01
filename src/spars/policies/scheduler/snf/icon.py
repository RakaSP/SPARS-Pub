from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Dict, Mapping, Sequence

from spars.core.decision import PlannedJob, SchedulingDecision
from spars.core.registry import config_with_defaults

from spars.policies.shared.icon_model import ICONForecast, ICONWorkloadModel, NodePartition
from .psas import SNFPSAS

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager
    from spars.core.simulator.monitor import Monitor
    from spars.core.trigger import PlanContext


_COMPUTE_RE = re.compile(r"^compute\(job=(\d+)\)$")
EPS = 1e-9


class SNFICON(SNFPSAS):
    """Shortest-node-first scheduler for the ICON spare-capacity policy.

    It starts the SNF-ordered jobs that fit now and plans the queued ones with
    runtimes predicted by its workload model. When recent arrivals and service
    times do not look Markovian (and the fallback is enabled) it plans like the
    plain SNFPSAS instead. Each decision carries an :class:`ICONForecast` for
    the ICON power manager, which makes every power decision.
    """

    DEFAULTS = {
        "arrival_rate": None,
        "service_rate": None,
        "resource_pmf": None,
        "initial_arrival_rate": 1.0 / 3600.0,
        "arrival_ema_alpha": 0.2,
        "service_ema_alpha": 0.1,
        "resource_ema_alpha": 0.1,
        "runtime_log_sigma_floor": 0.25,
        "max_decision_horizon": 24.0 * 3600.0,
        "markov_fallback_enabled": True,
        "markov_window_seconds": 24.0 * 3600.0,
        "markov_min_samples": 8,
        "markov_max_samples": 1000,
        "markov_cv_tolerance": 0.45,
        "markov_max_lag1_autocorrelation": 0.35,
        "markov_ks_multiplier": 1.25,
        "markov_require_service_samples": True,
        "markov_ks_param": 3.0,
    }

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        start_time,
        monitor: "Monitor",
        config,
    ):
        config = config_with_defaults(config, type(self).__name__, self.DEFAULTS)

        super().__init__(
            machines,
            jobs_manager,
            start_time,
            {},
        )

        if monitor is None:
            raise ValueError("monitor must not be None")
        self.monitor = monitor

        self.markov_fallback_enabled = bool(config.pop("markov_fallback_enabled"))
        self.model = ICONWorkloadModel(self, **config)

        self.arrives_count = 0
        self.completion_count = 0

        # Local visibility for jobs allocated during the current schedule call.
        self._started_this_schedule: Dict[int, Dict[str, object]] = {}

        self._first_schedule = True
        self.selected_list = []

    # ------------------------------------------------------------------
    # Public scheduling entry point
    # ------------------------------------------------------------------

    def prep_schedule(self):
        super().prep_schedule()

        arrival_log = self._monitor_log("jobs_arrival_log")
        completion_log = self._monitor_log("jobs_execution_log")

        arrival_total = len(arrival_log)
        completion_total = len(completion_log)

        if arrival_total < self.arrives_count:
            raise RuntimeError(
                "monitor.jobs_arrival_log shrank from "
                f"{self.arrives_count} to {arrival_total}"
            )
        if completion_total < self.completion_count:
            raise RuntimeError(
                "monitor.jobs_execution_log shrank from "
                f"{self.completion_count} to {completion_total}"
            )

        new_arrival_count = arrival_total - self.arrives_count
        new_completion_count = completion_total - self.completion_count

        # Save the new records for diagnostics
        new_arrival_records = list(arrival_log[self.arrives_count:arrival_total])
        new_completion_records = list(
            completion_log[self.completion_count:completion_total]
        )

        self.arrives_count = arrival_total
        self.completion_count = completion_total

        return {
            "new_arrival": new_arrival_count > 0,
            "new_completion": new_completion_count > 0,
            "new_arrival_count": new_arrival_count,
            "new_completion_count": new_completion_count,
            "new_arrival_records": new_arrival_records,
            "new_completion_records": new_completion_records,
        }

    def needs_replan(self, ctx: "PlanContext") -> bool:
        # New arrivals/completions must go through schedule():
        # EMA estimator, decision_event and stale-completion shutdown.
        return (
            len(self._monitor_log("jobs_arrival_log")) != self.arrives_count
            or len(self._monitor_log("jobs_execution_log"))
            != self.completion_count
        )

    def schedule(self):
        monitor_events = self.prep_schedule()
        now = float(self.current_time)
        idle_before_commit = list(self.idle)

        new_arrival_records = list(monitor_events["new_arrival_records"])
        new_completion_records = list(monitor_events["new_completion_records"])
        self.model.observe_arrivals(new_arrival_records, now)
        self.model.observe_completions(new_completion_records)

        markovianity = self.model.recent_markovianity(now)
        if (
            self.markov_fallback_enabled
            and not markovianity["well_markovian"]
        ):
            return self._schedule_snf_fallback(
                monitor_events=monitor_events,
                markovianity=markovianity,
                idle_before_commit=idle_before_commit,
            )

        decision_event = (
            self._first_schedule
            or bool(monitor_events["new_arrival"])
            or bool(monitor_events["new_completion"])
        )

        self._started_this_schedule = {}
        started_now, allocated_now = self._current_snf_commit()
        remaining_jobs = [
            job
            for job in self.waiting_queue
            if job.get("job_id") not in started_now
        ]

        # Plan queued jobs sequentially; the power manager wakes their nodes.
        self.selected_list = (
            list(self._future_snf_plan(remaining_jobs, barrier=now))
            if remaining_jobs
            else []
        )

        self._first_schedule = False
        return self._as_scheduling_decision(
            ICONForecast(
                nodes=self._node_partition(),
                next_releases=self.next_releases,
                model=self.model,
                mode="markov_power",
                markov_fallback_enabled=self.markov_fallback_enabled,
                markovianity=markovianity,
                monitor_events=monitor_events,
                idle_before_commit=idle_before_commit,
                started_now=started_now,
                allocated_now=allocated_now,
                remaining_jobs=remaining_jobs,
                decision_event=decision_event,
                plan=list(self.selected_list),
            )
        )

    def _schedule_snf_fallback(
        self,
        *,
        monitor_events: Mapping[str, object],
        markovianity: Mapping[str, object],
        idle_before_commit: Sequence[int],
    ):
        """Plan like the baseline SNFPSAS.

        Queue ordering, future planning and node choice follow
        :class:`SNFPSAS`; the power manager skips all speculative Markov
        spare actions for this decision.
        """
        now = float(self.current_time)
        self._started_this_schedule = {}

        started_now = SNFPSAS._current_snf_commit(self)
        allocated_now: Dict[int, int] = {}

        for bucket in self.events:
            if abs(float(bucket["timestamp"]) - now) > EPS:
                continue
            for event in bucket.get("events", []):
                if event.get("type") != "execution_start":
                    continue
                job_id = int(event["job_id"])
                nodes = [int(node_id) for node_id in event.get("nodes", [])]
                allocated_now.update({node_id: job_id for node_id in nodes})
                self._started_this_schedule[job_id] = {
                    "job_id": job_id,
                    "res": int(event.get("res", len(nodes))),
                    "reqtime": float(event.get("reqtime", 0.0)),
                    "start_time": now,
                    "nodes": list(nodes),
                }

        remaining_jobs = [
            job
            for job in self.waiting_queue
            if job.get("job_id") not in started_now
        ]
        future_plan = SNFPSAS._future_snf_plan(
            self,
            remaining_jobs,
            barrier=now,
        )
        self.selected_list = list(future_plan)

        decision_event = (
            self._first_schedule
            or bool(monitor_events["new_arrival"])
            or bool(monitor_events["new_completion"])
        )

        self._first_schedule = False
        return self._as_scheduling_decision(
            ICONForecast(
                nodes=self._node_partition(),
                next_releases=self.next_releases,
                model=self.model,
                mode="snf_fallback",
                markov_fallback_enabled=self.markov_fallback_enabled,
                markovianity=markovianity,
                monitor_events=monitor_events,
                idle_before_commit=list(idle_before_commit),
                started_now=started_now,
                allocated_now=allocated_now,
                remaining_jobs=remaining_jobs,
                decision_event=decision_event,
                plan=list(self.selected_list),
            )
        )

    def _node_partition(self) -> NodePartition:
        return NodePartition(
            idle=list(self.idle),
            sleeping=list(self.sleeping),
            switching_on=list(self.switching_on),
            switching_off=list(self.switching_off),
            computing=list(self.computing),
        )

    def _as_scheduling_decision(self, forecast: ICONForecast) -> SchedulingDecision:
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
            forecast=forecast,
        )

    def _monitor_log(self, attribute: str):
        if not hasattr(self.monitor, attribute):
            raise AttributeError(
                f"monitor must expose {attribute!r} for SNFPSAS event detection"
            )
        log = getattr(self.monitor, attribute)
        if log is None:
            raise ValueError(f"monitor.{attribute} must not be None")
        if not hasattr(log, "__len__") or not hasattr(log, "__getitem__"):
            raise TypeError(
                f"monitor.{attribute} must be a sequence-like object"
            )
        return log
    
    # Immediate scheduling and the plan for queued jobs
    
    @staticmethod
    def _job_key(job):
        return (
            int(job["res"]),
            float(job["reqtime"]),
            int(job.get("job_id", -1)),
        )

    def _current_snf_commit(self):
        # Start all SNF-ordered jobs that fit on currently idle nodes
        now = float(self.current_time)
        started_now = set()
        allocated_nodes = {}
        node_selection_static = self._build_node_selection_static_data(
            list(self.idle),
            self.next_releases,
        )

        for job in sorted(self.waiting_queue[:], key=self._job_key):
            required_nodes = int(job["res"])

            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive resource request: "
                    f"{required_nodes}"
                )
            if required_nodes > len(self.state):
                raise RuntimeError(
                    f"Job {job['job_id']} requests {required_nodes} nodes, "
                    f"but the platform has only {len(self.state)}"
                )

            # SNF sorts by node count.  If the smallest remaining job does not
            # fit, no later job can fit either.
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
            job_id = int(job["job_id"])
            started_now.add(job_id)
            self._started_this_schedule[job_id] = {
                "job_id": job_id,
                "res": required_nodes,
                "reqtime": float(job["reqtime"]),
                "start_time": float(start_time),
                "nodes": list(nodes),
            }
            for node_id in nodes:
                allocated_nodes[node_id] = job_id

        return started_now, allocated_nodes

    def _future_snf_plan(self, jobs, barrier):
        # BasePSAS release bounds can be based on requested wall time.  Replace
        # running compute releases with this policy's job-specific learned
        # runtime predictions before planning queued work.
        planned_releases = self.model.prediction_release_map()

        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        # During one future plan, only compute segments and release times are
        # appended to ``planned_releases``.  Compute segments are intentionally
        # excluded from energy-waste cost, while node state, idle power, and
        # timeout priority remain unchanged.  Precompute those invariant values
        # once instead of rebuilding them for every queued job.
        node_selection_static = self._build_node_selection_static_data(
            candidates,
            planned_releases,
        )

        plan = []
        barrier = float(barrier)

        for job in sorted(jobs, key=self._job_key):
            required_nodes = int(job["res"])
            if required_nodes <= 0:
                raise RuntimeError(
                    f"Job {job['job_id']} has non-positive resource request: "
                    f"{required_nodes}"
                )
            if required_nodes > len(self.state):
                raise RuntimeError(
                    f"Job {job['job_id']} requests {required_nodes} nodes, "
                    f"but the platform has only {len(self.state)}"
                )

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
            compute_speed = min(
                self.model.active_compute_speed(node_id)
                for node_id in nodes
            )
            if compute_speed <= 0.0:
                raise RuntimeError("compute_speed must be > 0")

            predicted_duration = self.model.predicted_job_duration(
                job,
                compute_speed=compute_speed,
            )
            finish_time = float(start_time) + predicted_duration
            phase = f'compute(job={job["job_id"]})'

            for node_id in nodes:
                entry = planned_releases[node_id]
                entry["queue"].append(
                    {
                        "phase": phase,
                        "start_time": float(start_time),
                        "finish_time": float(finish_time),
                    }
                )
                entry["release_time"] = float(finish_time)

            plan.append(
                (job, list(nodes), float(start_time), float(finish_time))
            )
            barrier = float(start_time)

        return plan

    # ------------------------------------------------------------------
    # Node power, for energy-aware node selection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Existing energy-aware idle-node selector
    # ------------------------------------------------------------------

    def _remaining_idle_timeout(self, node_id):
        return self.power_state.remaining_idle_timeout(
            node_id,
            self.current_time,
        )

    def _build_node_selection_static_data(self, candidates, release_map):
        """Precompute selector values that stay fixed during one plan.

        Future planning appends only ``compute(job=...)`` segments and updates
        release times.  Compute segments do not contribute to non-compute
        energy waste, so the values returned here remain valid for every job
        in the same future plan.
        """
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
            machine = self._machine_for(node_id)

            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
                state_priority = 0
            elif node["state"] == "active" and node.get("job_id") is not None:
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
                duration = max(0.0, duration)

                energy_rate = machine["states"][phase]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine=machine,
                        node=node,
                        power_type="idle",
                    )
                base_energy_waste += float(energy_rate) * duration

            static_data[node_id] = {
                "base": float(base_energy_waste),
                "idle": self._idle_power(node_id),
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
        candidates,
        min_start_time=None,
        release_map=None,
        node_static_data=None,
    ):
        if release_map is None:
            release_map = self.next_releases

        candidates = [
            node_id
            for node_id in candidates
            if (
                node_id in release_map
                and not math.isinf(
                    float(release_map[node_id]["release_time"])
                )
            )
        ]
        if len(candidates) < required_nodes:
            return None

        min_start_time = (
            -math.inf if min_start_time is None else float(min_start_time)
        )

        if node_static_data is None:
            node_static_data = self._build_node_selection_static_data(
                candidates,
                release_map,
            )

        # The original implementation checked every unique release time until
        # at least ``required_nodes`` nodes were eligible.  That first feasible
        # time is exactly max(min_start_time, r-th earliest release), where r is
        # ``required_nodes``.  Computing it directly preserves the same start
        # time while avoiding repeated full-node scans.
        available_nodes = [
            node_id
            for node_id in candidates
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
            release_time = float(release_map[node_id]["release_time"])
            if release_time > candidate_time:
                continue

            data = node_static_data[node_id]
            if data["state_label"] in ("switching_off", "sleeping"):
                cost = data["base"]
            else:
                cost = data["base"] + data["idle"] * (
                    candidate_time - release_time
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
            for item in sorted(eligible)[:required_nodes]
        ]
        return selected_nodes, float(candidate_time)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

