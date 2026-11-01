from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence

from spars.core.decision import SchedulingDecision
from spars.core.registry import config_with_defaults
from spars.policies.power_management.timeout.policy import TimeoutPowerManager
from spars.policies.shared.power import NodePower
from spars.policies.shared.icon_model import (
    ICONForecast,
    nonnegative_float,
    optional_nonnegative_float,
)

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.trigger import PlanContext
    import os


EPS = 1e-9


class ICONPowerManager(NodePower):
    """Power decisions of the ICON policy; pair with the SNFICON scheduler.

    Wakes the nodes the scheduler's plan needs, keeps the spare capacity that
    minimises ``alpha`` x expected power-induced wait + ``beta`` x expected
    energy until the next event (given the scheduler's workload model), shuts
    down nodes released while no job has arrived recently, applies the idle
    timeout, and writes the decision log.
    """

    DEFAULTS = {
        "alpha": 10000.0,
        "beta": 1.0,
        "min_spare_nodes": 1,
        "max_spare_nodes": None,
        "spare_arrival_window": 12.0 * 3600.0,
        "immediate_shutdown_on_stale_completion": True,
        "decision_log_enabled": True,
        "decision_log_path": None,
        "timeout": None,
        "respect_timeout": True,
    }

    # Same bookkeeping as the timeout policy: events, timeout_list and the
    # transition-time cache.
    push_event = TimeoutPowerManager._push_event
    _transition_time = TimeoutPowerManager._transition_time
    remaining_idle_timeout = TimeoutPowerManager.remaining_idle_timeout

    def __init__(
        self,
        machines: "Machine",
        start_time: float,
        config: dict[str, Any],
        output_dir: "str | os.PathLike[str] | None" = None,
    ) -> None:
        config = config_with_defaults(config, type(self).__name__, self.DEFAULTS)
        self.machines = machines
        self.state = machines.nodes
        self.transitions = machines.transition_map
        self.current_time = float(start_time)
        self.timeout = config["timeout"]
        self.respect_timeout = bool(config["respect_timeout"])

        self.timeout_list: dict[int, float] = {}
        self.to_be_switched_off: list[int] = []
        self.next_timeout_at: float | None = None
        self.events: list[dict[str, Any]] = []
        self._transition_time_cache: dict[tuple[int, str, str], float] = {}
        self._wake_lead_time_cache: dict[int, float] = {}

        # ``alpha`` and ``beta`` must put seconds and joules on comparable
        # scales. For example, if one second of waiting is considered
        # equivalent to 50 kJ, use ``alpha=1`` and ``beta=1 / 50_000``.
        self.alpha = nonnegative_float("alpha", config["alpha"])
        self.beta = nonnegative_float("beta", config["beta"])
        if self.alpha <= 0.0 and self.beta <= 0.0:
            raise ValueError("At least one of alpha and beta must be positive")

        self.spare_arrival_window = optional_nonnegative_float(
            "spare_arrival_window", config["spare_arrival_window"]
        )
        self.immediate_shutdown_on_stale_completion = bool(
            config["immediate_shutdown_on_stale_completion"]
        )

        min_spare_nodes = config["min_spare_nodes"]
        max_spare_nodes = config["max_spare_nodes"]
        total_nodes = len(self.state)
        requested_min_spare_nodes = int(min_spare_nodes)
        if requested_min_spare_nodes < 0:
            raise ValueError("min_spare_nodes must be non-negative")
        self.min_spare_nodes = min(requested_min_spare_nodes, total_nodes)
        if max_spare_nodes is None:
            self.max_spare_nodes = total_nodes
        else:
            requested_max_spare_nodes = int(max_spare_nodes)
            if requested_max_spare_nodes < 0:
                raise ValueError("max_spare_nodes must be non-negative")
            if requested_max_spare_nodes < requested_min_spare_nodes:
                raise ValueError(
                    "max_spare_nodes must be greater than or equal to "
                    "min_spare_nodes"
                )
            self.max_spare_nodes = min(
                requested_max_spare_nodes,
                total_nodes,
            )

        self._scheduled_wake_callbacks = set()
        self.call_me_laters_tl = []
        self.last_decision: Dict[str, object] = {}

        self.decision_log_enabled = bool(config["decision_log_enabled"])
        self.decision_log_path = self._resolve_decision_log_path(
            config["decision_log_path"],
            output_dir,
        )
        if (
            self.decision_log_enabled
            and self.decision_log_path is None
        ):
            raise ValueError(
                "decision_log_enabled=True, but decision_log_path "
                "was not configured"
            )

        self._decision_log_index = 0
        self._previous_logged_fallback: Optional[bool] = None
        self._previous_logged_gate: Optional[bool] = None
        if self.decision_log_enabled and self.decision_log_path is not None:
            self._initialise_decision_log()

        self._previous_node_jobs = self._node_job_snapshot()

    def prepare(self, current_time: float) -> None:
        self.current_time = float(current_time)
        # Timeout deadlines must follow the actual idle state,
        # even when the scheduler is not called on this update.
        self.refresh_timeout_list([
            node_id
            for node_id, node in self.state.items()
            if node.get("state") == "active" and node.get("job_id") is None
        ])

    def needs_replan(self, ctx: "PlanContext") -> bool:
        if self.timeout is None or not self.timeout_list:
            return False

        earliest = min(self.timeout_list.values())
        if ctx.now >= earliest:
            return True
        return self.next_timeout_at != earliest

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(scheduling_decision, SchedulingDecision):
            raise TypeError(
                "scheduling_decision must be a SchedulingDecision"
            )

        self.current_time = float(current_time)
        forecast = scheduling_decision.forecast
        if not isinstance(forecast, ICONForecast):
            raise RuntimeError(
                "ICONPowerManager requires the SNFICON scheduler"
            )

        self.nodes = forecast.nodes
        self.next_releases = forecast.next_releases
        self.model = forecast.model
        self.plan = forecast.plan
        self.compute_finish_times = scheduling_decision.compute_finish_times
        self.markov_fallback_enabled = forecast.markov_fallback_enabled
        released_by_completion = self._nodes_released_since_last_schedule(
            forecast.idle_before_commit
        )
        if forecast.mode == "snf_fallback":
            self._decide_snf_fallback(forecast, released_by_completion)
        else:
            self._decide_markov_power(forecast, released_by_completion)

        events = tuple(self.events)
        self.events = []
        return events

    def _decide_markov_power(self, forecast, released_by_completion):
        now = float(self.current_time)
        monitor_events = forecast.monitor_events
        markovianity = forecast.markovianity
        started_now = forecast.started_now
        allocated_now = forecast.allocated_now
        remaining_jobs = forecast.remaining_jobs
        decision_event = forecast.decision_event
        new_arrival_records = list(monitor_events["new_arrival_records"])

        mandatory_wake, mandatory_diag = self._ensure_waiting_capacity(
            remaining_jobs
        )

        spare_arrival_gate = self._spare_arrival_gate(now)
        spare_diag = None
        immediate_completion_shutdown = None

        if not remaining_jobs and decision_event:
            if spare_arrival_gate["allowed"]:
                spare_diag = self._optimise_and_apply_spare_target(
                    protected_nodes=set(mandatory_wake)
                )
            else:
                if (
                    self.immediate_shutdown_on_stale_completion
                    and bool(monitor_events["new_completion"])
                ):
                    immediate_completion_shutdown = (
                        self._switch_off_stale_completion_nodes(
                            released_nodes=released_by_completion,
                            allocated_nodes=allocated_now,
                            protected_nodes=set(mandatory_wake),
                        )
                    )

                immediate_nodes = (
                    immediate_completion_shutdown["switch_off"]
                    if immediate_completion_shutdown is not None
                    else []
                )
                spare_diag = {
                    "suppressed": True,
                    "reason": "no_recent_job_arrival",
                    "target": len(self.nodes.idle) + len(self.nodes.switching_on),
                    "wake": [],
                    "switch_off": list(immediate_nodes),
                    "arrival_age": spare_arrival_gate["arrival_age"],
                    "arrival_window": spare_arrival_gate["arrival_window"],
                    "time": now,
                }

        spare_timeout_diag = None
        if not remaining_jobs and self.timeout is not None:
            spare_timeout_diag = self._apply_spare_timeout_policy()

        self.last_decision = {
            "time": now,
            "new_arrival": bool(monitor_events["new_arrival"]),
            "new_completion": bool(monitor_events["new_completion"]),
            "new_arrival_count": monitor_events["new_arrival_count"],
            "new_completion_count": monitor_events["new_completion_count"],
            "new_arrivals": [
                job.get("job_id") for job in new_arrival_records
            ],
            "started_now": sorted(started_now),
            "remaining_queue": [
                job.get("job_id") for job in remaining_jobs
            ],
            "released_by_completion": list(released_by_completion),
            "mandatory": mandatory_diag,
            "spare": spare_diag,
            "spare_arrival_gate": spare_arrival_gate,
            "immediate_completion_shutdown": immediate_completion_shutdown,
            "spare_timeout": spare_timeout_diag,
            "arrival_rate": self.model.arrival_rate(),
            "service_rate": self.model.service_rate(),
            "runtime_ratio_median": self.model.runtime_ratio_median(),
            "runtime_log_ratio_mean": self.model.ema_log_runtime_ratio,
            "runtime_log_ratio_second_moment": (
                self.model.ema_log_runtime_ratio_sq
            ),
            "runtime_ratio_samples": self.model.runtime_ratio_samples,
            "active_job_predictions": self.model.active_job_predictions(),
            "resource_pmf": self.model.resource_pmf(),
            "policy_mode": "markov_power",
            "markovianity": markovianity,
        }
        self._append_decision_log(
            decision=self.last_decision,
            fallback=False,
            gate=spare_arrival_gate,
            decision_event=decision_event,
        )

        self._update_node_jobs(allocated_now)

    def _decide_snf_fallback(self, forecast, released_by_completion):
        """Only the wake-ups and timeouts plain SNFPSAS planning needs.

        The fallback intentionally skips all speculative Markov spare actions.
        """
        now = float(self.current_time)
        monitor_events = forecast.monitor_events
        markovianity = forecast.markovianity
        started_now = forecast.started_now
        remaining_jobs = forecast.remaining_jobs
        future_plan = self.plan

        event_counts = {
            float(bucket["timestamp"]): len(bucket.get("events", []))
            for bucket in self.events
        }
        self._emit_wake_triggers_from_plan(self.plan)

        if self.timeout is not None:
            self._timeout_policy()
        self._build_callbacks()

        immediate_wake = []
        future_wake_callbacks = []
        for bucket in self.events:
            timestamp = float(bucket["timestamp"])
            start = event_counts.get(timestamp, 0)
            for event in bucket.get("events", [])[start:]:
                if event.get("type") == "switch_on":
                    immediate_wake.extend(event.get("nodes", []))
                elif event.get("type") == "call_me_later_so":
                    future_wake_callbacks.append(timestamp)
                    self._scheduled_wake_callbacks.add(timestamp)

        spare_arrival_gate = self._spare_arrival_gate(now)
        decision_event = forecast.decision_event

        self.last_decision = {
            "time": now,
            "policy_mode": "snf_fallback",
            "markovianity": dict(markovianity),
            "spare_arrival_gate": spare_arrival_gate,
            "new_arrival": bool(monitor_events["new_arrival"]),
            "new_completion": bool(monitor_events["new_completion"]),
            "new_arrival_count": int(
                monitor_events["new_arrival_count"]
            ),
            "new_completion_count": int(
                monitor_events["new_completion_count"]
            ),
            "started_now": sorted(int(job_id) for job_id in started_now),
            "remaining_queue": [
                job.get("job_id") for job in remaining_jobs
            ],
            "released_by_completion": list(released_by_completion),
            "mandatory": {
                "planned_jobs": [
                    {
                        "job_id": job.get("job_id"),
                        "nodes": list(nodes),
                        "start_time": float(start_time),
                        "finish_time": float(finish_time),
                    }
                    for job, nodes, start_time, finish_time in future_plan
                ],
                "woken": sorted(set(immediate_wake)),
                "future_wake_callbacks": sorted(
                    set(future_wake_callbacks)
                ),
            },
            "spare": {
                "suppressed": True,
                "reason": "recent_workload_not_well_markovian",
            },
            "arrival_rate": self.model.arrival_rate(),
            "service_rate": self.model.service_rate(),
            "resource_pmf": self.model.resource_pmf(),
        }
        self._append_decision_log(
            decision=self.last_decision,
            fallback=True,
            gate=spare_arrival_gate,
            decision_event=decision_event,
        )

        self._update_node_jobs(forecast.allocated_now)

    def _ensure_waiting_capacity(self, jobs):
        # Emit only the wake actions the plan for queued jobs requires.
        now = float(self.current_time)

        # Preserve callback-set pruning and the exact empty diagnostics.
        if not jobs:
            immediate_wake, callback_times = self._emit_wake_triggers_from_plan(
                []
            )
            return immediate_wake, {
                "required_nodes": 0,
                "planned_jobs": [],
                "woken": list(immediate_wake),
                "future_wake_callbacks": list(callback_times),
                "estimated_wait": 0.0,
                "estimated_energy": 0.0,
            }

        future_plan = self.plan
        immediate_wake, callback_times = self._emit_wake_triggers_from_plan(
            future_plan
        )

        return immediate_wake, {
            "required_nodes": sum(int(job["res"]) for job in jobs),
            "planned_jobs": [
                {
                    "job_id": job.get("job_id"),
                    "nodes": list(nodes),
                    "start_time": float(start_time),
                    "finish_time": float(finish_time),
                }
                for job, nodes, start_time, finish_time in future_plan
            ],
            "woken": list(immediate_wake),
            "future_wake_callbacks": list(callback_times),
            "estimated_wait": sum(
                max(0.0, float(start_time) - now)
                for _, _, start_time, _ in future_plan
            ),
            "estimated_energy": sum(
                self._wake_transition_energy(node_id)
                for node_id in immediate_wake
            ),
        }

    def _update_node_jobs(self, allocated_now):
        # Remember node ownership, to find nodes released before the next plan.
        next_node_jobs = self._node_job_snapshot()
        for node_id, previous_job_id in self._previous_node_jobs.items():
            if (
                previous_job_id is not None
                and next_node_jobs.get(node_id) is None
                and node_id not in self.nodes.idle
                and node_id not in self.nodes.switching_off
                and node_id not in self.nodes.sleeping
            ):
                next_node_jobs[node_id] = previous_job_id
        for node_id, job_id in allocated_now.items():
            next_node_jobs[node_id] = job_id

        self._previous_node_jobs = next_node_jobs

    def refresh_timeout_list(self, idle_nodes: Iterable[int]) -> None:
        if self.timeout is None:
            self.timeout_list = {}
            self.next_timeout_at = None
            self.to_be_switched_off = []
            return

        now = float(self.current_time)
        expires_at = now + float(self.timeout)
        idle_list = list(dict.fromkeys(idle_nodes))
        idle_set = set(idle_list)
        self.timeout_list = {
            node_id: deadline
            for node_id, deadline in self.timeout_list.items()
            if node_id in idle_set
        }
        for node_id in idle_list:
            self.timeout_list.setdefault(node_id, expires_at)

    def can_switch_off(
        self,
        node_id: int,
        idle_nodes: Iterable[int],
    ) -> bool:
        if node_id not in idle_nodes:
            return False
        if not self.respect_timeout or self.timeout is None:
            return True
        expiry = self.timeout_list.get(node_id)
        return (
            expiry is not None
            and float(self.current_time) + EPS >= float(expiry)
        )

    def emit_switch_on(
        self,
        nodes: Iterable[int],
        sleeping_nodes: Iterable[int],
    ) -> list[int]:
        selected = list(dict.fromkeys(nodes))
        selected = [node_id for node_id in selected if node_id in sleeping_nodes]
        if selected:
            self.push_event(
                float(self.current_time),
                {"type": "switch_on", "nodes": selected},
            )
        return selected

    def emit_switch_off(
        self,
        nodes: Iterable[int],
        idle_nodes: Iterable[int],
        *,
        ignore_timeout: bool = False,
    ) -> list[int]:
        selected = list(dict.fromkeys(nodes))
        idle_set = set(idle_nodes)
        selected = [
            node_id
            for node_id in selected
            if (
                node_id in idle_set
                and (
                    ignore_timeout
                    or self.can_switch_off(node_id, idle_set)
                )
            )
        ]
        if not selected:
            return []

        self.push_event(
            float(self.current_time),
            {"type": "switch_off", "nodes": selected},
        )
        self.remove_from_timeout_list(selected)
        return selected

    def schedule_wake_callback(self, timestamp: float) -> None:
        self.push_event(
            float(timestamp),
            {"type": "call_me_later_so"},
        )

    def remove_from_timeout_list(self, node_ids: Iterable[int]) -> None:
        for node_id in node_ids:
            self.timeout_list.pop(node_id, None)

    def apply_timeout_policy(
        self,
        selected_list: Sequence[tuple[Any, Sequence[int], float, float]],
        idle_nodes: Iterable[int],
    ) -> None:
        if self.timeout is None:
            return

        idle_set = set(idle_nodes)
        self.refresh_timeout_list(idle_set)
        now = float(self.current_time)
        keep: dict[int, float] = {}
        switch_off: list[int] = []
        next_earliest: float | None = None

        for node_id, deadline in self.timeout_list.items():
            if node_id not in idle_set:
                continue

            if now + EPS >= float(deadline):
                selected_plan = next(
                    (
                        item
                        for item in selected_list
                        if node_id in item[1]
                    ),
                    None,
                )
                if selected_plan is None:
                    switch_off.append(node_id)
                    continue

                start_time = float(selected_plan[2])
                transition_time = (
                    self._switch_off_duration(node_id)
                    + self._wake_lead_time_safe(node_id)
                )
                if now + transition_time <= start_time + EPS:
                    switch_off.append(node_id)
                else:
                    keep[node_id] = float(deadline)
            else:
                keep[node_id] = float(deadline)
                next_earliest = (
                    float(deadline)
                    if next_earliest is None
                    else min(next_earliest, float(deadline))
                )

        self.timeout_list = keep
        if switch_off:
            self.push_event(
                now,
                {"type": "switch_off", "nodes": switch_off},
            )

        if (
            next_earliest is not None
            and self.next_timeout_at != next_earliest
        ):
            self.push_event(
                float(next_earliest),
                {"type": "call_me_later_to"},
            )
            self.next_timeout_at = float(next_earliest)

    def _transition_time_safe(
        self,
        node_id: int,
        from_state: str,
        to_state: str,
    ) -> float:
        try:
            return max(
                0.0,
                self._transition_time(node_id, from_state, to_state),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _wake_lead_time_safe(self, node_id: int) -> float:
        cached = self._wake_lead_time_cache.get(node_id)
        if cached is not None:
            return cached
        value = (
            self._transition_time_safe(
                node_id,
                "sleeping",
                "switching_on",
            )
            + self._transition_time_safe(
                node_id,
                "switching_on",
                "active",
            )
        )
        self._wake_lead_time_cache[node_id] = value
        return value

    def _switch_off_duration(self, node_id: int) -> float:
        return self._transition_time_safe(
            node_id,
            "switching_off",
            "sleeping",
        )

    # ------------------------------------------------------------------
    # Spare capacity, mandatory wake-ups, timeouts and the decision log
    # ------------------------------------------------------------------

    DECISION_LOG_FIELDS = [
        "decision_index",
        "current_time",
        "fallback",
        "fallback_changed",
        "fallback_enabled",
        "fallback_reason",
        "gate",
        "gate_changed",
        "gate_enabled",
        "gate_reason",
        "gate_relevant",
        "policy_mode",
        "trigger",
        "decision_event",
        "markov_well",
        "markov_reason",
        "markov_window_seconds",
        "arrival_markov_accepted",
        "service_markov_accepted",
        "arrival_sample_count",
        "service_sample_count",
        "arrival_coefficient_of_variation",
        "arrival_lag1_autocorrelation",
        "arrival_ks_distance",
        "arrival_failed_checks",
        "service_coefficient_of_variation",
        "service_lag1_autocorrelation",
        "service_ks_distance",
        "service_failed_checks",
        "new_arrival",
        "new_completion",
        "new_arrival_count",
        "new_completion_count",
        "waiting_jobs",
        "started_jobs",
        "idle_nodes",
        "sleeping_nodes",
        "switching_on_nodes",
        "switching_off_nodes",
        "computing_nodes",
        "arrival_age",
        "arrival_window",
        "arrival_rate",
        "service_rate",
        "spare_suppressed",
        "spare_reason",
        "spare_target",
        "wake_count",
        "switch_off_count",
    ]

    @staticmethod
    def _resolve_decision_log_path(
        configured_path: Optional[str],
        output_dir: Optional[str],
    ) -> Optional[Path]:
        # Without an explicit path the log goes into the run's output folder.
        if configured_path:
            return Path(configured_path).expanduser().resolve()
        if output_dir:
            return (
                Path(output_dir).expanduser().resolve()
                / "scheduler_decision_log.csv"
            )
        return None

    def _initialise_decision_log(self) -> None:
        path = self.decision_log_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=self.DECISION_LOG_FIELDS,
            ).writeheader()

    @staticmethod
    def _decision_trigger(
        decision: Mapping[str, object],
        *,
        first_schedule: bool,
    ) -> str:
        triggers = []
        if first_schedule:
            triggers.append("initial")
        if bool(decision.get("new_arrival")):
            triggers.append("arrival")
        if bool(decision.get("new_completion")):
            triggers.append("completion")
        return "+".join(triggers) if triggers else "callback"

    def _append_decision_log(
        self,
        *,
        decision: Mapping[str, object],
        fallback: bool,
        gate: Mapping[str, object],
        decision_event: bool,
    ) -> None:
        if not self.decision_log_enabled:
            return

        path = self.decision_log_path
        if path is None:
            return

        markov = decision.get("markovianity") or {}
        arrival_markov = markov.get("arrival") or {}
        service_markov = markov.get("service") or {}
        spare = decision.get("spare") or {}
        mandatory = decision.get("mandatory") or {}

        gate_allowed = bool(gate.get("allowed", True))
        fallback = bool(fallback)
        fallback_changed = (
            self._previous_logged_fallback is None
            or fallback != self._previous_logged_fallback
        )
        gate_changed = (
            self._previous_logged_gate is None
            or gate_allowed != self._previous_logged_gate
        )

        remaining_queue = decision.get("remaining_queue") or []
        started_now = decision.get("started_now") or []
        wake_nodes = []
        switch_off_nodes = []
        if isinstance(spare, Mapping):
            wake_nodes = spare.get("wake") or []
            switch_off_nodes = spare.get("switch_off") or []
        if not wake_nodes and isinstance(mandatory, Mapping):
            wake_nodes = mandatory.get("woken") or []

        first_schedule = self._decision_log_index == 0
        row = {
            "decision_index": self._decision_log_index,
            "current_time": float(decision.get("time", self.current_time)),
            "fallback": fallback,
            "fallback_changed": fallback_changed,
            "fallback_enabled": self.markov_fallback_enabled,
            "fallback_reason": (
                markov.get("reason", "") if fallback else ""
            ),
            "gate": gate_allowed,
            "gate_changed": gate_changed,
            "gate_enabled": bool(gate.get("enabled", False)),
            "gate_reason": gate.get("reason", ""),
            "gate_relevant": bool(
                not fallback
                and decision_event
                and len(remaining_queue) == 0
            ),
            "policy_mode": decision.get("policy_mode", ""),
            "trigger": self._decision_trigger(
                decision,
                first_schedule=first_schedule,
            ),
            "decision_event": bool(decision_event),
            "markov_well": bool(markov.get("well_markovian", False)),
            "markov_reason": markov.get("reason", ""),
            "markov_window_seconds": markov.get("window_seconds", ""),
            "arrival_markov_accepted": arrival_markov.get("accepted", ""),
            "service_markov_accepted": service_markov.get("accepted", ""),
            "arrival_sample_count": arrival_markov.get("sample_count", ""),
            "service_sample_count": service_markov.get("sample_count", ""),
            "arrival_coefficient_of_variation": arrival_markov.get("coefficient_of_variation", ""),
            "arrival_lag1_autocorrelation": arrival_markov.get("lag1_autocorrelation", ""),
            "arrival_ks_distance": arrival_markov.get("ks_distance", ""),
            "arrival_failed_checks": ",".join(arrival_markov.get("failed_checks", [])),
            "service_coefficient_of_variation": service_markov.get("coefficient_of_variation", ""),
            "service_lag1_autocorrelation": service_markov.get("lag1_autocorrelation", ""),
            "service_ks_distance": service_markov.get("ks_distance", ""),
            "service_failed_checks": ",".join(service_markov.get("failed_checks", [])),
            "new_arrival": bool(decision.get("new_arrival", False)),
            "new_completion": bool(decision.get("new_completion", False)),
            "new_arrival_count": int(decision.get("new_arrival_count", 0)),
            "new_completion_count": int(
                decision.get("new_completion_count", 0)
            ),
            "waiting_jobs": len(remaining_queue),
            "started_jobs": len(started_now),
            "idle_nodes": len(self.nodes.idle),
            "sleeping_nodes": len(self.nodes.sleeping),
            "switching_on_nodes": len(self.nodes.switching_on),
            "switching_off_nodes": len(self.nodes.switching_off),
            "computing_nodes": len(self.nodes.computing),
            "arrival_age": gate.get("arrival_age", ""),
            "arrival_window": gate.get("arrival_window", ""),
            "arrival_rate": decision.get("arrival_rate", ""),
            "service_rate": decision.get("service_rate", ""),
            "spare_suppressed": (
                bool(spare.get("suppressed", False))
                if isinstance(spare, Mapping)
                else False
            ),
            "spare_reason": (
                spare.get("reason", "")
                if isinstance(spare, Mapping)
                else ""
            ),
            "spare_target": (
                spare.get("target", "")
                if isinstance(spare, Mapping)
                else ""
            ),
            "wake_count": len(wake_nodes),
            "switch_off_count": len(switch_off_nodes),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.DECISION_LOG_FIELDS,
            )
            writer.writerow(row)

        self._previous_logged_fallback = fallback
        self._previous_logged_gate = gate_allowed
        self._decision_log_index += 1

    def _node_job_snapshot(self) -> Dict[int, object]:
        # Return scheduler-visible job ownership for every node
        return {
            node_id: (
                node.get("job_id")
                if node.get("state") == "active"
                else None
            )
            for node_id, node in self.state.items()
        }

    def _nodes_released_since_last_schedule(self, idle_nodes) -> List[int]:
        # Find nodes that changed from computing to active-and-idle
        released = []
        for node_id, previous_job_id in self._previous_node_jobs.items():
            if previous_job_id is None:
                continue
            node = self.state.get(node_id, {})
            if (
                node.get("state") == "active"
                and node.get("job_id") is None
                and node_id in idle_nodes
            ):
                released.append(node_id)
        return sorted(released)

    def _switch_off_stale_completion_nodes(
        self,
        *,
        released_nodes: Sequence[int],
        allocated_nodes: Iterable[int],
        protected_nodes: Iterable[int],
    ) -> Dict[str, object]:
        """Immediately switch off nodes released while the gate is closed.

        Nodes reused by an immediately-started job, selected for mandatory
        wake-up, or no longer idle are excluded.  The ordinary idle timeout is
        intentionally bypassed for the remaining nodes.
        """
        excluded = set(allocated_nodes) | set(protected_nodes)
        candidates = [
            node_id
            for node_id in released_nodes
            if node_id not in excluded and node_id in self.nodes.idle
        ]
        switched_off = self._emit_switch_off(
            candidates,
            ignore_timeout=True,
        )
        return {
            "enabled": True,
            "released_nodes": sorted(set(released_nodes)),
            "excluded_nodes": sorted(
                node_id for node_id in set(released_nodes) if node_id in excluded
            ),
            "switch_off": list(switched_off),
            "reason": "arrival_gate_closed_after_completion",
        }

    def _spare_arrival_gate(self, now: float) -> Dict[str, object]:
        """Report whether speculative spare preparation is currently allowed.

        ``spare_arrival_window=None`` preserves the original Markov policy.
        Otherwise, the gate opens only after a monitor-confirmed arrival and
        remains open for the configured number of simulation seconds.
        """
        window = self.spare_arrival_window
        last_arrival = self.model.last_arrival_event_time

        if window is None:
            return {
                "enabled": False,
                "allowed": True,
                "last_arrival_time": last_arrival,
                "arrival_age": None,
                "arrival_window": None,
                "reason": "disabled",
            }

        if last_arrival is None:
            return {
                "enabled": True,
                "allowed": False,
                "last_arrival_time": None,
                "arrival_age": None,
                "arrival_window": float(window),
                "reason": "no_arrival_observed",
            }

        age = max(0.0, float(now) - float(last_arrival))
        allowed = age <= float(window) + EPS
        return {
            "enabled": True,
            "allowed": bool(allowed),
            "last_arrival_time": float(last_arrival),
            "arrival_age": float(age),
            "arrival_window": float(window),
            "reason": "recent_arrival" if allowed else "arrival_window_expired",
        }

    def _apply_spare_timeout_policy(self) -> Dict[str, object]:
        previous_counts = {
            float(bucket["timestamp"]): len(bucket.get("events", []))
            for bucket in self.events
        }
        self._timeout_policy()

        switch_off = []
        callbacks = []
        for bucket in self.events:
            timestamp = float(bucket["timestamp"])
            start = previous_counts.get(timestamp, 0)
            for event in bucket.get("events", [])[start:]:
                event_type = event.get("type")
                if event_type == "switch_off":
                    switch_off.extend(event.get("nodes", []))
                elif event_type == "call_me_later_to":
                    callbacks.append(timestamp)

        return {
            "timeout": float(self.timeout),
            "switch_off": sorted(set(switch_off)),
            "next_callback": min(callbacks) if callbacks else None,
        }

    def _remove_from_timeout_list(self, node_ids: Iterable[int]) -> None:
        self.remove_from_timeout_list(node_ids)

    def _timeout_policy(self) -> None:
        self.apply_timeout_policy(
            self.plan,
            self.nodes.idle,
        )

    def _build_callbacks(self) -> None:
        finish_times = self.compute_finish_times
        for node_id in self.nodes.sleeping:
            lead_time = self._wake_lead_time_safe(node_id)
            for finish_time in finish_times:
                callback_time = float(finish_time) - lead_time
                if callback_time < float(self.current_time):
                    continue
                if callback_time in self.call_me_laters_tl:
                    continue
                self.push_event(
                    callback_time,
                    {"type": "CALL_ME_LATER"},
                )
                self.call_me_laters_tl.append(callback_time)

    def _emit_wake_triggers_from_plan(self, plan):
        now = float(self.current_time)
        self._scheduled_wake_callbacks = {
            timestamp
            for timestamp in self._scheduled_wake_callbacks
            if timestamp > now + EPS
        }

        earliest_wake = {}
        for _, nodes, start_time, _ in plan:
            for node_id in nodes:
                if node_id not in self.nodes.sleeping:
                    continue

                wake_time = (
                    float(start_time)
                    - self._wake_lead_time_safe(node_id)
                )
                previous = earliest_wake.get(node_id)
                if previous is None or wake_time < previous:
                    earliest_wake[node_id] = wake_time

        immediate = sorted(
            node_id
            for node_id, wake_time in earliest_wake.items()
            if wake_time <= now + EPS
        )
        self._emit_switch_on(immediate)

        future_times = sorted(
            {
                float(wake_time)
                for wake_time in earliest_wake.values()
                if wake_time > now + EPS
            }
        )

        emitted_callbacks = []
        for wake_time in future_times:
            if wake_time in self._scheduled_wake_callbacks:
                continue
            self.schedule_wake_callback(wake_time)
            self._scheduled_wake_callbacks.add(wake_time)
            emitted_callbacks.append(wake_time)

        return immediate, emitted_callbacks

    def _optimise_and_apply_spare_target(self, protected_nodes: set):
        """Enumerate feasible spare targets and apply the minimum-cost plan."""
        del protected_nodes

        now = float(self.current_time)
        arrival_rate = self.model.arrival_rate()
        service_rate = self.model.service_rate()

        active_job_ids = {
            node.get("job_id")
            for node in self.state.values()
            if node.get("state") == "active"
            and node.get("job_id") is not None
        }

        next_completion = self._next_completion_delay()
        if math.isfinite(next_completion) and next_completion > EPS:
            completion_hazard = 1.0 / next_completion
        else:
            completion_hazard = len(active_job_ids) * service_rate

        total_hazard = arrival_rate + completion_hazard
        horizon_cap = min(
            self.model.max_decision_horizon,
            self._next_timeout_delay(),
        )

        if total_hazard <= EPS:
            horizon = horizon_cap
            probability_next_event_is_arrival = 0.0
            gamma_for_residual = 1.0 / max(EPS, horizon_cap)
        else:
            horizon = (
                -math.expm1(-total_hazard * horizon_cap)
                / total_hazard
            )
            probability_next_event_is_arrival = (
                arrival_rate / total_hazard
            )
            gamma_for_residual = total_hazard

        expired_idle = []
        if self.respect_timeout and self.timeout is not None:
            expired_idle = [
                node_id
                for node_id in self.nodes.idle
                if (
                    self.timeout_list.get(node_id) is not None
                    and now + EPS
                    >= float(self.timeout_list[node_id])
                )
            ]
        expired_idle_set = set(expired_idle)

        forced_idle = [
            node_id
            for node_id in self.nodes.idle
            if (
                node_id not in expired_idle_set
                and not self._can_switch_off(node_id)
            )
        ]
        forced_idle_set = set(forced_idle)
        optional_idle = sorted(
            (
                node_id
                for node_id in self.nodes.idle
                if (
                    node_id not in forced_idle_set
                    and node_id not in expired_idle_set
                )
            ),
            key=lambda node_id: (
                self._idle_power(node_id),
                -self._wake_lead_time_safe(node_id),
                node_id,
            ),
        )
        ranked_sleeping = sorted(
            self.nodes.sleeping,
            key=lambda node_id: (
                self.alpha * self._wake_lead_time_safe(node_id)
                + self.beta * self._wake_transition_energy(node_id),
                self._wake_lead_time_safe(node_id),
                self._wake_transition_energy(node_id),
                node_id,
            ),
        )

        forced_warm = len(self.nodes.switching_on) + len(forced_idle)
        maximum_realisable = (
            len(self.nodes.switching_on)
            + len(forced_idle)
            + len(optional_idle)
            + len(self.nodes.sleeping)
        )
        minimum_target = max(
            forced_warm,
            min(self.min_spare_nodes, maximum_realisable),
        )
        maximum_target = max(
            minimum_target,
            min(self.max_spare_nodes, maximum_realisable),
        )

        pmf = self.model.resource_pmf()
        candidates = []

        for target in range(minimum_target, maximum_target + 1):
            plan = self._build_spare_plan(
                target=target,
                forced_idle=forced_idle,
                optional_idle=optional_idle,
                forced_switch_off=expired_idle,
                ranked_sleeping=ranked_sleeping,
            )

            expected_wait = self._expected_power_wait(
                plan=plan,
                resource_pmf=pmf,
                event_hazard=gamma_for_residual,
            )
            expected_wait *= probability_next_event_is_arrival

            expected_energy = self._expected_plan_energy(plan, horizon)
            objective = self.alpha * expected_wait + self.beta * expected_energy

            candidates.append(
                {
                    "target": target,
                    "plan": plan,
                    "expected_wait": float(expected_wait),
                    "expected_energy": float(expected_energy),
                    "objective": float(objective),
                }
            )

        best = min(
            candidates,
            key=lambda item: (
                item["objective"],
                item["expected_wait"],
                item["expected_energy"],
                item["target"],
            ),
        )
        plan = best["plan"]

        self._emit_switch_on(plan["wake"])
        self._emit_switch_off(plan["switch_off"])

        return {
            "target": best["target"],
            "wake": list(plan["wake"]),
            "switch_off": list(plan["switch_off"]),
            "keep_idle": list(plan["keep_idle"]),
            "expected_wait": best["expected_wait"],
            "expected_energy": best["expected_energy"],
            "objective": best["objective"],
            "decision_horizon": float(horizon),
            "horizon_cap": float(horizon_cap),
            "predicted_next_completion": (
                float(next_completion)
                if math.isfinite(next_completion)
                else None
            ),
            "probability_next_event_is_arrival": float(
                probability_next_event_is_arrival
            ),
            "candidate_costs": [
                {
                    "target": item["target"],
                    "expected_wait": item["expected_wait"],
                    "expected_energy": item["expected_energy"],
                    "objective": item["objective"],
                }
                for item in candidates
            ],
            "time": now,
        }

    def _next_completion_delay(self) -> float:
        remaining = [
            float(item["predicted_remaining"])
            for item in self.model.active_job_predictions()
            if math.isfinite(float(item["predicted_remaining"]))
        ]
        return min(remaining) if remaining else math.inf

    def _next_timeout_delay(self) -> float:
        if self.timeout is None:
            return self.model.max_decision_horizon

        now = float(self.current_time)
        delays = [
            float(expiry) - now
            for node_id, expiry in self.timeout_list.items()
            if (
                node_id in self.nodes.idle
                and float(expiry) > now + EPS
            )
        ]
        return (
            min(delays)
            if delays
            else self.model.max_decision_horizon
        )

    def _build_spare_plan(
        self,
        *,
        target: int,
        forced_idle: Sequence[int],
        optional_idle: Sequence[int],
        forced_switch_off: Sequence[int] = (),
        ranked_sleeping: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[int]]:
        switching_on = list(self.nodes.switching_on)
        forced_idle = list(forced_idle)
        optional_idle = list(optional_idle)
        forced_switch_off = list(forced_switch_off)

        base_warm = len(switching_on) + len(forced_idle)
        additional_needed = max(0, int(target) - base_warm)

        keep_optional_count = min(additional_needed, len(optional_idle))
        keep_optional = optional_idle[:keep_optional_count]
        switch_off = forced_switch_off + optional_idle[keep_optional_count:]

        still_needed = additional_needed - keep_optional_count
        if ranked_sleeping is None:
            wake = self._select_sleeping_to_wake(still_needed)
        else:
            wake = list(ranked_sleeping[:still_needed])

        wake_set = set(wake)
        return {
            "keep_idle": forced_idle + keep_optional,
            "switching_on": switching_on,
            "wake": wake,
            "switch_off": switch_off,
            "sleep": [
                node_id
                for node_id in self.nodes.sleeping
                if node_id not in wake_set
            ],
            "switching_off": list(self.nodes.switching_off),
        }

    def _expected_power_wait(
        self,
        *,
        plan: Mapping[str, Sequence[int]],
        resource_pmf: Mapping[int, float],
        event_hazard: float,
    ) -> float:
        """Expected power-induced delay for the next arriving job.

        For each requested node count r, work can start after the r-th
        fastest non-computing node is ready.  Nodes intentionally left cold are
        assumed to begin waking at the arrival instant.  Nodes being switched
        off must finish that transition before they can wake again.

        Queueing delay caused by currently computing nodes is intentionally not
        included because it is not controllable by the power action.  Requests
        larger than the current non-computing capacity are therefore truncated
        to that capacity; this adds the same omitted compute-delay constant to
        every candidate and does not change the selected target.
        """
        warm_delays = []
        warm_delays.extend(0.0 for _ in plan["keep_idle"])
        warm_delays.extend(
            self._expected_residual_delay(
                self._remaining_switch_on_time(nid), event_hazard
            )
            for nid in plan["switching_on"]
        )
        warm_delays.extend(
            self._expected_residual_delay(
                self._wake_lead_time_safe(nid), event_hazard
            )
            for nid in plan["wake"]
        )

        cold_delays = []
        cold_delays.extend(
            self._wake_lead_time_safe(nid) for nid in plan["sleep"]
        )
        cold_delays.extend(
            self._wake_lead_time_safe(nid)
            + self._expected_residual_delay(
                self._switch_off_duration(nid), event_hazard
            )
            for nid in plan["switch_off"]
        )
        cold_delays.extend(
            self._wake_lead_time_safe(nid)
            + self._expected_residual_delay(
                self._remaining_switch_off_time(nid), event_hazard
            )
            for nid in plan["switching_off"]
        )

        readiness = sorted(float(x) for x in warm_delays + cold_delays)
        if not readiness:
            return 0.0

        expected = 0.0
        for requested, probability in resource_pmf.items():
            r = max(1, min(int(requested), len(readiness)))
            expected += float(probability) * readiness[r - 1]

        return float(expected)

    def _expected_plan_energy(
        self,
        plan: Mapping[str, Sequence[int]],
        horizon: float,
    ) -> float:
        # Expected non-compute energy until the next decision event
        horizon = max(0.0, float(horizon))
        energy = 0.0

        for nid in plan["keep_idle"]:
            energy += self._idle_power(nid) * horizon

        for nid in plan["switching_on"]:
            rem = self._remaining_switch_on_time(nid)
            transition = min(rem, horizon)
            energy += self._state_power(nid, "switching_on") * transition
            energy += self._idle_power(nid) * max(0.0, horizon - transition)

        for nid in plan["wake"]:
            energy += self._wake_energy_over_horizon(nid, horizon)

        for nid in plan["sleep"]:
            energy += self._state_power(nid, "sleeping") * horizon

        for nid in plan["switch_off"]:
            duration = self._switch_off_duration(nid)
            transition = min(duration, horizon)
            energy += self._state_power(nid, "switching_off") * transition
            energy += self._state_power(nid, "sleeping") * max(
                0.0, horizon - transition
            )

        for nid in plan["switching_off"]:
            rem = self._remaining_switch_off_time(nid)
            transition = min(rem, horizon)
            energy += self._state_power(nid, "switching_off") * transition
            energy += self._state_power(nid, "sleeping") * max(
                0.0, horizon - transition
            )

        return float(energy)

    def _select_sleeping_to_wake(self, count: int) -> List[int]:
        count = max(0, int(count))
        if count == 0:
            return []

        ranked = sorted(
            self.nodes.sleeping,
            key=lambda nid: (
                self.alpha * self._wake_lead_time_safe(nid)
                + self.beta * self._wake_transition_energy(nid),
                self._wake_lead_time_safe(nid),
                self._wake_transition_energy(nid),
                nid,
            ),
        )
        return ranked[:count]

    def _emit_switch_on(self, nodes: Iterable[int]) -> None:
        nodes = self.emit_switch_on(
            nodes,
            self.nodes.sleeping,
        )
        if not nodes:
            return

        node_set = set(nodes)
        self.nodes.sleeping = [nid for nid in self.nodes.sleeping if nid not in node_set]
        for nid in nodes:
            if nid not in self.nodes.switching_on:
                self.nodes.switching_on.append(nid)

    def _emit_switch_off(
        self,
        nodes: Iterable[int],
        *,
        ignore_timeout: bool = False,
    ) -> List[int]:
        nodes = self.emit_switch_off(
            nodes,
            self.nodes.idle,
            ignore_timeout=ignore_timeout,
        )
        if not nodes:
            return []

        node_set = set(nodes)
        self.nodes.idle = [nid for nid in self.nodes.idle if nid not in node_set]
        for nid in nodes:
            if nid not in self.nodes.switching_off:
                self.nodes.switching_off.append(nid)
        self._remove_from_timeout_list(nodes)
        return list(nodes)

    def _can_switch_off(self, node_id: int) -> bool:
        return self.can_switch_off(
            node_id,
            self.nodes.idle,
        )

    def _remaining_switch_on_time(self, node_id: int) -> float:
        queued = self._remaining_phase_time(node_id, "switching_on")
        if queued is not None:
            return queued

        # Fallback for a simulator state that has not yet been mirrored into
        # next_releases.  state_start_time is optional in the attached base.
        node = self.state[node_id]
        total = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        started = float(node.get("state_start_time", self.current_time))
        elapsed = max(0.0, float(self.current_time) - started)
        return max(0.0, total - elapsed)

    def _remaining_switch_off_time(self, node_id: int) -> float:
        queued = self._remaining_phase_time(node_id, "switching_off")
        if queued is not None:
            return queued

        node = self.state[node_id]
        total = self._switch_off_duration(node_id)
        started = float(node.get("state_start_time", self.current_time))
        elapsed = max(0.0, float(self.current_time) - started)
        return max(0.0, total - elapsed)

    def _remaining_phase_time(
        self,
        node_id: int,
        phase_name: str,
    ) -> Optional[float]:
        entry = self.next_releases.get(node_id)
        if not entry:
            return None
        now = float(self.current_time)
        for segment in entry.get("queue", []):
            if str(segment.get("phase")) != phase_name:
                continue
            start = float(segment.get("start_time", now))
            finish = float(segment.get("finish_time", now))
            if start <= now + EPS and finish > now:
                return max(0.0, finish - now)
        return None

    def _wake_transition_energy(self, node_id: int) -> float:
        sleep_delay = self._transition_time_safe(
            node_id, "sleeping", "switching_on"
        )
        switch_delay = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        return (
            self._state_power(node_id, "sleeping") * sleep_delay
            + self._state_power(node_id, "switching_on") * switch_delay
        )

    def _wake_energy_over_horizon(self, node_id: int, horizon: float) -> float:
        remaining = max(0.0, float(horizon))
        energy = 0.0

        sleep_delay = self._transition_time_safe(
            node_id, "sleeping", "switching_on"
        )
        duration = min(sleep_delay, remaining)
        energy += self._state_power(node_id, "sleeping") * duration
        remaining -= duration

        switch_delay = self._transition_time_safe(
            node_id, "switching_on", "active"
        )
        duration = min(switch_delay, remaining)
        energy += self._state_power(node_id, "switching_on") * duration
        remaining -= duration

        energy += self._idle_power(node_id) * remaining
        return float(energy)

    @staticmethod
    def _expected_residual_delay(duration: float, hazard: float) -> float:
        """E[(duration - T)^+] for T ~ Exp(hazard)."""
        duration = max(0.0, float(duration))
        hazard = max(EPS, float(hazard))
        return max(
            0.0,
            duration - (1.0 - math.exp(-hazard * duration)) / hazard,
        )
