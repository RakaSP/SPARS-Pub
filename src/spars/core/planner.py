from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from spars.core.logging import log_trace
from spars.core.power_manager import PowerManager
from spars.core.scheduler import Scheduler
from spars.core.decision import SchedulingDecision
from spars.core.trigger import EPS, PlanContext, general_reasons

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager
    from spars.core.simulator.monitor import Monitor
    from spars.core.simulator.platform_control import PlatformControl
    import os


EventGroup = dict[str, Any]
PushEventBuckets = Callable[[list[EventGroup]], None]


class Planner:
    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        scheduler_name: str,
        scheduler_config: dict[str, Any],
        power_management_name: str,
        power_management_config: dict[str, Any],
        start_time: float,
        workload: "dict[str, Any]" = None,
        platform: "dict[int, dict[str, Any]]" = None,
        monitor: "Monitor" = None,
        platform_control: "PlatformControl" = None,
        output_dir: "str | os.PathLike[str] | None" = None,
    ) -> None:
        self.scheduler = Scheduler(
            machines=machines,
            jobs_manager=jobs_manager,
            name=scheduler_name,
            start_time=start_time,
            config=scheduler_config,
            workload=workload,
            platform=platform,
            monitor=monitor,
            platform_control=platform_control,
            output_dir=output_dir,
        )

        self.power_manager = PowerManager(
            machines=machines,
            jobs_manager=jobs_manager,
            name=power_management_name,
            start_time=start_time,
            config=power_management_config,
            platform_control=platform_control,
            output_dir=output_dir,
        )
        self.jobs_manager = jobs_manager
        self.last_scheduling_decision: SchedulingDecision | None = None
        self._last_plan_time: float | None = None
        self._invalidated: set[str] = set()
        # node -> state label expected from the events the last plan
        # emitted at the same timestamp.
        self._emitted_nodes: dict[Any, str] = {}

    def invalidate(self, reason: str = "external") -> None:
        """Force a replan on the next update."""
        self._invalidated.add(reason)

    def schedule(self, current_time: float) -> SchedulingDecision:
        decision = self.scheduler.schedule(
            current_time=current_time,
            power_state=self.power_manager,
        )
        self.last_scheduling_decision = decision
        return decision

    _EMITTED_LABEL = {
        "switch_on": "switching_on",
        "switch_off": "switching_off",
        "execution_start": "computing",
    }

    def _record_emitted(
        self,
        now: float,
        buckets: Any,
        emitted: dict[Any, str],
    ) -> None:
        for bucket in buckets:
            if float(bucket["timestamp"]) > now + EPS:
                continue
            for event in bucket["events"]:
                label = self._EMITTED_LABEL.get(event.get("type"))
                if label is None:
                    continue
                for node_id in event.get("nodes", ()):
                    emitted[node_id] = label

    def _plan_reasons(
        self,
        now: float,
        processed_events: list[dict[str, Any]] | None,
    ) -> list[str]:
        ctx = PlanContext(
            now=now,
            event_types=(
                None
                if processed_events is None
                else tuple(event["type"] for event in processed_events)
            ),
            last_plan_time=self._last_plan_time,
            last_decision=self.last_scheduling_decision,
            invalidated=frozenset(self._invalidated),
            waiting_job_ids=tuple(
                job["job_id"] for job in self.jobs_manager.waiting_queue
            ),
        )
        ignored = (
            self._emitted_nodes
            if (
                self._last_plan_time is not None
                and now <= self._last_plan_time + EPS
            )
            else None
        )

        reasons = general_reasons(ctx, self.scheduler, ignored)
        if self.scheduler.needs_replan(ctx):
            reasons.append("scheduler")
        if self.power_manager.needs_replan(ctx):
            reasons.append("power_manager")
        return reasons

    def update(
        self,
        current_time: float,
        push_event_buckets: PushEventBuckets,
        processed_events: list[dict[str, Any]] | None = None,
    ) -> SchedulingDecision | None:
        now = float(current_time)
        self.power_manager.prepare(now)

        reasons = self._plan_reasons(now, processed_events)
        if not reasons:
            # The scheduler's bookkeeping still follows the actual state.
            self.scheduler.observe(now, self.power_manager)
            log_trace(f"[{now}] plan skipped")
            return self.last_scheduling_decision

        log_trace(
            "---------- PLAN NEW ACTIONS: time=%s reasons=%s ----------",
            now,
            reasons,
        )
        self._invalidated.clear()
        self._last_plan_time = now
        emitted: dict[Any, str] = {}

        scheduling_decision = self.schedule(now)

        log_trace(f"[{now}] New scheduler decision: {scheduling_decision}")
        push_event_buckets(scheduling_decision.events)
        self._record_emitted(now, scheduling_decision.events, emitted)

        power_events = self.power_manager.decide(
            current_time=now,
            scheduling_decision=scheduling_decision,
        )

        log_trace(f"[{now}] New power-management events: {power_events}")
        push_event_buckets(power_events)
        self._record_emitted(now, power_events, emitted)
        self._emitted_nodes = emitted
        return scheduling_decision

    def apply_rl_action(
        self,
        action: Any,
        current_time: float,
    ) -> Any:
        if self.last_scheduling_decision is None:
            raise RuntimeError(
                "RL action cannot be applied before a scheduling decision"
            )
        self.invalidate("rl_action")
        return self.power_manager.apply_rl_action(
            action=action,
            current_time=current_time,
        )
