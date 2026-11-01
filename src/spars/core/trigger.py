from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spars.core.scheduler import Scheduler

EPS = 1e-9

# Events whose effect the state/queue/plan-due rules already capture.
# Other types (wake callbacks, CALL_RL, change_dvfs_mode, oracle, new types)
# always trigger a replan.
STATE_EVENT_TYPES = frozenset({
    "arrival",
    "execution_start",
    "execution_finished",
    "switch_on",
    "switch_off",
    "turn_on",
    "turn_off",
})


@dataclass(frozen=True)
class PlanContext:
    now: float
    event_types: tuple[str, ...] | None
    last_plan_time: float | None
    last_decision: Any
    invalidated: frozenset[str]
    waiting_job_ids: tuple[Any, ...]


def general_reasons(
    ctx: PlanContext,
    scheduler: "Scheduler",
    ignored_nodes: "Mapping[int, str] | None",
) -> list[str]:
    """General plan-trigger rules. An empty list means the plan may be skipped."""
    if ctx.last_plan_time is None or ctx.last_decision is None:
        return ["first"]

    reasons = [f"invalidated:{name}" for name in sorted(ctx.invalidated)]

    if ctx.event_types is None:
        reasons.append("unspecified")
        return reasons

    for event_type in dict.fromkeys(ctx.event_types):
        if event_type not in STATE_EVENT_TYPES:
            reasons.append(f"event:{event_type}")

    plan = ctx.last_decision.future_plan
    planned_ids = {planned.job["job_id"] for planned in plan}
    if any(job_id not in planned_ids for job_id in ctx.waiting_job_ids):
        reasons.append("unplanned_jobs")

    if any(float(planned.start_time) <= ctx.now + EPS for planned in plan):
        reasons.append("plan_due")

    # Most expensive, so checked last and only when no reason was found yet.
    if not reasons and scheduler.plan_deviation(ctx.now, ignored_nodes):
        reasons.append("plan_deviation")

    return reasons
