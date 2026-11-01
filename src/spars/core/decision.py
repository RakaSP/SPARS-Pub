from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlannedJob:
    job: dict[str, Any]
    nodes: tuple[int, ...]
    start_time: float
    finish_time: float


@dataclass(frozen=True)
class SchedulingDecision:
    events: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
    future_plan: tuple[PlannedJob, ...] = field(
        default_factory=tuple
    )
    compute_finish_times: tuple[float, ...] = field(
        default_factory=tuple
    )
    reserved_nodes: tuple[int, ...] = field(
        default_factory=tuple
    )
    # Sleeping nodes the power manager refused to wake, so the scheduler
    # planned without them.
    refused_wake: tuple[int, ...] = field(
        default_factory=tuple
    )
    # Anything else a scheduler tells the power manager it is paired with.
    forecast: Any = None
