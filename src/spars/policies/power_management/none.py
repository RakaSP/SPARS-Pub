from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from spars.core.decision import SchedulingDecision

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.trigger import PlanContext


class NoPowerManager:
    """Never switches a node: nodes stay on unless the scheduler wakes them."""
    DEFAULTS = {}  # no settings

    def __init__(
        self,
        machines: "Machine",
        start_time: float,
        config: dict[str, Any],
    ) -> None:
        if config:
            raise ValueError(
                "the none power manager does not accept config values"
            )

        self.machines = machines
        self.current_time = float(start_time)

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
        return ()
