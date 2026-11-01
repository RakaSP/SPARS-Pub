from ..base.psus import BasePSUS
from spars.core.decision import SchedulingDecision


class FCFSPSUS(BasePSUS):
    """First come, first served; wakes sleeping nodes when a job needs them (power-state unaware)."""
    DEFAULTS = {}  # no settings

    def schedule(self) -> SchedulingDecision:
        super().prep_schedule()
        self.FCFSPSUS()
        super().events_builder()
        return SchedulingDecision(
            events=tuple(self.events),
            compute_finish_times=self._compute_finish_times(),
            reserved_nodes=tuple(sorted(set(self.allocated))),
            refused_wake=tuple(sorted(self.refused_wake)),
        )

    def FCFSPSUS(self):
        for job in self.waiting_queue[:]:
            if job['res'] <= len(self.available):
                allocated_nodes = self.available[:job['res']]
                super().allocate(job, allocated_nodes)
            else:
                break
