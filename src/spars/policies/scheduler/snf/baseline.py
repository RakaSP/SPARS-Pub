from ..base.baseline import BaselineScheduler


class SNFBaseline(BaselineScheduler):
    """Smallest node request first, then the shortest requested time."""
    DEFAULTS = {}  # no settings

    def order(self, jobs):
        return sorted(jobs, key=lambda job: (job["res"], job["reqtime"]))
