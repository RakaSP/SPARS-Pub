from collections.abc import Mapping, Sequence
from typing import Any


Job = Mapping[str, Any]
JobRecord = dict[str, Any]

QUEUE_KEYS: tuple[str, ...] = (
    "job_id",
    "res",
    "subtime",
    "reqtime",
    "runtime",
    "user_id",
)
ACTIVE_KEYS: tuple[str, ...] = (
    "job_id",
    "res",
    "subtime",
    "start_time",
    "reqtime",
    "runtime",
    "user_id",
    "nodes",
)


def _project_job(job: Job, keys: Sequence[str]) -> JobRecord:
    """Copy only the fields owned by a JobsManager collection."""
    return {key: job[key] for key in keys}


class JobsManager:
    """Own waiting and active job records used by the simulator."""

    def __init__(self) -> None:
        self.waiting_queue: list[JobRecord] = []
        self.active_jobs: list[JobRecord] = []

    def add_to_waiting_queue(self, job: Job) -> None:
        self.waiting_queue.append(
            _project_job(job, QUEUE_KEYS)
        )

    def remove_from_waiting_queue(self, job: Job) -> None:
        self.waiting_queue.remove(
            _project_job(job, QUEUE_KEYS)
        )

    def add_to_active_jobs(self, job: Job) -> None:
        self.active_jobs.append(
            _project_job(job, ACTIVE_KEYS)
        )

    def remove_from_active_jobs(self, job: Job) -> None:
        self.active_jobs.remove(
            _project_job(job, ACTIVE_KEYS)
        )
