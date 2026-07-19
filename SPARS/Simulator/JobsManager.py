from SPARS.Logger import log_info, log_trace

QUEUE_KEYS = ("job_id", "res", "subtime", "reqtime", "runtime", "user_id")
ACTIVE_KEYS = ("job_id", "res", "subtime", "start_time", "reqtime", "runtime", "user_id", "nodes")

def _trim(job, keys):
    return {k: job[k] for k in keys}

class JobsManager:
    def __init__(self):
        self.waiting_queue = []
        self.active_jobs = []

    def add_to_waiting_queue(self, job):
        self.waiting_queue.append(_trim(job, QUEUE_KEYS))

    def remove_from_waiting_queue(self, job):
        self.waiting_queue.remove(_trim(job, QUEUE_KEYS))

    def add_to_active_jobs(self, job):
        self.active_jobs.append(_trim(job, ACTIVE_KEYS))

    def remove_from_active_jobs(self, job):
        self.active_jobs.remove(_trim(job, ACTIVE_KEYS))
