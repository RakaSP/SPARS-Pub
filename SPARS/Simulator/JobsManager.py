from SPARS.Logger import log_info, log_trace

class JobsManager:
    def __init__(self):
        self.waiting_queue = []
        self.terminated_jobs = []
        self.finished_jobs = []
        self.active_jobs_id = []
        self.num_terminated_jobs = 0
        self.num_finished_jobs = 0

    def on_finish(self):
        for job in self.waiting_queue:
            self.remove_job_from_waiting_queue(job['job_id'], 'terminated')

        log_info(f'Job terminated count: {self.num_terminated_jobs}')
        
    def add_job_to_waiting_queue(self, job):
        self.waiting_queue.append(job)

    def remove_job_from_waiting_queue(self, job_id, type):
        for i, job in enumerate(self.waiting_queue):
            if job['job_id'] == job_id:
                if type == 'terminated':
                    self.terminated_jobs.append(job)
                    self.num_terminated_jobs += 1
                    log_trace(f'Job {job_id} is terminated')
                elif type == 'execution_start':
                    self.active_jobs_id.append(job_id)
                else:
                    raise ValueError(
                        f"Invalid removal type: '{type}' (expected 'terminated' or 'execution_start')")
                del self.waiting_queue[i]
                break
