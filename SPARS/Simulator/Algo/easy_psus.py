from .fcfs_psus import FCFSPSUS


class EASYPSUS(FCFSPSUS):
    def schedule(self):
        super().prep_schedule()
        super().FCFSPSUS()

        self.backfill()
        super().events_builder()
        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    def backfill(self):
        if len(self.waiting_queue) > 2:
            p_job = self.waiting_queue[0]
            backfill_queue = self.waiting_queue[1:]
            next_releases = sorted(
                self.compute_agenda, key=lambda a: a['release_time'])
            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']
            candidates = [
                nr['node_id'] for nr in next_releases if nr['release_time'] <= p_start_t]
            reservation = candidates[-p_job['res']:]

            for job in backfill_queue:
                not_reserved = [
                    node for node in self.available if node['id'] not in reservation]
                if job['res'] <= len(not_reserved):
                    allocated_nodes = not_reserved[:job['res']]
                    super().allocate(job, allocated_nodes)
                elif job['reqtime'] and job['reqtime'] <= p_start_t and job['res'] <= len(self.available):
                    allocated_nodes = self.available[:job['res']]
                    super().allocate(job, allocated_nodes)
