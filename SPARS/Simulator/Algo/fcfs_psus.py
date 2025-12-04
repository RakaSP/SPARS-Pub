from SPARS.Simulator.Algo.BasePSUS import BasePSUS


class FCFSPSUS(BasePSUS):
    def schedule(self):

        super().prep_schedule()
        self.FCFSPSUS()
        super().events_builder()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSPSUS(self):
        for job in self.waiting_queue[:]:
            if job['res'] <= len(self.available):
                allocated_nodes = self.available[:job['res']]
                super().allocate(job, allocated_nodes)
            else:
                break
