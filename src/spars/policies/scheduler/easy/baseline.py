from ..fcfs.baseline import FCFSBaseline


class EASYBaseline(FCFSBaseline):
    """FCFS, plus backfilling behind the first job that does not fit.

    The first job that does not fit gets a reservation: the nodes that free up
    first, by the finish times the running jobs requested (``reqtime``). A
    later job may start now if it leaves the reserved nodes alone or is
    expected to finish before the reservation starts.
    """
    DEFAULTS = {}  # no settings

    def start_jobs(self, idle):
        starts = super().start_jobs(idle)
        started = {job["job_id"] for job, _ in starts}
        waiting = [
            job for job in self.jobs_manager.waiting_queue if job["job_id"] not in started
        ]
        if not waiting:
            return starts

        head, *behind = waiting
        free_at = self._free_at()
        reserved = sorted(free_at, key=lambda node_id: (free_at[node_id], node_id))[: head["res"]]
        if len(reserved) < head["res"] or any(free_at[n] == float("inf") for n in reserved):
            return starts  # no reservation can be made, so nothing may jump the head
        reservation_start = max(free_at[node_id] for node_id in reserved)

        for job in behind:
            unreserved = [node_id for node_id in idle if node_id not in reserved]
            if job["res"] <= len(unreserved):
                nodes = unreserved[: job["res"]]
            elif job["res"] <= len(idle) and self._finish_estimate(job, idle[: job["res"]]) <= reservation_start:
                nodes = idle[: job["res"]]
            else:
                continue
            starts.append((job, nodes))
            idle[:] = [node_id for node_id in idle if node_id not in nodes]
        return starts

    def _finish_estimate(self, job, nodes):
        speed = min(self.state[node_id]["compute_speed"] for node_id in nodes)
        return self.current_time + job["reqtime"] / speed

    def _free_at(self):
        """When each node is expected to be free; infinity for a node that is not on."""
        free_at = {
            node_id: self.current_time if node["state"] == "active" and node.get("job_id") is None else float("inf")
            for node_id, node in self.state.items()
        }
        for job in self.jobs_manager.active_jobs:
            speed = min(self.state[node_id]["compute_speed"] for node_id in job["nodes"])
            expected = max(self.current_time, job["start_time"] + job["reqtime"] / speed)
            for node_id in job["nodes"]:
                free_at[node_id] = expected
        return free_at
