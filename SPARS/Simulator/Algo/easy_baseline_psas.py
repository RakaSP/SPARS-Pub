# easy_baseline_psas.py
"""EASY backfilling baseline using the PSAS machine/energy model."""

from .fcfs_baseline_psas import FCFSBaselinePSAS


EPS = 1e-9


class EASYBaselinePSAS(FCFSBaselinePSAS):
    """Current-time EASY scheduler without PSAS power-state planning.

    EASY still needs one local prediction: the earliest reservation of the
    first blocked job.  That reservation is used only to decide whether a job
    may backfill now.  It is not stored in ``selected_list`` and produces no
    wake, timeout, or callback events.
    """

    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        # First preserve FCFS behavior for every job that can start now.
        started_now = self._current_fcfs_commit()

        remaining = [
            job
            for job in self.waiting_queue
            if job["job_id"] not in started_now
        ]

        if not remaining:
            return self.events

        head_job = remaining[0]
        reservation = self._find_head_reservation(
            head_job,
            barrier=now,
        )

        # Without a valid reservation, EASY cannot prove that a backfill job
        # will leave the head job undelayed.
        if reservation is None:
            return self.events

        head_nodes, head_start_time = reservation

        self._current_easy_commit(
            started_now=started_now,
            head_job_id=head_job["job_id"],
            head_start_time=head_start_time,
            head_reserved_ids=head_nodes,
        )

        return self.events

    def _find_head_reservation(self, head_job, barrier):
        """Return the head job's earliest no-wake reservation.

        This is deliberately not a general future schedule.  Sleeping and
        switching-off nodes are excluded because the baseline emits no
        switch-on events.  Nodes already computing or switching on may be
        reserved once their existing phase finishes.
        """
        required_nodes = int(head_job["res"])

        if required_nodes <= 0:
            return None

        candidates = (
            list(self.idle)
            + list(self.computing)
            + list(self.switching_on)
        )

        result = self._select_nodes_energy_aware(
            required_nodes=required_nodes,
            _candidates=candidates,
            min_start_time=float(barrier),
            release_map=self.next_releases,
        )

        if result is None:
            return None

        nodes, start_time = result
        return list(nodes), float(start_time)

    def _current_easy_commit(
        self,
        started_now,
        head_job_id,
        head_start_time,
        head_reserved_ids,
    ):
        """Start jobs now only when the head reservation remains intact."""
        now = float(self.current_time)
        seen_head = False
        reserved_nodes = set(head_reserved_ids)
        node_selection_static = self._build_node_selection_static_data(
            list(self.idle),
            self.next_releases,
        )

        for job in self.waiting_queue:
            job_id = job["job_id"]

            if job_id == head_job_id:
                seen_head = True
                continue

            if not seen_head or job_id in started_now:
                continue

            required_nodes = int(job["res"])

            if required_nodes <= 0:
                continue

            # First try nodes not reserved by the head job.  Such a job cannot
            # delay the reservation regardless of its duration.
            non_reserved_idle = [
                node_id
                for node_id in self.idle
                if node_id not in reserved_nodes
            ]

            if len(non_reserved_idle) >= required_nodes:
                result = self._select_nodes_energy_aware(
                    required_nodes=required_nodes,
                    _candidates=non_reserved_idle,
                    min_start_time=now,
                    node_static_data=node_selection_static,
                )

                if result is not None:
                    nodes, start_time = result

                    if float(start_time) <= now + EPS:
                        super().allocate(job, nodes)
                        started_now.add(job_id)
                        continue

            if len(self.idle) < required_nodes:
                continue

            result = self._select_nodes_energy_aware(
                required_nodes=required_nodes,
                _candidates=list(self.idle),
                min_start_time=now,
                node_static_data=node_selection_static,
            )

            if result is None:
                continue

            nodes, start_time = result

            if float(start_time) > now + EPS:
                continue

            # If the backfill uses a node reserved by the head job, it must
            # finish by the reservation time.
            if any(node_id in reserved_nodes for node_id in nodes):
                compute_speed = min(
                    float(self.state[node_id]["compute_speed"])
                    for node_id in nodes
                )

                if compute_speed <= 0.0:
                    continue

                finish_time = (
                    now
                    + float(job["reqtime"]) / compute_speed
                )

                if finish_time > float(head_start_time) + EPS:
                    continue

            super().allocate(job, nodes)
            started_now.add(job_id)
