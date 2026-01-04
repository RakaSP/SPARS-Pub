# easy_psas.py
import math
from .fcfs_psas import FCFSPSAS

EPS = 1e-9


class EASYPSAS(FCFSPSAS):
    """
    Your requested pipeline:

    current FCFS (commit now, idle-only, no waking)
    -> current EASY (commit now, backfill behind head, doesn't delay head)
    -> future FCFS (plan)
    -> future EASY (plan; allowed to override FCFS because we only emit callbacks)
    -> emit wake callbacks (switch_on now + call_me_later_so)
    """

    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        # 1) current FCFS
        started_now = self._current_fcfs_commit()

        # Remaining after current FCFS
        remaining0 = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        if not remaining0:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        # 2) define head by future FCFS plan (before current EASY)
        head_fcfs_plan = self._future_fcfs_plan(remaining0, barrier=now)
        if not head_fcfs_plan:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        head_job, head_nodes, head_start, head_finish = head_fcfs_plan[0]
        head_reserved_ids = {n["id"] for n in head_nodes}

        # 3) current EASY backfill commit-only (behind head)
        self._current_easy_commit(
            started_now=started_now,
            head_job_id=head_job["job_id"],
            head_start_time=float(head_start),
            head_reserved_ids=head_reserved_ids,
        )

        # Remaining after current EASY
        remaining = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        if not remaining:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        # 4) future FCFS seeded with fixed head (optional, useful for debugging)
        fcfs_seed_plan = self._future_fcfs_seed_head(
            head_job=head_job,
            head_nodes=head_nodes,
            head_start=float(head_start),
            jobs_after=[j for j in remaining if j["job_id"] != head_job["job_id"]],
        )

        # 5) future EASY plan-only (allowed to override)
        if self.current_time == 292950:
            print()
        easy_future_plan = self._future_easy_backfill_from_fcfs(fcfs_seed_plan)

        self.selected_list = list(easy_future_plan)

        # 6) wake callbacks
        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    # ---------------- current EASY (commit-only) ----------------
    def _current_easy_commit(self, started_now, head_job_id, head_start_time, head_reserved_ids):
        now = float(self.current_time)
        seen_head = False

        for job in self.waiting_queue:
            jid = job["job_id"]

            if jid == head_job_id:
                seen_head = True
                continue
            if not seen_head:
                continue
            if jid in started_now:
                continue

            req = int(job["res"])
            if req <= 0:
                continue

            # A) try idle non-reserved first
            idle_non_reserved = [n for n in self.idle if n["id"] not in head_reserved_ids]
            if len(idle_non_reserved) >= req:
                r = self._select_nodes_energy_aware(req, idle_non_reserved, min_start_time=now)
                if r is not None:
                    nodes, st = r
                    if float(st) <= now + EPS:
                        super().allocate(job, nodes)
                        started_now.add(jid)
                        continue

            # B) allow reserved idle nodes only if finish <= head_start_time
            idle_all = list(self.idle)
            if len(idle_all) < req:
                continue

            r = self._select_nodes_energy_aware(req, idle_all, min_start_time=now)
            if r is None:
                continue

            nodes, st = r
            if float(st) > now + EPS:
                continue

            uses_reserved = any(n["id"] in head_reserved_ids for n in nodes)
            if uses_reserved:
                sp = min(float(n["compute_speed"]) for n in nodes)
                ft = now + (float(job["reqtime"]) / sp)
                if ft > float(head_start_time) + EPS:
                    continue

            super().allocate(job, nodes)
            started_now.add(jid)

    # ---------------- future FCFS seeded with head ----------------
    def _future_fcfs_seed_head(self, head_job, head_nodes, head_start, jobs_after):
        now = float(self.current_time)

        base_by_id = super()._releases_by_id()
        scheduled_by_id = {
            nid: {
                "node_id": nid,
                "queue": [dict(seg) for seg in base_by_id[nid]["queue"]],
                "release_time": float(base_by_id[nid]["release_time"]),
            }
            for nid in base_by_id
        }

        def _append_planned_compute(job, selected_nodes, job_start_time):
            sp = min(float(n["compute_speed"]) for n in selected_nodes)
            wall = float(job["reqtime"]) / sp
            ft = float(job_start_time) + wall
            phase = f'compute(job={job["job_id"]})'
            for n in selected_nodes:
                e = scheduled_by_id[n["id"]]
                e["queue"].append({"phase": phase, "start_time": float(job_start_time), "finish_time": float(ft)})
                e["release_time"] = float(ft)
            return float(ft)

        head_finish = _append_planned_compute(head_job, head_nodes, float(head_start))
        plan = [(head_job, head_nodes, float(head_start), float(head_finish))]

        candidates = list(self.idle) + list(self.sleeping) + list(self.computing) + list(self.switching_on) + list(self.switching_off)

        barrier = float(head_start)
        for job in jobs_after:
            req = int(job["res"])
            if req <= 0:
                continue

            res = self._select_nodes_energy_aware(req, candidates, releases_by_id=scheduled_by_id, min_start_time=barrier)
            if res is None:
                break

            nodes, st = res
            ft = _append_planned_compute(job, nodes, float(st))
            plan.append((job, nodes, float(st), float(ft)))
            barrier = float(st)

        return plan

    # ---------------- future EASY (plan-only; ok to override FCFS) ----------------
    def _future_easy_backfill_from_fcfs(self, fcfs_plan):
        """
        Future EASY = FCFS baseline + backfill.

        - Baseline is the FCFS plan list: [(job, fcfs_nodes, fcfs_st, fcfs_ft), ...] in queue order.
        - Head reservation MUST be inserted first into the calendar.
        - For each later job:
            (a) try schedule earlier than FCFS start (backfill)
            (b) else try keep FCFS start+nodes
            (c) else slide to earliest >= FCFS start
        """
        now = float(self.current_time)

        if not fcfs_plan:
            return []

        base_by_id = super()._releases_by_id()
        scheduled_by_id = {
            nid: {
                "node_id": nid,
                "queue": [dict(seg) for seg in base_by_id[nid]["queue"]],
                "release_time": float(base_by_id[nid]["release_time"]),
            }
            for nid in base_by_id
        }

        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        def _append_compute_fixed(job, nodes, st):
            sp = min(float(n["compute_speed"]) for n in nodes)
            wall = float(job["reqtime"]) / sp
            ft = float(st) + wall
            phase = f'compute(job={job["job_id"]})'
            for n in nodes:
                e = scheduled_by_id[n["id"]]
                e["queue"].append({"phase": phase, "start_time": float(st), "finish_time": float(ft)})
                e["release_time"] = float(ft)
            return float(ft)

        def _fixed_feasible(nodes, st):
            st = float(st)
            for n in nodes:
                if float(scheduled_by_id[n["id"]]["release_time"]) > st + EPS:
                    return False
            return True

        # ---------------- seed HEAD first (critical fix) ----------------
        head_job, head_nodes, head_st, head_ft = fcfs_plan[0]
        head_st = float(head_st)

        # head must be feasible at its FCFS start
        if not _fixed_feasible(head_nodes, head_st):
            # If this ever happens, baseline is inconsistent with the current calendar.
            # Fall back: schedule head as early as possible from now (rare).
            res = self._select_nodes_energy_aware(
                required_nodes=int(head_job["res"]),
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=now,
            )
            if res is None:
                return []
            head_nodes, head_st = res
            head_ft = _append_compute_fixed(head_job, head_nodes, head_st)
        else:
            head_ft = _append_compute_fixed(head_job, head_nodes, head_st)

        out = [(head_job, head_nodes, float(head_st), float(head_ft))]

        # ---------------- backfill the rest on top of FCFS ----------------
        for job, fcfs_nodes, fcfs_st, fcfs_ft in fcfs_plan[1:]:
            req = int(job["res"])
            if req <= 0:
                continue

            fcfs_st = float(fcfs_st)

            # (a) try earlier-than-FCFS backfill
            res_early = self._select_nodes_energy_aware(
                required_nodes=req,
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=now,
            )
            if res_early is not None:
                nodes_early, st_early = res_early
                st_early = float(st_early)

                if st_early + EPS < fcfs_st:
                    ft_early = _append_compute_fixed(job, nodes_early, st_early)
                    out.append((job, nodes_early, float(st_early), float(ft_early)))
                    continue

            # (b) keep FCFS start+nodes if still feasible
            if _fixed_feasible(fcfs_nodes, fcfs_st):
                ft_keep = _append_compute_fixed(job, fcfs_nodes, fcfs_st)
                out.append((job, fcfs_nodes, float(fcfs_st), float(ft_keep)))
                continue

            # (c) otherwise slide it to earliest >= FCFS start
            res_slide = self._select_nodes_energy_aware(
                required_nodes=req,
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=fcfs_st,
            )
            if res_slide is None:
                break

            nodes_slide, st_slide = res_slide
            ft_slide = _append_compute_fixed(job, nodes_slide, float(st_slide))
            out.append((job, nodes_slide, float(st_slide), float(ft_slide)))

        # keep readable order
        out.sort(key=lambda x: float(x[2]))
        return out

