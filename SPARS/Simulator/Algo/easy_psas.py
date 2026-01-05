# easy_psas.py
import math
import re
from .fcfs_psas import FCFSPSAS

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


class EASYPSAS(FCFSPSAS):
    """
    Your requested pipeline:

    current FCFS (commit now, idle-only, no waking)
    -> current EASY (commit now, backfill behind head, doesn't delay head)
    -> future FCFS (plan)
    -> future EASY (plan-only backfill on top of FCFS baseline)
    -> emit wake callbacks (switch_on now + call_me_later_so)

    IMPORTANT: We only change FUTURE PLANNING behavior (gap-aware).
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

        # 4) future FCFS seeded with fixed head (baseline)
        fcfs_seed_plan = self._future_fcfs_seed_head(
            head_job=head_job,
            head_nodes=head_nodes,
            head_start=float(head_start),
            jobs_after=[j for j in remaining if j["job_id"] != head_job["job_id"]],
        )

        # 5) future EASY plan-only (gap-aware)
        easy_future_plan = self._future_easy_backfill_from_fcfs(fcfs_seed_plan)

        self.selected_list = list(easy_future_plan)

        # 6) wake callbacks (unchanged)
        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    # ---------------- current EASY (commit-only) ----------------
    # UNCHANGED
    def _current_easy_commit(self, started_now, head_job_id, head_start_time, head_reserved_ids):
        now = float(self.current_time)
        if now == 79236:
            print('x')
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
    # UNCHANGED
    def _future_fcfs_seed_head(self, head_job, head_nodes, head_start, jobs_after):
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

    # ---------------- future EASY (plan-only) ----------------
    def _future_easy_backfill_from_fcfs(self, fcfs_plan):
        """
        Future EASY = FCFS baseline + backfill.

        ONLY FIX (planning-only):
        - res_early/res_slide must respect GAPS in scheduled_by_id queues.
        - planned segments must be inserted into queue in time order and NOT destroy later reservations.
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

        # ---------------- CHANGED (planning-only): insert segment ordered, keep release_time=max ----------------
        def _insert_segment(q, seg):
            st = float(seg["start_time"])
            i = 0
            while i < len(q) and float(q[i]["start_time"]) <= st:
                i += 1
            q.insert(i, seg)

        def _append_compute_fixed(job, nodes, st):
            sp = min(float(n["compute_speed"]) for n in nodes)
            wall = float(job["reqtime"]) / sp
            ft = float(st) + wall
            phase = f'compute(job={job["job_id"]})'
            seg = {"phase": phase, "start_time": float(st), "finish_time": float(ft)}
            for n in nodes:
                e = scheduled_by_id[n["id"]]
                _insert_segment(e["queue"], dict(seg))
                # IMPORTANT: do not overwrite later reservations
                e["release_time"] = max(float(e["release_time"]), float(ft))
            return float(ft)

        # ---------------- NEW (planning-only): true feasibility inside gaps ----------------
        def _window_feasible(job, nodes, st):
            st = float(st)
            sp = min(float(n["compute_speed"]) for n in nodes)
            if sp <= 0:
                return False
            wall = float(job["reqtime"]) / sp
            end = st + wall

            for n in nodes:
                q = scheduled_by_id[n["id"]]["queue"]
                for seg in q:
                    s = float(seg["start_time"])
                    f = float(seg["finish_time"])
                    # overlap?
                    if not (end <= s + EPS or st >= f - EPS):
                        return False
            return True

        # ---------------- seed HEAD first ----------------
        head_job, head_nodes, head_st, head_ft = fcfs_plan[0]
        head_st = float(head_st)

        # keep your old behavior: head is FCFS baseline
        if not _window_feasible(head_job, head_nodes, head_st):
            # If baseline is inconsistent (rare), just keep it as-is (don’t “fix” anything else).
            # (You can debug it separately if needed.)
            return []

        head_ft = _append_compute_fixed(head_job, head_nodes, head_st)
        out = [(head_job, head_nodes, float(head_st), float(head_ft))]

        # ---------------- backfill the rest on top of FCFS baseline ----------------
        for job, fcfs_nodes, fcfs_st, fcfs_ft in fcfs_plan[1:]:
            req = int(job["res"])
            if req <= 0:
                continue

            fcfs_st = float(fcfs_st)

            # (a) CHANGED (planning-only): try earlier-than-FCFS backfill using GAP-AWARE selector
            res_early = self._future_select_nodes_energy_aware_gap(
                required_nodes=req,
                _candidates=candidates,
                scheduled_by_id=scheduled_by_id,
                min_start_time=now,
                job=job,
            )
            if res_early is not None:
                nodes_early, st_early = res_early
                st_early = float(st_early)
                if st_early + EPS < fcfs_st:
                    ft_early = _append_compute_fixed(job, nodes_early, st_early)
                    out.append((job, nodes_early, float(st_early), float(ft_early)))
                    continue

            # (b) CHANGED (planning-only): keep FCFS start+nodes if it actually fits in the gap
            if _window_feasible(job, fcfs_nodes, fcfs_st):
                ft_keep = _append_compute_fixed(job, fcfs_nodes, fcfs_st)
                out.append((job, fcfs_nodes, float(fcfs_st), float(ft_keep)))
                continue

            # (c) CHANGED (planning-only): slide to earliest >= FCFS start using GAP-AWARE selector
            res_slide = self._future_select_nodes_energy_aware_gap(
                required_nodes=req,
                _candidates=candidates,
                scheduled_by_id=scheduled_by_id,
                min_start_time=fcfs_st,
                job=job,
            )
            if res_slide is None:
                break

            nodes_slide, st_slide = res_slide
            ft_slide = _append_compute_fixed(job, nodes_slide, float(st_slide))
            out.append((job, nodes_slide, float(st_slide), float(ft_slide)))

        out.sort(key=lambda x: float(x[2]))
        return out

    # ---------------- NEW: gap-aware future selector (planning-only) ----------------
    def _future_select_nodes_energy_aware_gap(self, required_nodes, _candidates, scheduled_by_id, min_start_time, job):
        """
        Planner-only selection that respects gaps inside scheduled_by_id[nid]["queue"].

        - Returns (nodes, start_time) where start_time is earliest >= min_start_time
          such that ALL selected nodes are free for the whole job duration.
        - Ranking is kept similar to your _select_nodes_energy_aware:
            (cost, state_priority, timeout_priority, nid)
        """
        now = float(self.current_time)
        min_start_time = float(min_start_time)
        reqtime = float(job["reqtime"])

        # filter candidates present in calendar
        cand = [n for n in _candidates if n["id"] in scheduled_by_id]
        if len(cand) < required_nodes:
            return None

        machine_by_id = {m["id"]: m for m in self.machines.machines}

        # Precompute per-node (segments sorted) and power/cost parts
        per = {}
        time_points = {min_start_time}

        for n in cand:
            nid = n["id"]
            machine = machine_by_id[nid]
            q = list(scheduled_by_id[nid]["queue"])
            q.sort(key=lambda seg: float(seg["start_time"]))

            # gather finish times for candidate start points
            for seg in q:
                f = float(seg["finish_time"])
                if not math.isinf(f):
                    time_points.add(f)

            # state label same as FCFS selector
            if n["state"] == "active" and n.get("job_id") is None:
                state_label = "idle"
            elif n["state"] == "active" and n.get("job_id") is not None:
                state_label = "computing"
            else:
                state_label = n["state"]

            if state_label == "idle":
                state_priority = 0
            elif state_label == "computing":
                state_priority = 1
            elif state_label == "switching_on":
                state_priority = 2
            else:
                state_priority = 3

            # idle power
            idle_power = machine["states"]["active"]["power"]
            if idle_power == "from_dvfs":
                dvfs_profiles = machine["dvfs_profiles"]
                dvfs_mode = n["dvfs_mode"]
                idle_power = dvfs_profiles[dvfs_mode]["power"]

            # base energy waste (non-compute phases)
            base_energy_waste = 0.0
            for seg in q:
                if _COMPUTE_RE.fullmatch(str(seg["phase"])):
                    continue
                s = float(seg["start_time"])
                f = float(seg["finish_time"])
                if math.isinf(f):
                    continue
                if s < now:
                    dur = f - now
                else:
                    dur = f - s

                e_rate = machine["states"][seg["phase"]]["power"]
                if e_rate == "from_dvfs":
                    dvfs_profiles = machine["dvfs_profiles"]
                    dvfs_mode = n["dvfs_mode"]
                    e_rate = dvfs_profiles[dvfs_mode]["power"]

                base_energy_waste += float(e_rate) * float(dur)

            per[nid] = {
                "node": n,
                "q": q,
                "speed": float(n.get("compute_speed", 1.0)),
                "idle_power": float(idle_power),
                "base": float(base_energy_waste),
                "state_label": state_label,
                "state_priority": int(state_priority),
            }

        def _gap_bounds(q, t):
            """
            If t is inside a busy segment => None.
            Else returns (idle_since, free_until) where free_until is next segment start or inf.
            """
            idle_since = -math.inf
            for seg in q:
                s = float(seg["start_time"])
                f = float(seg["finish_time"])
                if f <= t:
                    idle_since = f
                    continue
                if s <= t < f:
                    return None
                if s > t:
                    return (idle_since, s)
            return (idle_since, math.inf)

        # Try times from earliest to latest
        for t in sorted(tp for tp in time_points if tp >= min_start_time - EPS):
            t = float(t)
            if t + EPS < min_start_time:
                continue

            pool = []
            for nid, dat in per.items():
                bounds = _gap_bounds(dat["q"], t)
                if bounds is None:
                    continue
                idle_since, free_until = bounds
                free_len = float(free_until) - t
                if free_len <= EPS:
                    continue

                # cost similar to your selector, but uses idle_since (gap-aware)
                if dat["state_label"] in ("switching_off", "sleeping"):
                    cost = dat["base"]
                else:
                    idle_ref = max(float(idle_since), now)
                    cost = dat["base"] + dat["idle_power"] * max(0.0, t - idle_ref)

                if dat["state_priority"] == 0:
                    timeout_priority = -self._remaining_idle_timeout(nid)
                else:
                    timeout_priority = 0.0

                pool.append((float(cost), int(dat["state_priority"]), float(timeout_priority), int(nid), float(dat["speed"]), float(free_len)))

            if len(pool) < required_nodes:
                continue

            pool.sort(key=lambda x: (x[0], x[1], x[2], x[3]))  # same ordering style

            # iterative filter to handle walltime depending on min speed
            cur_pool = pool
            chosen = None
            for _ in range(8):
                top = cur_pool[:required_nodes]
                min_speed = min(e[4] for e in top)
                if min_speed <= 0:
                    chosen = None
                    break
                wall = reqtime / min_speed

                filtered = [e for e in cur_pool if e[5] + EPS >= wall]
                if len(filtered) < required_nodes:
                    chosen = None
                    break

                filtered.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
                top2 = filtered[:required_nodes]

                if top2 == top:
                    chosen = top2
                    break

                cur_pool = filtered

            if chosen is None:
                continue

            nids = [e[3] for e in chosen]
            nodes = [per[nid]["node"] for nid in nids]
            return (nodes, float(t))

        return None
