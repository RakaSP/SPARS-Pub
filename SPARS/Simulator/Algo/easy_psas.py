import math
from .fcfs_psas import FCFSPSAS

import math
import re

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


class EASYPSAS(FCFSPSAS):
    """
    Option B (fixed):
      - FCFS is used for planning only (no switch_on emitted by FCFS).
      - EASY backfill only ALLOCATES jobs that start NOW (commit-only).
      - Head job is RESERVED (FCFS-chosen nodes), never delayed by backfill.
      - Then we build a FINAL future FCFS plan seeded with the head reservation.
      - From that FINAL plan we emit wake triggers:
          * switch_on now if needed
          * call_me_later_so at wake times for future starts
    """

    # ---------- public ----------
    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        # 1) FCFS plan-only from current state (no events)
        super().FCFSPSAS(plan_only=True)
        plan0 = list(self.selected_list)  # [(job, nodes, st, ft), ...]

        # 2) Allocate the FCFS "start-now prefix" (must match FCFS behavior)
        started_now = set()
        for job, nodes, st, ft in plan0:
            if float(st) <= now + 1e-9:
                super().allocate(job, nodes)
                started_now.add(job["job_id"])
            else:
                break

        # 3) Head job = first waiting job not started-now
        head_job = next((j for j in self.waiting_queue if j["job_id"] not in started_now), None)
        if head_job is None:
            # no future job to reserve
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        # Find head tuple from FCFS plan0
        head_nodes, head_start, head_finish = None, None, None
        for j, nodes, st, ft in plan0:
            if j["job_id"] == head_job["job_id"]:
                head_nodes = nodes
                head_start = float(st)
                head_finish = float(ft)
                break

        # Fallback if FCFS couldn't plan it (rare)
        if head_nodes is None:
            candidates = list(self.idle) + list(self.sleeping) + list(self.computing) + list(self.switching_on)
            res = self._select_nodes_energy_aware(int(head_job["res"]), candidates, min_start_time=now)
            if res is None:
                self.selected_list = []
                if self.timeout is not None:
                    super().timeout_policy()
                super().build_callbacks()
                return self.events
            head_nodes, head_start = res
            sp = min(float(n["compute_speed"]) for n in head_nodes)
            head_finish = float(head_start) + (float(head_job["reqtime"]) / sp)

        head_reserved_ids = {n["id"] for n in head_nodes}

        # 4) EASY backfill (commit-only): start jobs NOW if they fit and don't delay head
        self._easy_backfill_now(
            started_now=started_now,
            head_job_id=head_job["job_id"],
            head_start_time=float(head_start),
            head_reserved_ids=head_reserved_ids,
        )

        # 5) Build FINAL future plan (FCFS-style) seeded with the head reservation,
        #    so wake planning matches the actual decisions after backfill.
        remaining_jobs = [
            j for j in self.waiting_queue
            if (j["job_id"] not in started_now) and (j["job_id"] != head_job["job_id"])
        ]

        rest_plan = self._fcfs_plan_with_head_seed(
            jobs=remaining_jobs,
            head_job=head_job,
            head_nodes=head_nodes,
            head_start=float(head_start),
        )

        # Expose final reservation plan to timeout_policy (like FCFS), but consistent.
        self.selected_list = [(head_job, head_nodes, float(head_start), float(head_finish))] + rest_plan

        # 6) Wake planning from FINAL plan:
        #    - switch_on now only if we're already at/after the wake time
        #    - otherwise schedule call_me_later_so at wake times
        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    # ---------- EASY backfill (commit-only) ----------
    def _easy_backfill_now(self, started_now, head_job_id, head_start_time, head_reserved_ids):
        now = float(self.current_time)
        head_start_time = float(head_start_time)

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

            required = int(job["res"])
            if required <= 0:
                continue

            # Step 1: idle nodes NOT reserved for head
            idle_non_reserved = [n for n in self.idle if n["id"] not in head_reserved_ids]
            if len(idle_non_reserved) >= required:
                r = self._select_nodes_energy_aware(required, idle_non_reserved, min_start_time=now)
                if r is not None:
                    selected, st = r
                    if float(st) <= now + 1e-9:
                        super().allocate(job, selected)
                        started_now.add(jid)
                        continue

            # Step 2: allow using reserved idle nodes ONLY if finish <= head_start_time
            idle_all = list(self.idle)
            if len(idle_all) >= required:
                r = self._select_nodes_energy_aware(required, idle_all, min_start_time=now)
                if r is None:
                    continue
                selected, st = r
                if float(st) > now + 1e-9:
                    continue

                uses_reserved = any(n["id"] in head_reserved_ids for n in selected)
                if uses_reserved:
                    sp = min(float(n["compute_speed"]) for n in selected)
                    finish = now + (float(job["reqtime"]) / sp)
                    if finish > head_start_time + 1e-9:
                        continue

                super().allocate(job, selected)
                started_now.add(jid)

    # ---------- FCFS planning with head seed ----------
    def _fcfs_plan_with_head_seed(self, jobs, head_job, head_nodes, head_start):
        """
        Build a future FCFS plan on top of the CURRENT state (after backfill allocations),
        but treat the head reservation as already occupying its nodes from head_start..head_finish.
        Returns: [(job, nodes, st, ft), ...] for jobs AFTER head.
        """
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

        # Seed head reservation into the plan copy (so later jobs don't steal those nodes)
        head_finish = _append_planned_compute(head_job, head_nodes, float(head_start))

        # FCFS barrier: after head, next jobs can't start before head_start (order constraint)
        barrier = float(head_start)

        plan = []
        for job in jobs:
            required = int(job["res"])
            min_start_time = float(barrier)

            # Candidates: all partitions (same as FCFS)
            candidates = list(self.idle) + list(self.sleeping) + list(self.computing) + list(self.switching_on)

            res = self._select_nodes_energy_aware(
                required_nodes=required,
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=min_start_time,
            )
            if res is None:
                break

            selected, st = res
            ft = _append_planned_compute(job, selected, float(st))

            plan.append((job, selected, float(st), float(ft)))
            barrier = float(st)

        return plan

    # ---------- Wake triggers from final plan ----------
    def _emit_wake_triggers_from_plan(self, plan):
        """
        For sleeping nodes used by any FUTURE job in the plan:
          - compute earliest wake_time = start_time - t_on_active
          - if wake_time <= now: emit switch_on now for those nodes
          - else: emit call_me_later_so at wake_time (unique times)
        """
        now = float(self.current_time)
        sleeping_ids = {n["id"] for n in self.sleeping}

        earliest_wake_by_node = {}
        for job, nodes, st, ft in plan:
            st = float(st)
            if st <= now + 1e-9:
                continue
            for n in nodes:
                nid = n["id"]
                if nid not in sleeping_ids:
                    continue
                t_on_active = super()._transition_time(nid, "switching_on", "active")
                wake_time = st - float(t_on_active)
                prev = earliest_wake_by_node.get(nid)
                if prev is None or wake_time < prev:
                    earliest_wake_by_node[nid] = wake_time

        if not earliest_wake_by_node:
            return

        immediate = [nid for nid, t in earliest_wake_by_node.items() if t <= now + 1e-9]
        future_times = sorted({t for nid, t in earliest_wake_by_node.items() if t > now + 1e-9})

        # Immediate switch_on (same event type FCFS uses)
        if immediate:
            self.push_event(now, {"type": "switch_on", "nodes": immediate})

            # Optional: keep partitions consistent for the rest of this schedule tick
            def _filter_out(lst):
                ids = set(immediate)
                return [x for x in lst if x["id"] not in ids]

            self.sleeping = _filter_out(self.sleeping)
            state_by_id = {n["id"]: n for n in self.state}
            self.switching_on.extend([state_by_id[nid] for nid in immediate if nid in state_by_id])

        # Future wake triggers: just call scheduler then (no node list), FCFS-compatible
        for t in future_times:
            self.push_event(float(t), {"type": "call_me_later_so"})
