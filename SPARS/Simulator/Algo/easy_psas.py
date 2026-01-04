from __future__ import annotations

import math
import re
from .fcfs_psas import FCFSPSAS

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


class EASYPSAS(FCFSPSAS):
    """
    EASY backfilling:
      - Run FCFS first (may reserve future jobs within this scheduling tick)
      - Then backfill jobs that won't delay the first unscheduled job (head job),
        using a planned release table (scheduled_node_release) that includes FCFS reservations.
    """

    # ---------- public ----------
    def schedule(self):
        super().prep_schedule()

        # 1) FCFS plan + (commit-now allocations + switch_on events) as implemented in FCFSPSAS
        fcfs_scheduled_jobs = super().FCFSPSAS()

        # 2) EASY backfill using planned releases that include FCFS reservations
        self.backfill(fcfs_scheduled_jobs)

        if self.timeout is not None:
            super().timeout_policy()

        super().build_callbacks()
        return self.events

    # ---------- helpers: planned release table ----------
    def _append_planned_compute(self, job, selected_nodes, job_start_time, releases_by_id):
        """
        Append compute phases into a planned releases table (scheduled_node_release).
        Ensures node release_time becomes the last scheduled finish_time.
        """
        compute_speed = min(float(n["compute_speed"]) for n in selected_nodes)
        walltime = float(job["reqtime"]) / compute_speed

        # Ensure start_time is not earlier than any node's current planned release
        cursor = max(float(releases_by_id[n["id"]]["release_time"]) for n in selected_nodes)
        st = max(float(job_start_time), cursor)
        ft = st + walltime

        phase = f'compute(job={job["job_id"]})'
        for n in selected_nodes:
            entry = releases_by_id[n["id"]]
            entry["queue"].append(
                {"phase": phase, "start_time": float(st), "finish_time": float(ft)}
            )
            entry["release_time"] = float(ft)

        return float(st), float(ft)

    def _build_scheduled_node_release_from_fcfs(self, fcfs_selected_list):
        """
        Build scheduled_node_release (dict by node_id) starting from current next_releases,
        then append ONLY FCFS future reservations (start_time > now).
        """
        base_by_id = super()._releases_by_id()

        scheduled_by_id = {
            nid: {
                "node_id": nid,
                "queue": [dict(seg) for seg in base_by_id[nid]["queue"]],
                "release_time": float(base_by_id[nid]["release_time"]),
            }
            for nid in base_by_id
        }

        # FCFS already committed start<=now via allocate() -> those compute phases already exist in base queues.
        # We only extend the plan with FCFS future reservations.
        for job, nodes, start_time, _finish_time in sorted(fcfs_selected_list, key=lambda x: float(x[2])):
            if float(start_time) <= self.current_time:
                continue
            self._append_planned_compute(job, nodes, float(start_time), scheduled_by_id)

        return scheduled_by_id

    def _protected_start_times(self, fcfs_selected_list, head_nodes=None, head_start_time=None):
        """
        For each node, record the earliest future reserved start_time that must not be delayed.
        Includes:
          - FCFS future reservations
          - Head job reservation (EASY rule), if provided
        """
        protected = {}

        for _job, nodes, start_time, _finish_time in fcfs_selected_list:
            st = float(start_time)
            if st <= self.current_time:
                continue
            for n in nodes:
                nid = int(n["id"])
                protected[nid] = min(protected.get(nid, math.inf), st)

        if head_nodes is not None and head_start_time is not None:
            st = float(head_start_time)
            for n in head_nodes:
                nid = int(n["id"])
                protected[nid] = min(protected.get(nid, math.inf), st)

        return protected

    # ---------- EASY backfill ----------
    def backfill(self, fcfs_started_jobs):
        """
        EASY backfill:
        - FCFS already started runnable jobs now and reserved exactly ONE head job in selected_list.
        - Backfill jobs behind head if they can start now and won't delay head start.
        """
        now = float(self.current_time)

        # Find head reservation created by FCFS: first tuple with start_time > now
        head_entry = None
        for job, nodes, st, ft in self.selected_list:
            if float(st) > now and job["job_id"] not in fcfs_started_jobs:
                head_entry = (job, nodes, float(st), float(ft))
                break

        if head_entry is None:
            return  # no head reserved => nothing to protect/backfill against

        head_job, head_nodes, head_start, _head_finish = head_entry
        head_job_id = head_job["job_id"]
        head_node_ids = {int(n["id"]) for n in head_nodes}

        # If head_start is unknown/infinite, be conservative: don't backfill
        if math.isinf(head_start) or head_start <= now:
            return

        # Only backfill jobs BEHIND the head job in queue order
        seen_head = False
        for job in self.waiting_queue[:]:
            if job["job_id"] == head_job_id:
                seen_head = True
                continue
            if not seen_head:
                continue

            # Skip jobs already started by FCFS
            if job["job_id"] in fcfs_started_jobs:
                continue

            required = int(job["res"])
            if len(self.idle) < required:
                continue

            # time window if we end up using head nodes
            window = head_start - now
            if window <= 0:
                return

            # Minimum per-node speed needed so finish <= head_start (since runtime uses min speed)
            min_speed_needed = float(job["reqtime"]) / float(window)

            releases_by_id = super()._releases_by_id()

            # Step A: prefer NOT using head nodes at all
            non_head_idle = [n for n in self.idle if int(n["id"]) not in head_node_ids]
            if len(non_head_idle) >= required:
                res = self._select_nodes_energy_aware(
                    required_nodes=required,
                    _candidates=non_head_idle,
                    releases_by_id=releases_by_id,
                    min_start_time=now,
                )
                if res is not None:
                    selected, st = res
                    if float(st) <= now:
                        super().allocate(job, selected)
                        # track schedule
                        speed = min(float(n["compute_speed"]) for n in selected)
                        ft = now + float(job["reqtime"]) / speed
                        self.selected_list.append((job, selected, now, ft))
                        continue

            # Step B: allow using head nodes IF (all chosen nodes) are fast enough to finish before head_start
            eligible = [n for n in self.idle if float(n["compute_speed"]) >= min_speed_needed]
            if len(eligible) < required:
                continue

            # still prefer non-head among eligible
            eligible_non_head = [n for n in eligible if int(n["id"]) not in head_node_ids]
            pool = eligible_non_head if len(eligible_non_head) >= required else eligible

            res = self._select_nodes_energy_aware(
                required_nodes=required,
                _candidates=pool,
                releases_by_id=releases_by_id,
                min_start_time=now,
            )
            if res is None:
                continue

            selected, st = res
            if float(st) > now:
                continue

            # Final safety check: if any head node used, must finish before head_start
            speed = min(float(n["compute_speed"]) for n in selected)
            finish = now + float(job["reqtime"]) / speed
            if any(int(n["id"]) in head_node_ids for n in selected):
                if finish > head_start:
                    continue

            super().allocate(job, selected)
            self.selected_list.append((job, selected, now, finish))


    # ---------- backfill selector (planned-release aware) ----------
    def _backfill_select_nodes_energy_aware(
        self,
        job,
        required_nodes: int,
        candidates,
        protected_starts: dict[int, float],
        releases_by_id: dict,
        min_start_time: float | None = None,
        max_start_time: float | None = None,
    ):
        """
        Like _select_nodes_energy_aware but with constraint:
          finish_time <= min(protected_starts[node_id]) across selected nodes.
        Uses a planned releases table (releases_by_id) so it includes “currently scheduled” nodes.
        """
        # filter: must exist + finite release_time
        candidates = [
            n for n in candidates
            if (n["id"] in releases_by_id) and (not math.isinf(float(releases_by_id[n["id"]]["release_time"])))
        ]
        if len(candidates) < required_nodes:
            return None

        if min_start_time is None:
            min_start_time = -math.inf
        else:
            min_start_time = float(min_start_time)

        if max_start_time is not None:
            max_start_time = float(max_start_time)

        machine_by_id = {m["id"]: m for m in self.machines.machines}

        node_power_data = {}
        for node in candidates:
            nid = int(node["id"])
            node_release = releases_by_id[nid]
            machine = machine_by_id[nid]

            # consistent label (idle/computing) for active state
            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
            elif node["state"] == "active" and node.get("job_id") is not None:
                state_label = "computing"
            else:
                state_label = node["state"]

            base_energy_waste = 0.0
            for q in node_release["queue"]:
                ft = float(q["finish_time"])
                if math.isinf(ft):
                    # if non-compute phase is inf (shouldn't happen), treat as not usable
                    if not _COMPUTE_RE.fullmatch(str(q["phase"])):
                        base_energy_waste = math.inf
                        break
                    continue

                if float(q["start_time"]) < self.current_time:
                    duration = ft - self.current_time
                else:
                    duration = ft - float(q["start_time"])

                if _COMPUTE_RE.fullmatch(str(q["phase"])):
                    continue

                e_rate = machine["states"][q["phase"]]["power"]
                if e_rate == "from_dvfs":
                    dvfs_profiles = machine["dvfs_profiles"]
                    dvfs_mode = node["dvfs_mode"]
                    e_rate = dvfs_profiles[dvfs_mode]["power"]

                base_energy_waste += float(e_rate) * float(duration)

            idle_power = machine["states"]["active"]["power"]
            if idle_power == "from_dvfs":
                dvfs_profiles = machine["dvfs_profiles"]
                dvfs_mode = node["dvfs_mode"]
                idle_power = dvfs_profiles[dvfs_mode]["power"]

            node_power_data[nid] = {
                "base": float(base_energy_waste),
                "idle": float(idle_power),
                "release": float(node_release["release_time"]),
                "state_label": state_label,
                "node": node,
            }

        releases_sorted = sorted({d["release"] for d in node_power_data.values()} | {min_start_time})
        items = list(node_power_data.items())

        for t in releases_sorted:
            if t < min_start_time:
                continue
            if (max_start_time is not None) and (t >= max_start_time):
                continue

            eligible = []
            for nid, dat in items:
                r = dat["release"]
                if r <= t:
                    if dat["state_label"] in ("switching_off", "sleeping"):
                        cost = dat["base"]
                    else:
                        cost = dat["base"] + dat["idle"] * (t - r)

                    state = dat["state_label"]
                    if state == "idle":
                        state_priority = 0
                    elif state == "computing":
                        state_priority = 1
                    elif state == "switching_on":
                        state_priority = 2
                    else:
                        state_priority = 3

                    if state == "idle":
                        timeout_priority = -self._remaining_idle_timeout(nid)
                    else:
                        timeout_priority = 0

                    eligible.append((nid, cost, state_priority, timeout_priority))

            if len(eligible) < required_nodes:
                continue

            ranked = sorted((cost, sp, tp, nid) for (nid, cost, sp, tp) in eligible)

            combo_ids = [nid for (_c, _sp, _tp, nid) in ranked[:required_nodes]]
            selected_nodes = [node_power_data[nid]["node"] for nid in combo_ids]

            # finish time check against protected starts (earliest reserved start per node)
            compute_speed = min(float(n["compute_speed"]) for n in selected_nodes)
            walltime = float(job["reqtime"]) / compute_speed
            finish_time = float(t) + walltime

            max_allowed_finish = min(protected_starts.get(int(n["id"]), math.inf) for n in selected_nodes)

            if finish_time <= max_allowed_finish:
                return (selected_nodes, float(t))

        return None
