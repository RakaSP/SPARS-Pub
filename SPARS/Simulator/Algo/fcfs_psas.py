# fcfs_psas.py
import math
import re
from .BasePSAS import BasePSAS

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


class FCFSPSAS(BasePSAS):
    """
    Node selection is energy-aware:
      Minimize ( sum(power) / min(compute_speed) ).
    Tie-breaks:
      1) Earliest Start Time
      2) Lower total power
    """

    def __init__(self, machines, jobs_manager, start_time, timeout=None):
        super().__init__(machines, jobs_manager, start_time, timeout)
        self.selected_list = []

    # ---------- public ----------
    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        # 1) current FCFS commit-only (no waking)
        started_now = self._current_fcfs_commit()
        
   

        # 2) future FCFS plan-only
        remaining = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        future_plan = self._future_fcfs_plan(remaining, barrier=now)
        
   

        self.selected_list = list(future_plan)

        # 3) wake callbacks
        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    # ---------------- current FCFS (commit-only) ----------------
    def _current_fcfs_commit(self):
        now = float(self.current_time)
        started_now = set()

        for job in self.waiting_queue[:]:
            req = int(job["res"])
            if req <= 0:
                continue

            # FCFS: if head cannot start now, stop
            if len(self.idle) < req:
                break

            res = self._select_nodes_energy_aware(
                required_nodes=req,
                _candidates=list(self.idle),
                releases_by_id=super()._releases_by_id(),
                min_start_time=now,
            )
            if res is None:
                break

            nodes, st = res
            if float(st) > now + EPS:
                break

            super().allocate(job, nodes)
            started_now.add(job["job_id"])

        return started_now

    # ---------------- future FCFS (plan-only) ----------------
    def _future_fcfs_plan(self, jobs, barrier):

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

        candidates = list(self.idle) + list(self.sleeping) + list(self.computing) + list(self.switching_on) + list(self.switching_off)

        plan = []
        barrier = float(barrier)

        for job in jobs:
            req = int(job["res"])
            if req <= 0:
                continue

            res = self._select_nodes_energy_aware(
                required_nodes=req,
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=barrier,
            )
            if res is None:
                break

            nodes, st = res
            ft = _append_planned_compute(job, nodes, float(st))
            plan.append((job, nodes, float(st), float(ft)))

            # FCFS ordering barrier
            barrier = float(st)

        return plan

    # ---------------- wake triggers from plan ----------------
    def _emit_wake_triggers_from_plan(self, plan):
        now = float(self.current_time)
   
        sleeping_ids = {n["id"] for n in self.sleeping}

        earliest_wake = {}
        for job, nodes, st, ft in plan:
            st = float(st)
            if st <= now + EPS:
                continue
            for n in nodes:
                nid = n["id"]
                if nid not in sleeping_ids:
                    continue
                lead = super()._wake_lead_time(nid)
                wake_time = st - float(lead)
                prev = earliest_wake.get(nid)
                if prev is None or wake_time < prev:
                    earliest_wake[nid] = wake_time

        if not earliest_wake:
            return

        immediate = [nid for nid, t in earliest_wake.items() if t <= now + EPS]
        future_times = sorted({t for nid, t in earliest_wake.items() if t > now + EPS})

        if immediate:
            self.push_event(now, {"type": "switch_on", "nodes": immediate})

            # keep partitions consistent this tick
            imm_set = set(immediate)
            self.sleeping = [n for n in self.sleeping if n["id"] not in imm_set]
            state_by_id = {n["id"]: n for n in self.state}
            self.switching_on.extend([state_by_id[nid] for nid in immediate if nid in state_by_id])

        for t in future_times:
            self.push_event(float(t), {"type": "call_me_later_so"})

    # ---------- internals ----------
    def _remaining_idle_timeout(self, node_id: int) -> float:
        if self.timeout is None:
            return math.inf

        for entry in self.timeout_list:
            if entry["node_id"] == node_id:
                return float(entry["time"] - self.current_time)

        return math.inf

    def _select_nodes_energy_aware(self, required_nodes: int, _candidates, releases_by_id=None, min_start_time=None):
        if releases_by_id is None:
            releases_by_id = super()._releases_by_id()

        _candidates = [
            n for n in _candidates
            if (n["id"] in releases_by_id) and (not math.isinf(float(releases_by_id[n["id"]]["release_time"])))
        ]
        if len(_candidates) < required_nodes:
            return None

        min_start_time = -math.inf if min_start_time is None else float(min_start_time)

        machine_by_id = {m["id"]: m for m in self.machines.machines}

        node_power_data = {}
        for node in _candidates:
            nid = node["id"]
            node_release = releases_by_id[nid]
            machine = machine_by_id[nid]

            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
            elif node["state"] == "active" and node.get("job_id") is not None:
                state_label = "computing"
            else:
                state_label = node["state"]

            base_energy_waste = 0.0
            for q in node_release["queue"]:
                if float(q["start_time"]) < self.current_time:
                    duration = float(q["finish_time"]) - self.current_time
                else:
                    duration = float(q["finish_time"]) - float(q["start_time"])

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
                        timeout_priority = 0.0

                    eligible.append((nid, float(cost), int(state_priority), float(timeout_priority)))

            if len(eligible) < required_nodes:
                continue

            ranked = sorted((cost, sp, tp, nid) for (nid, cost, sp, tp) in eligible)

            combo = []
            for cost, sp, tp, nid in ranked:
                combo.append(nid)
                if len(combo) == required_nodes:
                    break

            if len(combo) == required_nodes:
                node_objects = [node_power_data[nid]["node"] for nid in combo]
                return (node_objects, float(t))

        return None
