from math import inf
import math
from .BasePSAS import BasePSAS
import re
_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


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
        # Track scheduled jobs with their nodes, start_time, and finish_time
        self.selected_list = []

    # ---------- public ----------
    def schedule(self):

        super().prep_schedule()

        self.FCFSPSAS()

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    def FCFSPSAS(self, plan_only: bool = False):
        """
        If plan_only=True:
        - build full FCFS plan (including future) into self.selected_list
        - DO NOT allocate, DO NOT schedule switch_on/call_me_later_so

        If plan_only=False (default):
        - current behavior (allocate now + schedule switch_on for future)
        """
        fcfs_scheduled_jobs = set()

        # --- planning copy: scheduled_node_release (dict), same shape as next_releases entries ---
        base_by_id = super()._releases_by_id()
        scheduled_by_id = {
            nid: {
                "node_id": nid,
                "queue": [dict(seg) for seg in base_by_id[nid]["queue"]],
                "release_time": float(base_by_id[nid]["release_time"]),
            }
            for nid in base_by_id
        }

        def _planned_release(node_obj):
            return float(scheduled_by_id[node_obj["id"]]["release_time"])

        def _append_planned_compute(job, selected_nodes, job_start_time):
            compute_speed = min(float(n["compute_speed"]) for n in selected_nodes)
            walltime = float(job["reqtime"]) / compute_speed
            finish_time = float(job_start_time) + walltime

            phase = f'compute(job={job["job_id"]})'
            for n in selected_nodes:
                entry = scheduled_by_id[n["id"]]
                entry["queue"].append({
                    "phase": phase,
                    "start_time": float(job_start_time),
                    "finish_time": float(finish_time),
                })
                entry["release_time"] = float(finish_time)

            return float(finish_time)

        job_schedules = []
        barrier = float(self.current_time)

        for job in self.waiting_queue[:]:
            required = int(job["res"])
            min_start_time = barrier

            # 1) Prefer idle-now if possible
            if min_start_time <= self.current_time:
                idle_now = [
                    n for n in self.idle
                    if (not math.isinf(_planned_release(n))) and (_planned_release(n) <= self.current_time)
                ]
                if len(idle_now) >= required:
                    selected = idle_now[:required]
                    start_time = float(self.current_time)
                    finish_time = _append_planned_compute(job, selected, start_time)

                    job_schedules.append((job, selected, start_time, finish_time))
                    fcfs_scheduled_jobs.add(job["job_id"])
                    barrier = start_time
                    continue

            # 2) Otherwise allow all nodes (including already planned)
            candidates = list(self.idle) + list(self.sleeping) + list(self.computing) + list(self.switching_on)

            result = self._select_nodes_energy_aware(
                required_nodes=required,
                _candidates=candidates,
                releases_by_id=scheduled_by_id,
                min_start_time=min_start_time,
            )
            if result is None:
                break

            selected, start_time = result
            finish_time = _append_planned_compute(job, selected, start_time)

            job_schedules.append((job, selected, float(start_time), float(finish_time)))
            fcfs_scheduled_jobs.add(job["job_id"])
            barrier = float(start_time)

        # Always store plan
        self.selected_list = list(job_schedules)

        # PLAN-ONLY: do not emit any actions/events
        if plan_only:
            return fcfs_scheduled_jobs

        # APPLY: allocate now, schedule switch_on for future
        for job, selected, start_time, finish_time in job_schedules:
            if float(start_time) <= self.current_time:
                super().allocate(job, selected)
            else:
                selected_ids = [n["id"] for n in selected]
                sleeping_ids = {n["id"] for n in self.sleeping}
                switch_on_nodes = [nid for nid in selected_ids if nid in sleeping_ids]
                if switch_on_nodes:
                    self._schedule_switch_on_events(job, selected, switch_on_nodes, float(start_time))

        return fcfs_scheduled_jobs




    def _schedule_switch_on_events(self, job, selected_nodes, switch_on_nodes, job_start_time):
        """
        Schedule switch_on events using call_me_later for future events
        and immediate switch_on for current time events.
        """
        immediate_switch_on = []
        future_switch_on_times = set()

        for node_id in switch_on_nodes:
            # Calculate when to start switching on this node
            switch_on_duration = super()._transition_time(node_id, 'switching_on', 'active')
            switch_on_start_time = job_start_time - switch_on_duration

            if switch_on_start_time <= self.current_time:
                # Immediate switch_on
                immediate_switch_on.append(node_id)
            else:
                # Future switch_on - schedule call_me_later
                future_switch_on_times.add(switch_on_start_time)

        # Handle immediate switch_on

        if immediate_switch_on:
            def _filter_out(lst): return [
                n for n in lst if n['id'] not in immediate_switch_on]
            self.sleeping = _filter_out(self.sleeping)
            state_by_id = {n['id']: n for n in self.state}
            switch_on_nodes_list = []
            for node_id in immediate_switch_on:
                switch_on_nodes_list.append(state_by_id[node_id])
            self.switching_on.extend(switch_on_nodes_list)

            self.push_event(self.current_time, {
                'type': 'switch_on',
                'nodes': immediate_switch_on
            })

        # Handle future switch_on via call_me_later
        for switch_on_time in future_switch_on_times:
            self.push_event(switch_on_time, {
                'type': 'call_me_later_so'
            })

    # ---------- internals ----------

    def _remaining_idle_timeout(self, node_id: int) -> float:
        """
        Remaining time until this idle node would be switched off by timeout_policy.
        If not tracked, return a large number so it sorts to the end.
        """
        if self.timeout is None:
            return math.inf

        for entry in self.timeout_list:
            if entry["node_id"] == node_id:
                remaining = float(entry["time"] - self.current_time)
                return remaining

        return math.inf

    def _select_nodes_energy_aware(self, required_nodes: int, _candidates, releases_by_id=None, min_start_time=None):
        if releases_by_id is None:
            releases_by_id = super()._releases_by_id()

        # filter: must exist + finite release_time
        _candidates = [
            n for n in _candidates
            if (n["id"] in releases_by_id) and (not math.isinf(float(releases_by_id[n["id"]]["release_time"])))
        ]
        if len(_candidates) < required_nodes:
            return None

        if min_start_time is None:
            min_start_time = -math.inf
        else:
            min_start_time = float(min_start_time)

        machine_by_id = {m["id"]: m for m in self.machines.machines}

        node_power_data = {}
        for node in _candidates:
            nid = node["id"]
            node_release = releases_by_id[nid]
            machine = machine_by_id[nid]

            # derive a consistent label (because node['state'] is 'active', but you want idle/computing)
            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
            elif node["state"] == "active" and node.get("job_id") is not None:
                state_label = "computing"
            else:
                state_label = node["state"]

            base_energy_waste = 0.0
            for q in node_release["queue"]:
                if q["start_time"] < self.current_time:
                    duration = q["finish_time"] - self.current_time
                else:
                    duration = q["finish_time"] - q["start_time"]

                if _COMPUTE_RE.fullmatch(str(q["phase"])):
                    continue

                e_rate = machine["states"][q["phase"]]["power"]
                if e_rate == "from_dvfs":
                    dvfs_profiles = machine["dvfs_profiles"]
                    dvfs_mode = node["dvfs_mode"]
                    e_rate = dvfs_profiles[dvfs_mode]["power"]

                base_energy_waste += e_rate * duration

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
                        timeout_priority = 0

                    eligible.append((nid, cost, state_priority, timeout_priority))

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
                return (node_objects, t)

        return None
