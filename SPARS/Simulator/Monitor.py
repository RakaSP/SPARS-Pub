from SPARS.Logger import log_info, log_trace
from typing import Dict, List



class Monitor:
    def __init__(self, platform_info, start_time):
        # Precompute ECR (energy cost rates) per node
        self.ecr: List[Dict] = [
            {
                "id": m["id"],
                "dvfs_profiles": {name: info["power"] for name, info in m["dvfs_profiles"].items()},
                "states": {state: m["states"][state]["power"] for state in m["states"]},
            }
            for m in platform_info["machines"]
        ]
        self.ecr_by_id: Dict[int, Dict] = {e["id"]: e for e in self.ecr}

        self.num_nodes = len(platform_info["machines"])
        self.node_ids = [node["id"] for node in platform_info["machines"]]

        # Energy tracking
        self.energy: List[Dict] = [
            {"id": i, "energy_consumption": 0.0, "energy_effective": 0.0,
                "energy_waste": 0.0, "last_update": start_time}
            for i in self.node_ids
        ]
        self.energy_by_id: Dict[int, Dict] = {e["id"]: e for e in self.energy}

        # Current node state
        self.nodes_state: List[Dict] = [
            {"id": i, "state": "active", "dvfs_mode": "base",
                "start_time": start_time, "duration": 0.0, "job_id": None}
            for i in self.node_ids
        ]
        self.nodes_state_by_id: Dict[int, Dict] = {
            n["id"]: n for n in self.nodes_state}

        # State history
        self.states_hist: List[Dict] = [
            {
                "id": i,
                "state_history": [
                    {"state": "active", "dvfs_mode": "base",
                        "start_time": start_time, "finish_time": 0.0}
                ],
            }
            for i in self.node_ids
        ]
        self.states_hist_by_id: Dict[int, Dict] = {
            h["id"]: h for h in self.states_hist}

        # Per-state duration (split active into idle/compute per DVFS)
        self.states_dur: List[Dict] = []
        for node in platform_info["machines"]:
            entry = {"id": node["id"]}
            for state in node["states"].keys():
                if state == "active":
                    entry["active_idle"] = {
                        dvfs: 0.0 for dvfs in node["dvfs_profiles"].keys()}
                    entry["active_compute"] = {
                        dvfs: 0.0 for dvfs in node["dvfs_profiles"].keys()}
                else:
                    entry[state] = {
                        dvfs: 0.0 for dvfs in node["dvfs_profiles"].keys()}
            self.states_dur.append(entry)
        self.states_dur_by_id: Dict[int, Dict] = {
            d["id"]: d for d in self.states_dur}

        # Job logs
        self.jobs_arrival_log: List[Dict] = []
        self.jobs_submission_log: List[Dict] = []
        self.jobs_execution_log: List[Dict] = []

        # Aggregate state counts over time
        self.state_switch: List[Dict] = []

        # ---------- debug prints ----------
    def print_energy(self):
        total_consumption = 0.0
        total_effective = 0.0
        total_waste = 0.0
        
        for e in self.energy:
            log_trace(
                f"Node {e['id']}: Energy Consumption = {e['energy_consumption']}, "
                f"Energy Effective = {e['energy_effective']}, Energy Waste = {e['energy_waste']}"
            )
            total_consumption += e['energy_consumption']
            total_effective += e['energy_effective']
            total_waste += e['energy_waste']
            
        log_info(
            f"TOTAL ENERGY: Consumption = {total_consumption}, "
            f"Effective = {total_effective}, Waste = {total_waste}"
        )

    def print_states_dur(self):
        total_duration = {}
        
        for d in self.states_dur:
            log_trace(f"Node {d['id']}: {d}")
            
            # Accumulate totals across all nodes
            for state, dvfs_dict in d.items():
                if state == "id":
                    continue
                if state not in total_duration:
                    total_duration[state] = {}
                
                for dvfs_mode, duration in dvfs_dict.items():
                    if dvfs_mode not in total_duration[state]:
                        total_duration[state][dvfs_mode] = 0.0
                    total_duration[state][dvfs_mode] += duration
        
        log_info(f"TOTAL STATES DURATION: {total_duration}")
    

    # ---------- end-of-sim finalize ----------
    def on_finish(self):
        # Close current segments into history
        for node_state in self.nodes_state:
            hist = self.states_hist_by_id[node_state["id"]]["state_history"]
            hist.append(
                {
                    "state": node_state["state"],
                    "start_time": node_state["start_time"],
                    "finish_time": node_state["start_time"] + node_state["duration"],
                    "dvfs_mode": node_state["dvfs_mode"],
                }
            )
        self.print_energy()
        self.print_states_dur()

    # ---------- main public entry ----------
    def record(
        self,
        mode,
        current_time=None,
        machines=None,
        record_job_arrival=None,
        record_job_submission=None,
        record_job_execution=None,
    ):
        if mode not in ("before", "after"):
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'before' or 'after'.")

        if mode == "before":
            if current_time is None:
                raise ValueError(
                    "`current_time` is required for mode 'before'.")
            self.update_node_state_duration(current_time)
            self.update_energy(current_time)
            return

        # mode == "after"
        if machines is None:
            raise ValueError("`machines` is required for mode 'after'.")
        if current_time is None:
            raise ValueError("`current_time` is required for mode 'after'.")

        # Extend logs (allow None)
        if record_job_arrival:
            self.jobs_arrival_log.extend(record_job_arrival)
        if record_job_submission:
            self.jobs_submission_log.extend(record_job_submission)
        if record_job_execution:
            for job in record_job_execution:
                job["finish_time"] = current_time
            self.jobs_execution_log.extend(record_job_execution)

        self.update_node_state(machines, current_time)

    # ---------- internals ----------
    def update_energy(self, current_time):
        """
        Accrue energy between the last energy update and current_time.
        Effective vs waste is determined by whether the node is computing.
        """
        for node in self.nodes_state:
            node_id = node["id"]
            e_entry = self.energy_by_id[node_id]
            ecr_entry = self.ecr_by_id.get(node_id)
            if not ecr_entry:
                continue

            # Correct and cheap time span
            dt = current_time - e_entry["last_update"]
            if dt <= 0:
                # nothing to accrue (clock not advanced)
                continue
            e_entry["last_update"] = current_time

            ecr_value = ecr_entry["states"][node["state"]]
            if ecr_value == "from_dvfs":
                ecr_value = ecr_entry["dvfs_profiles"][node["dvfs_mode"]]

            if (node['job_id'] is not None and node['state'] == 'active') or node['state'] == 'sleeping':
                e_entry["energy_effective"] += ecr_value * dt
            else:
                e_entry["energy_waste"] += ecr_value * dt

            e_entry["energy_consumption"] = e_entry["energy_effective"] + \
                e_entry["energy_waste"]

    def update_node_state_duration(self, current_time):
        """
        Increment per-state per-dvfs duration by the elapsed time since the last call.
        """
        for node in self.nodes_state:
            # elapsed since last (start_time + duration)
            delta = current_time - node["start_time"] - node["duration"]
            if delta <= 0:
                continue

            node["duration"] += delta
            state = node["state"]
            dvfs_mode = node["dvfs_mode"]

            if state == "active":
                state = "active_compute" if node["job_id"] is not None else "active_idle"

            dentry = self.states_dur_by_id[node["id"]]
            dentry[state][dvfs_mode] += delta

    def update_node_state(self, machines, current_time):
        """
        Sync nodes_state from `machines.nodes`, record history for changed nodes,
        and push an aggregate snapshot in `state_switch`.
        """
        # Build one-pass lookups from machines
        ms_nodes = machines.nodes
        state_by_id = {n["id"]: n["state"] for n in ms_nodes}
        job_by_id = {n["id"]: n["job_id"] for n in ms_nodes}
        dvfs_by_id = {n["id"]: n["dvfs_mode"] for n in ms_nodes}

        # Apply changes & write history for nodes that changed
        for node in self.nodes_state:
            nid = node["id"]
            new_state = state_by_id.get(nid, node["state"])
            new_job = job_by_id.get(nid, node["job_id"])
            new_dvfs = dvfs_by_id.get(nid, node["dvfs_mode"])

            if node["state"] != new_state or node["job_id"] != new_job or node["dvfs_mode"] != new_dvfs:
                hist = self.states_hist_by_id[nid]["state_history"]
                hist.append(
                    {
                        "state": node["state"],
                        "start_time": node["start_time"],
                        "finish_time": node["start_time"] + node["duration"],
                        "dvfs_mode": node["dvfs_mode"],
                    }
                )
                # reset current segment
                node["state"] = new_state
                node["job_id"] = new_job
                node["dvfs_mode"] = new_dvfs
                node["start_time"] = current_time
                node["duration"] = 0.0

        # Aggregate counts (single pass)
        nb_sleeping = sum(1 for n in ms_nodes if n["state"] == "sleeping")
        nb_switching_on = sum(
            1 for n in ms_nodes if n["state"] == "switching_on")
        nb_switching_off = sum(
            1 for n in ms_nodes if n["state"] == "switching_off")
        nb_idle = sum(
            1 for n in ms_nodes if n["state"] == "active" and n["job_id"] is None)
        nb_computing = sum(
            1 for n in ms_nodes if n["state"] == "active" and n["job_id"] is not None)

        self.state_switch.append(
            {
                "time": current_time,
                "nb_sleeping": nb_sleeping,
                "nb_switching_on": nb_switching_on,
                "nb_switching_off": nb_switching_off,
                "nb_idle": nb_idle,
                "nb_computing": nb_computing,
            }
        )
