from SPARS.Logger import log_info, log_trace
from typing import Dict, List
import json
import os

class Monitor:
    def __init__(self, machine, start_time):
        self.node_ids = list(machine.nodes.keys())

        self.energy: Dict[int, Dict] = {
            i: {"energy_consumption": 0.0, "energy_effective": 0.0,
                "energy_waste": 0.0, "last_update": start_time}
            for i in self.node_ids
        }

        self.state: Dict[int, Dict] = {
            i: {"state": "active", "dvfs_mode": "base",
                "start_time": start_time, "duration": 0.0, "job_id": None}
            for i in self.node_ids
        }

        self.states_hist: Dict[int, Dict] = {
            node_id: {
                "state_history": [],
            }
            for node_id in self.node_ids
        }

        self.states_dur: Dict[int, Dict] = {}

        for node_id, node in machine.nodes.items():
            node_name = node["node_name"]
            node_spec = machine.node_specs[node_name]

            entry = {}

            for state in node_spec["states"].keys():
                if state == "active":
                    entry["active_idle"] = {
                        dvfs: 0.0
                        for dvfs in node_spec["dvfs_profiles"].keys()
                    }

                    entry["active_compute"] = {
                        dvfs: 0.0
                        for dvfs in node_spec["dvfs_profiles"].keys()
                    }
                else:
                    entry[state] = {
                        dvfs: 0.0
                        for dvfs in node_spec["dvfs_profiles"].keys()
                    }

            self.states_dur[node_id] = entry

        self.jobs_arrival_log: List[Dict] = []
        self.jobs_submission_log: List[Dict] = []
        self.jobs_execution_log: List[Dict] = []

        self.state_switch: List[Dict] = []
        
        # Hanya untuk state_history spilling.
        self._state_hist_output_path = None
        self._state_hist_flush_every = 1000
        self._state_hist_events_since_flush = 0
        self._state_hist_flush_pending = False
        self._state_hist_batch_id = 0

    def print_energy(self):
        total_consumption = 0.0
        total_effective = 0.0
        total_waste = 0.0

        for nid, e in self.energy.items():
            log_trace(
                f"Node {nid}: Energy Consumption = {e['energy_consumption']}, "
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

        for nid, d in self.states_dur.items():
            log_trace(f"Node {nid}: {d}")

            for state, dvfs_dict in d.items():
                if state not in total_duration:
                    total_duration[state] = {}

                for dvfs_mode, duration in dvfs_dict.items():
                    if dvfs_mode not in total_duration[state]:
                        total_duration[state][dvfs_mode] = 0.0
                    total_duration[state][dvfs_mode] += duration

        log_info(f"TOTAL STATES DURATION: {total_duration}")

    def on_finish(self):
        for nid, node in self.state.items():
            hist = self.states_hist[nid]["state_history"]
            hist.append(
                {
                    "state": node["state"],
                    "start_time": node["start_time"],
                    "finish_time": node["start_time"] + node["duration"],
                    "dvfs_mode": node["dvfs_mode"],
                }
            )
        self.print_energy()
        self.print_states_dur()

    def record(
        self,
        mode,
        current_time=None,
        machine=None,
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
            self.update_energy(machine.nodes, current_time)
            return

        if machine is None:
            raise ValueError("`machines` is required for mode 'after'.")
        if current_time is None:
            raise ValueError("`current_time` is required for mode 'after'.")

        if record_job_arrival:
            self.jobs_arrival_log.extend(record_job_arrival)
        if record_job_submission:
            self.jobs_submission_log.extend(record_job_submission)
        if record_job_execution:
            for job in record_job_execution:
                job["finish_time"] = current_time
            self.jobs_execution_log.extend(record_job_execution)

        self.update_node_state(machine, current_time)

    def update_energy(self, nodes, current_time):
        for nid, node in self.state.items():
            e_entry = self.energy[nid]
            ecr = nodes[nid]["power"]

            dt = current_time - e_entry["last_update"]
            if dt <= 0:
                continue
            e_entry["last_update"] = current_time

            if (node['job_id'] is not None and node['state'] == 'active'):
                e_entry["energy_effective"] += ecr * dt
            else:
                e_entry["energy_waste"] += ecr * dt

            e_entry["energy_consumption"] = e_entry["energy_effective"] + \
                e_entry["energy_waste"]

    def update_node_state_duration(self, current_time):
        """
        Increment per-state per-dvfs duration by the elapsed time since the last call.
        """
        for nid, node in self.state.items():
            delta = current_time - node["start_time"] - node["duration"]
            if delta <= 0:
                continue

            node["duration"] += delta
            state = node["state"]
            dvfs_mode = node["dvfs_mode"]

            if state == "active":
                state = "active_compute" if node["job_id"] is not None else "active_idle"

            dentry = self.states_dur[nid]
            dentry[state][dvfs_mode] += delta

    def update_node_state(self, machines, current_time):
        ms_nodes = machines.nodes
        for nid, node in self.state.items():
            machine_node = ms_nodes.get(nid)
            if machine_node is None:
                continue
            new_state = machine_node["state"]
            new_job = machine_node["job_id"]
            new_dvfs = machine_node["dvfs_mode"]
            if node["state"] != new_state or node["job_id"] != new_job or node["dvfs_mode"] != new_dvfs:
                hist = self.states_hist[nid]["state_history"]
                hist.append({
                    "state": node["state"],
                    "start_time": node["start_time"],
                    "finish_time": node["start_time"] + node["duration"],
                    "dvfs_mode": node["dvfs_mode"],
                })
                node["state"] = new_state
                node["job_id"] = new_job
                node["dvfs_mode"] = new_dvfs
                node["start_time"] = current_time
                node["duration"] = 0.0
        nb_sleeping = sum(1 for n in ms_nodes.values() if n["state"] == "sleeping")
        nb_switching_on = sum(1 for n in ms_nodes.values() if n["state"] == "switching_on")
        nb_switching_off = sum(1 for n in ms_nodes.values() if n["state"] == "switching_off")
        nb_idle = sum(1 for n in ms_nodes.values() if n["state"] == "active" and n["job_id"] is None)
        nb_computing = sum(1 for n in ms_nodes.values() if n["state"] == "active" and n["job_id"] is not None)
        self.state_switch.append({
            "time": current_time,
            "nb_sleeping": nb_sleeping,
            "nb_switching_on": nb_switching_on,
            "nb_switching_off": nb_switching_off,
            "nb_idle": nb_idle,
            "nb_computing": nb_computing,
        })
    
    def configure_state_hist_spill(
        self,
        output_folder: str,
        flush_every: int = 1000,
    ) -> None:
        """
        Configure disk spilling untuk states_hist.

        File handle sengaja tidak disimpan sebagai atribut karena Monitor
        ikut di-copy.deepcopy() oleh HPCGymEnv.
        """
        if flush_every <= 0:
            raise ValueError("flush_every must be greater than zero")

        os.makedirs(output_folder, exist_ok=True)

        self._state_hist_output_path = os.path.join(
            output_folder,
            "state_history.jsonl",
        )

        self._state_hist_flush_every = int(flush_every)
        self._state_hist_events_since_flush = 0
        self._state_hist_flush_pending = False
        self._state_hist_batch_id = 0

        # Setiap simulator/epoch mendapatkan file baru.
        with open(
            self._state_hist_output_path,
            "w",
            encoding="utf-8",
        ):
            pass


    def note_processed_events(self, count: int) -> None:
        """
        Hanya menghitung event dan menandai flush sebagai pending.

        Tidak menulis file dan tidak membersihkan monitor di sini.
        """
        if count <= 0:
            return

        self._state_hist_events_since_flush += count

        if (
            self._state_hist_events_since_flush
            >= self._state_hist_flush_every
        ):
            self._state_hist_flush_pending = True


    def flush_state_hist_if_safe(
        self,
        force: bool = False,
    ) -> int:
        """
        Tulis closed state intervals ke disk lalu bersihkan hanya
        states_hist[*]["state_history"].

        Pada RL, fungsi ini hanya boleh dipanggil setelah next_obs dan
        reward selesai dihitung.
        """
        if self._state_hist_output_path is None:
            return 0

        if not force and not self._state_hist_flush_pending:
            return 0

        records = []

        next_batch_id = self._state_hist_batch_id + 1

        for node_id, node_history in self.states_hist.items():
            for interval in node_history["state_history"]:
                # Hindari interval kosong/invalid.
                if (
                    interval["finish_time"]
                    <= interval["start_time"]
                ):
                    continue

                records.append({
                    "batch_id": next_batch_id,
                    "node_id": node_id,
                    "state": interval["state"],
                    "dvfs_mode": interval["dvfs_mode"],
                    "start_time": interval["start_time"],
                    "finish_time": interval["finish_time"],
                })

        if records:
            # Semua write harus berhasil sebelum history di RAM dibersihkan.
            with open(
                self._state_hist_output_path,
                "a",
                encoding="utf-8",
            ) as file:
                for record in records:
                    file.write(
                        json.dumps(
                            record,
                            separators=(",", ":"),
                        )
                    )
                    file.write("\n")

            self._state_hist_batch_id = next_batch_id

        # Cleanup hanya state_history.
        for node_history in self.states_hist.values():
            node_history["state_history"].clear()

        if force:
            self._state_hist_events_since_flush = 0
        else:
            # Misalnya 2.300 event telah diproses:
            # setelah flush, 300 tetap diperhitungkan.
            self._state_hist_events_since_flush %= (
                self._state_hist_flush_every
            )

        self._state_hist_flush_pending = (
            self._state_hist_events_since_flush
            >= self._state_hist_flush_every
        )

        return len(records)


    def iter_state_hist_records(self):
        """
        Iterasi history yang sudah di-spill dan history yang masih di RAM.
        """
        if (
            self._state_hist_output_path is not None
            and os.path.exists(self._state_hist_output_path)
        ):
            with open(
                self._state_hist_output_path,
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    yield {
                        "node_id": record["node_id"],
                        "state": record["state"],
                        "dvfs_mode": record["dvfs_mode"],
                        "start_time": record["start_time"],
                        "finish_time": record["finish_time"],
                    }

        for node_id, node_history in self.states_hist.items():
            for interval in node_history["state_history"]:
                yield {
                    "node_id": node_id,
                    **interval,
                }


    def build_complete_states_hist(self) -> Dict[int, Dict]:
        """
        Bentuk kompatibilitas untuk process_node_job_data() yang lama.

        Perhatian: ini memuat kembali seluruh state history ke RAM, tetapi
        hanya saat final output, bukan selama simulasi atau deepcopy RL.
        """
        result = {
            node_id: {
                "state_history": [],
            }
            for node_id in self.node_ids
        }

        for record in self.iter_state_hist_records():
            node_id = record.pop("node_id")

            result[node_id]["state_history"].append(
                record
            )

        return result