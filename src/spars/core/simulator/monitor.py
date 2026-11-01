import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from spars.core.logging import log_info, log_trace

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine


NodeState = dict[str, Any]
StateHistoryRecord = dict[str, Any]


class Monitor:
    def __init__(self, machine: "Machine", start_time: float) -> None:
        self.node_ids = list(machine.nodes)

        self.energy: dict[int, dict[str, float]] = {
            node_id: {
                "energy_consumption": 0.0,
                "energy_effective": 0.0,
                "energy_waste": 0.0,
                "last_update": start_time,
            }
            for node_id in self.node_ids
        }

        self.state: dict[int, NodeState] = {
            node_id: {
                "state": "active",
                "dvfs_mode": "base",
                "start_time": start_time,
                "duration": 0.0,
                "job_id": None,
            }
            for node_id in self.node_ids
        }

        self.states_hist: dict[int, dict[str, list[StateHistoryRecord]]] = {
            node_id: {"state_history": []}
            for node_id in self.node_ids
        }
        self.states_dur: dict[int, dict[str, dict[str, float]]] = {}

        for node_id, node in machine.nodes.items():
            node_name = node["node_name"]
            node_spec = machine.node_specs[node_name]
            state_durations: dict[str, dict[str, float]] = {}

            for state_name in node_spec["states"]:
                if state_name == "active":
                    state_durations["active_idle"] = {
                        dvfs_mode: 0.0
                        for dvfs_mode in node_spec["dvfs_profiles"]
                    }
                    state_durations["active_compute"] = {
                        dvfs_mode: 0.0
                        for dvfs_mode in node_spec["dvfs_profiles"]
                    }
                else:
                    state_durations[state_name] = {
                        dvfs_mode: 0.0
                        for dvfs_mode in node_spec["dvfs_profiles"]
                    }

            self.states_dur[node_id] = state_durations

        self.jobs_arrival_log: list[dict[str, Any]] = []
        self.jobs_submission_log: list[dict[str, Any]] = []
        self.jobs_execution_log: list[dict[str, Any]] = []
        self.state_switch: list[dict[str, Any]] = []

        # State history is spilled in batches to keep long simulations bounded
        # in memory. File handles are intentionally not kept on the instance
        # because Monitor is deep-copied by the RL environment.
        self._state_hist_output_path: str | None = None
        self._state_hist_flush_every = 1000
        self._state_hist_events_since_flush = 0
        self._state_hist_flush_pending = False
        self._state_hist_batch_id = 0

    # Diagnostics
    def print_energy(self) -> None:
        total_consumption = 0.0
        total_effective = 0.0
        total_waste = 0.0

        for node_id, energy_entry in self.energy.items():
            log_trace(
                f"Node {node_id}: Energy Consumption = "
                f"{energy_entry['energy_consumption']}, "
                f"Energy Effective = {energy_entry['energy_effective']}, "
                f"Energy Waste = {energy_entry['energy_waste']}"
            )
            total_consumption += energy_entry["energy_consumption"]
            total_effective += energy_entry["energy_effective"]
            total_waste += energy_entry["energy_waste"]

        log_info(
            f"TOTAL ENERGY: Consumption = {total_consumption}, "
            f"Effective = {total_effective}, Waste = {total_waste}"
        )

    def print_states_dur(self) -> None:
        total_duration: dict[str, dict[str, float]] = {}

        for node_id, state_durations in self.states_dur.items():
            log_trace(f"Node {node_id}: {state_durations}")

            for state_name, dvfs_durations in state_durations.items():
                if state_name not in total_duration:
                    total_duration[state_name] = {}

                for dvfs_mode, duration in dvfs_durations.items():
                    if dvfs_mode not in total_duration[state_name]:
                        total_duration[state_name][dvfs_mode] = 0.0
                    total_duration[state_name][dvfs_mode] += duration

        log_info(f"TOTAL STATES DURATION: {total_duration}")

    def on_finish(self) -> None:
        for node_id, node in self.state.items():
            history = self.states_hist[node_id]["state_history"]
            history.append({
                "state": node["state"],
                "start_time": node["start_time"],
                "finish_time": node["start_time"] + node["duration"],
                "dvfs_mode": node["dvfs_mode"],
            })

        self.print_energy()
        self.print_states_dur()

    # Simulation recording
    def record(
        self,
        mode: str,
        current_time: float | None = None,
        machine: "Machine" = None,
        record_job_arrival: Sequence[dict[str, Any]] | None = None,
        record_job_submission: Sequence[dict[str, Any]] | None = None,
        record_job_execution: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if mode not in ("before", "after"):
            raise ValueError(
                f"Invalid mode {mode!r}. Expected 'before' or 'after'."
            )

        if current_time is None:
            raise ValueError(
                f"current_time is required for mode {mode!r}."
            )

        if machine is None:
            raise ValueError(
                f"machine is required for mode {mode!r}."
            )

        if mode == "before":
            self.update_node_state_duration(current_time)
            self.update_energy(machine.nodes, current_time)
            return

        if record_job_arrival:
            self.jobs_arrival_log.extend(record_job_arrival)

        if record_job_submission:
            self.jobs_submission_log.extend(record_job_submission)

        if record_job_execution:
            for job in record_job_execution:
                job["finish_time"] = current_time
            self.jobs_execution_log.extend(record_job_execution)

        self.update_node_state(machine, current_time)

    def update_energy(
        self,
        nodes: Mapping[int, Mapping[str, Any]],
        current_time: float,
    ) -> None:
        for node_id, node in self.state.items():
            energy_entry = self.energy[node_id]
            energy_rate = nodes[node_id]["power"]

            elapsed = current_time - energy_entry["last_update"]
            if elapsed <= 0:
                continue

            energy_entry["last_update"] = current_time

            if (
                node["job_id"] is not None
                and node["state"] == "active"
            ):
                energy_entry["energy_effective"] += (
                    energy_rate * elapsed
                )
            else:
                energy_entry["energy_waste"] += (
                    energy_rate * elapsed
                )

            energy_entry["energy_consumption"] = (
                energy_entry["energy_effective"]
                + energy_entry["energy_waste"]
            )

    def update_node_state_duration(
        self,
        current_time: float,
    ) -> None:
        """Accumulate elapsed time for each node state and DVFS mode."""
        for node_id, node in self.state.items():
            elapsed = (
                current_time
                - node["start_time"]
                - node["duration"]
            )
            if elapsed <= 0:
                continue

            node["duration"] += elapsed
            state_name = node["state"]
            dvfs_mode = node["dvfs_mode"]

            if state_name == "active":
                state_name = (
                    "active_compute"
                    if node["job_id"] is not None
                    else "active_idle"
                )

            self.states_dur[node_id][state_name][dvfs_mode] += elapsed

    def update_node_state(
        self,
        machine: "Machine",
        current_time: float,
    ) -> None:
        machine_nodes = machine.nodes

        for node_id, monitored_node in self.state.items():
            machine_node = machine_nodes.get(node_id)
            if machine_node is None:
                continue

            new_state = machine_node["state"]
            new_job_id = machine_node["job_id"]
            new_dvfs_mode = machine_node["dvfs_mode"]

            state_changed = (
                monitored_node["state"] != new_state
                or monitored_node["job_id"] != new_job_id
                or monitored_node["dvfs_mode"] != new_dvfs_mode
            )
            if not state_changed:
                continue

            history = self.states_hist[node_id]["state_history"]
            history.append({
                "state": monitored_node["state"],
                "start_time": monitored_node["start_time"],
                "finish_time": (
                    monitored_node["start_time"]
                    + monitored_node["duration"]
                ),
                "dvfs_mode": monitored_node["dvfs_mode"],
            })

            monitored_node["state"] = new_state
            monitored_node["job_id"] = new_job_id
            monitored_node["dvfs_mode"] = new_dvfs_mode
            monitored_node["start_time"] = current_time
            monitored_node["duration"] = 0.0

        self.state_switch.append({
            "time": current_time,
            "nb_sleeping": sum(
                1
                for node in machine_nodes.values()
                if node["state"] == "sleeping"
            ),
            "nb_switching_on": sum(
                1
                for node in machine_nodes.values()
                if node["state"] == "switching_on"
            ),
            "nb_switching_off": sum(
                1
                for node in machine_nodes.values()
                if node["state"] == "switching_off"
            ),
            "nb_idle": sum(
                1
                for node in machine_nodes.values()
                if (
                    node["state"] == "active"
                    and node["job_id"] is None
                )
            ),
            "nb_computing": sum(
                1
                for node in machine_nodes.values()
                if (
                    node["state"] == "active"
                    and node["job_id"] is not None
                )
            ),
        })

    # State-history spilling
    def configure_state_hist_spill(
        self,
        output_folder: str,
        flush_every: int = 1000,
    ) -> None:
        """Configure periodic state-history spilling to ``state_history.jsonl``."""
        if isinstance(flush_every, bool) or not isinstance(flush_every, int):
            raise TypeError("flush_every must be an integer")
        if flush_every <= 0:
            raise ValueError("flush_every must be greater than zero")

        os.makedirs(output_folder, exist_ok=True)

        self._state_hist_output_path = os.path.join(
            output_folder,
            "state_history.jsonl",
        )
        self._state_hist_flush_every = flush_every
        self._state_hist_events_since_flush = 0
        self._state_hist_flush_pending = False
        self._state_hist_batch_id = 0

        # Each simulator/epoch owns a fresh spill file.
        with open(
            self._state_hist_output_path,
            "w",
            encoding="utf-8",
        ):
            pass

    def note_processed_events(self, count: int) -> None:
        """Track processed events and mark when a safe spill is due."""
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
        """Write closed intervals, then clear only in-memory state history."""
        if self._state_hist_output_path is None:
            return 0

        if not force and not self._state_hist_flush_pending:
            return 0

        records: list[dict[str, Any]] = []
        next_batch_id = self._state_hist_batch_id + 1

        for node_id, node_history in self.states_hist.items():
            for interval in node_history["state_history"]:
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
            # Clear RAM only after every record has been written successfully.
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

        for node_history in self.states_hist.values():
            node_history["state_history"].clear()

        if force:
            self._state_hist_events_since_flush = 0
        else:
            # Preserve the remainder after a batch-sized flush threshold.
            self._state_hist_events_since_flush %= (
                self._state_hist_flush_every
            )

        self._state_hist_flush_pending = (
            self._state_hist_events_since_flush
            >= self._state_hist_flush_every
        )

        return len(records)

    def iter_state_hist_records(
        self,
    ) -> Iterator[StateHistoryRecord]:
        """Yield spilled state history first, followed by in-memory history."""
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

    def build_complete_states_hist(
        self,
    ) -> dict[int, dict[str, list[StateHistoryRecord]]]:
        """Rebuild the legacy complete-history structure for final output."""
        complete_history = {
            node_id: {"state_history": []}
            for node_id in self.node_ids
        }

        for record in self.iter_state_hist_records():
            node_id = record.pop("node_id")
            complete_history[node_id]["state_history"].append(
                record
            )

        return complete_history
