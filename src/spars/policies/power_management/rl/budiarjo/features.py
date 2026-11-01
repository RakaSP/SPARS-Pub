from typing import TYPE_CHECKING

import numpy as np

OBSERVATION_KIND = "nodes"
OBSERVATION_DIM = 11

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator


_STATE_CODE = {
    "active": 0,
    "sleeping": 1,
    "switching_on": 2,
    "switching_off": 3,
}


def feature_extraction(
    simulator: "Simulator",
    training: bool = False,
) -> np.ndarray:
    """Build the per-node 11-feature observation used by Budiarjo."""
    if simulator.rl_dt is None or simulator.rl_dt <= 0:
        raise ValueError("Budiarjo features require a positive rl_dt")

    current_time = simulator.current_time
    waiting_queue = simulator.jobs_manager.waiting_queue
    active_jobs = simulator.jobs_manager.active_jobs
    nodes = simulator.platform_control.machine.nodes
    monitor = simulator.monitor

    queue_size = len(waiting_queue)

    # Arrival rate is measured over the previous RL decision interval.
    window_start = current_time - simulator.rl_dt
    arrivals_in_window = sum(
        window_start < job["subtime"] <= current_time
        for job in monitor.jobs_arrival_log
    )
    arrival_rate = arrivals_in_window / simulator.rl_dt

    if waiting_queue:
        mean_waiting_time = (
            sum(current_time - job["subtime"] for job in waiting_queue) / queue_size
        )
    else:
        mean_waiting_time = 0.0

    total_energy_waste = sum(
        energy["energy_waste"] for energy in monitor.energy.values()
    )

    if waiting_queue:
        mean_requested_time = sum(job["reqtime"] for job in waiting_queue) / queue_size
    else:
        mean_requested_time = 0.0

    remaining_runtime_fraction_by_node = {node_id: 0.0 for node_id in nodes}

    for job in active_jobs:
        remaining_runtime_fraction = (
            job["start_time"] + job["reqtime"] - current_time
        ) / job["reqtime"]

        for node_id in job["nodes"]:
            remaining_runtime_fraction_by_node[node_id] = remaining_runtime_fraction

    features = []

    for node_id, node in nodes.items():
        power_state = _STATE_CODE[node["state"]]
        is_computing = int(node["state"] == "active" and node["job_id"] is not None)
        is_idle = node["state"] == "active" and node["job_id"] is None
        consecutive_idle_time = (
            current_time - monitor.state[node_id]["start_time"] if is_idle else 0.0
        )
        remaining_runtime_fraction = remaining_runtime_fraction_by_node[node_id]
        node_energy_waste = monitor.energy[node_id]["energy_waste"]
        switching_time = sum(
            monitor.states_dur[node_id]["switching_on"].values()
        ) + sum(monitor.states_dur[node_id]["switching_off"].values())

        features.append(
            [
                queue_size,
                arrival_rate,
                mean_waiting_time,
                total_energy_waste,
                mean_requested_time,
                power_state,
                is_computing,
                consecutive_idle_time,
                remaining_runtime_fraction,
                node_energy_waste,
                switching_time,
            ]
        )

    return np.asarray(features, dtype=np.float32)
