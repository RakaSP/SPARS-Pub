import numpy as np


def feature_extraction(simulator, training: bool = False):
    current_time = simulator.current_time
    waiting_queue = simulator.jobs_manager.waiting_queue
    active_jobs = simulator.jobs_manager.active_jobs
    nodes = simulator.platform_control.machine.nodes
    monitor = simulator.monitor

    # f1: number of jobs in queue
    f1 = len(waiting_queue)

    # f2: current arrival rate during the previous action interval
    window_start = current_time - simulator.rl_dt

    arrivals = sum(
        window_start < job["subtime"] <= current_time
        for job in monitor.jobs_arrival_log
    )

    f2 = arrivals / simulator.rl_dt

    # f3: average waiting time of jobs currently in queue
    if waiting_queue:
        f3 = sum(
            current_time - job["subtime"]
            for job in waiting_queue
        ) / f1
    else:
        f3 = 0.0

    # f4: total wasted energy
    f4 = sum(
        energy["energy_waste"]
        for energy in monitor.energy.values()
    )

    # f5: average requested time of jobs currently in queue
    if waiting_queue:
        f5 = sum(
            job["reqtime"]
            for job in waiting_queue
        ) / f1
    else:
        f5 = 0.0

    state_code = {
        "active": 0,
        "sleeping": 1,
        "switching_on": 2,
        "switching_off": 3,
    }

    # f9: remaining requested runtime percentage
    release_time = {
        node_id: 0.0
        for node_id in nodes
    }

    for job in active_jobs:
        remaining_release_percent = (
            job["start_time"] + job["reqtime"] - current_time
        ) / job["reqtime"]

        for node_id in job["nodes"]:
            release_time[node_id] = remaining_release_percent

    features = []

    for node_id, node in nodes.items():
        # f6: power state of node m
        f6 = state_code[node["state"]]

        # f7: computing flag
        f7 = int(
            node["state"] == "active"
            and node["job_id"] is not None
        )

        # f8: current consecutive idle time of node m
        is_idle = (
            node["state"] == "active"
            and node["job_id"] is None
        )

        f8 = (
            current_time - monitor.state[node_id]["start_time"]
            if is_idle
            else 0.0
        )

        # f9: remaining requested runtime percentage
        f9 = release_time[node_id]

        # f10: wasted energy of node m
        f10 = monitor.energy[node_id]["energy_waste"]

        # f11: total time used for switching on and off
        f11 = (
            sum(
                monitor.states_dur[node_id]["switching_on"].values()
            )
            + sum(
                monitor.states_dur[node_id]["switching_off"].values()
            )
        )

        features.append([
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            f7,
            f8,
            f9,
            f10,
            f11,
        ])

    return np.asarray(features, dtype=np.float32)