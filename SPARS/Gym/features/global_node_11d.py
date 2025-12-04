import numpy as np

FEATURE_DIM = 11


def feature_extraction(simulator) -> np.ndarray:
    """
    Returns a 1D global feature vector of shape [11].
    """

    # === GLOBAL (simulator-level) FEATURES ===
    tq = simulator.jobs_manager.waiting_queue
    tnow = simulator.current_time
    t0 = simulator.start_time
    dt = max(tnow - t0, 1e-8)

    job_num = float(len(tq))
    arrival_rate = float(len(simulator.Monitor.jobs_arrival_log)) / dt

    total_req_wt_q = sum(job.get("reqtime", 0.0)
                      for job in tq) 
    total_req_nodes = sum(job.get("res", 0.0)
        for job in tq)

    sim_feats = np.array([
        job_num,
        arrival_rate,
        float(total_req_wt_q),
        float(total_req_nodes),
    ], dtype=np.float32)  # [6]

    # === AGGREGATED NODE FEATURES ===
    state = list(simulator.PlatformControl.get_state())
    idle_nodes = [n["id"] for n in state if n.get(
        "state") == "active" and n.get("job_id") is None]
    sleeping_nodes = [n["id"] for n in state if n.get("state") == "sleeping"]

    transitions_info = getattr(getattr(
        simulator.PlatformControl, "machines", object()), "machines_transition", [])
    sleeping_set, idle_set = set(sleeping_nodes), set(idle_nodes)
    switch_on_times, switch_off_times = [], []
    for node_info in transitions_info:
        nid = node_info.get("node_id")
        for tr in node_info.get("transitions", []):
            frm = tr.get("from")
            to = tr.get("to")
            tt = float(tr.get("transition_time", 0.0))
            if frm == "switching_on" and to == "active" and nid in sleeping_set:
                switch_on_times.append(tt)
            if frm == "switching_off" and to == "sleeping" and nid in idle_set:
                switch_off_times.append(tt)

    node_feats = np.array([
        float(len(idle_nodes)),
        float(len(sleeping_nodes)),
    ], dtype=np.float32)  # [5]

    features = np.concatenate(
        [sim_feats, node_feats], axis=0).astype(np.float32)  # [11]

    features = features.reshape(1, 6)
    return features
