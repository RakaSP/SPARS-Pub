from SPARS.Logger import log_trace
import torch as T


SWITCH_OFF = 0
SWITCH_ON = 1


def action_translator(
    logits,
    simulator
):
    is_training = simulator.learn
    
    nodes = simulator.platform_control.machine.nodes
    node_ids = list(nodes.keys())

    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    if logits.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1, got {logits.shape[0]}"
        )

    if logits.shape[1] != len(node_ids):
        raise ValueError(
            f"Expected logits for {len(node_ids)} nodes, "
            f"got {logits.shape[1]}"
        )

    mask = T.zeros(
        (1, len(node_ids), 2),
        dtype=T.bool,
        device=logits.device,
    )

    for index, node_id in enumerate(node_ids):
        node = nodes[node_id]

        mask[0, index, SWITCH_OFF] = (
            node["state"] == "active"
            and node["job_id"] is None
        )

        mask[0, index, SWITCH_ON] = (
            node["state"] == "sleeping"
        )

    need_decision_idx = T.any(
        mask,
        dim=2,
    ).nonzero(as_tuple=False)[:, 1]

    if need_decision_idx.numel() == 0:
        actions = T.empty(
            (1, 0),
            dtype=T.long,
            device=logits.device,
        )

        logprobs = T.zeros(
            (1,),
            dtype=logits.dtype,
            device=logits.device,
        )

        return [], actions, logprobs

    decision_logits = logits[
        :,
        need_decision_idx,
        :,
    ]

    if is_training:
        distribution = T.distributions.Categorical(
            probs=decision_logits
        )

        actions = distribution.sample()
        logprobs = distribution.log_prob(actions)
    else:
        selected_logits, actions = T.max(
            decision_logits,
            dim=2,
        )

        logprobs = T.log(selected_logits)

    logprobs = logprobs.sum(dim=1)

    nodes_to_switch_off = []
    nodes_to_switch_on = []

    node_indices = (
        need_decision_idx
        .detach()
        .cpu()
        .tolist()
    )

    selected_actions = (
        actions
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    for node_index, action in zip(
        node_indices,
        selected_actions,
    ):
        if not mask[0, node_index, action]:
            continue

        node_id = node_ids[node_index]

        if action == SWITCH_OFF:
            nodes_to_switch_off.append(node_id)
        else:
            nodes_to_switch_on.append(node_id)

    log_trace(
        f"Final actions: "
        f"switch_off={nodes_to_switch_off}, "
        f"switch_on={nodes_to_switch_on}"
    )

    current_time = simulator.current_time
    events = []

    if nodes_to_switch_off:
        events.append({
            "time": current_time,
            "event": {
                "type": "switch_off",
                "nodes": nodes_to_switch_off,
            },
        })

    if nodes_to_switch_on:
        events.append({
            "time": current_time,
            "event": {
                "type": "switch_on",
                "nodes": nodes_to_switch_on,
            },
        })

    return events, actions, logprobs