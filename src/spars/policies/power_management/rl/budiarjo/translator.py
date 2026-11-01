from typing import TYPE_CHECKING, Any

import torch as T

from spars.core.logging import log_trace

ACTION_KIND = "nodes"
CONTROLLER_METHODS = ("get_node_action_mask", "apply_rl_action")

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator


SWITCH_OFF = 0
SWITCH_ON = 1
_ACTION_COUNT = 2


def _normalize_logits(
    logits: T.Tensor,
    node_count: int,
) -> T.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    if logits.ndim != 3:
        raise ValueError("Budiarjo logits must have shape (nodes, 2) or (1, nodes, 2)")

    if logits.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {logits.shape[0]}")

    if logits.shape[1] != node_count:
        raise ValueError(
            f"Expected logits for {node_count} nodes, got {logits.shape[1]}"
        )

    if logits.shape[2] != _ACTION_COUNT:
        raise ValueError(
            f"Expected {_ACTION_COUNT} actions per node, got {logits.shape[2]}"
        )

    return logits


def _build_action_mask(
    nodes: dict[int, dict[str, Any]],
    node_ids: list[int],
    *,
    allowed_actions: dict[str, set[int]] | None,
    device: T.device,
) -> T.Tensor:
    action_mask = T.zeros(
        (1, len(node_ids), _ACTION_COUNT),
        dtype=T.bool,
        device=device,
    )

    for node_index, node_id in enumerate(node_ids):
        node = nodes[node_id]
        state = node["state"]

        if state == "active" and node["job_id"] is None:
            can_switch_off = (
                allowed_actions is None or node_id in allowed_actions["switch_off"]
            )
            action_mask[0, node_index, SWITCH_OFF] = can_switch_off
            # For an active node, SWITCH_ON means keep it active.
            action_mask[0, node_index, SWITCH_ON] = True
        elif state == "sleeping":
            # For a sleeping node, SWITCH_OFF means leave it asleep.
            action_mask[0, node_index, SWITCH_OFF] = True
            action_mask[0, node_index, SWITCH_ON] = (
                allowed_actions is None or node_id in allowed_actions["switch_on"]
            )

    return action_mask


def _select_actions(
    decision_logits: T.Tensor,
    *,
    is_training: bool,
) -> tuple[T.Tensor, T.Tensor]:
    if is_training:
        distribution = T.distributions.Categorical(probs=decision_logits)
        actions = distribution.sample()
        logprobs = distribution.log_prob(actions)
    else:
        selected_logits, actions = T.max(
            decision_logits,
            dim=2,
        )
        logprobs = T.log(selected_logits)

    return actions, logprobs.sum(dim=1)


def action_translator(
    logits: T.Tensor,
    simulator: "Simulator",
) -> tuple[list[dict[str, Any]], T.Tensor, T.Tensor]:
    nodes = simulator.platform_control.machine.nodes
    node_ids = list(nodes)
    logits = _normalize_logits(logits, len(node_ids))
    power_manager = simulator.planner.power_manager
    allowed_actions = power_manager.get_node_action_mask(simulator.current_time)

    action_mask = _build_action_mask(
        nodes,
        node_ids,
        allowed_actions=allowed_actions,
        device=logits.device,
    )
    # Only give the policy nodes with a genuine choice. Nodes with one legal
    # action are controlled by the safety gate, not by the learned policy.
    decision_node_indices = T.all(
        action_mask,
        dim=2,
    ).nonzero(as_tuple=False)[:, 1]

    if decision_node_indices.numel() == 0:
        # Preserve the node-indexed action shape even when the safety mask
        # leaves no decisions for the policy. The learner uses -1 to ignore
        # nodes that were not policy choices.
        actions = T.full(
            (1, len(node_ids)),
            -1,
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
        decision_node_indices,
        :,
    ]
    decision_action_mask = action_mask[
        :,
        decision_node_indices,
        :,
    ]
    decision_logits = decision_logits.masked_fill(
        ~decision_action_mask,
        0.0,
    )
    decision_logits = decision_logits / decision_logits.sum(
        dim=2,
        keepdim=True,
    ).clamp_min(1e-12)
    actions, logprobs = _select_actions(
        decision_logits,
        is_training=simulator.learn,
    )

    nodes_to_switch_off: list[int] = []
    nodes_to_switch_on: list[int] = []

    selected_node_indices = decision_node_indices.detach().cpu().tolist()
    selected_actions = actions.squeeze(0).detach().cpu().tolist()

    for node_index, action in zip(
        selected_node_indices,
        selected_actions,
    ):
        if not action_mask[0, node_index, action]:
            continue

        node_id = node_ids[node_index]
        node_state = nodes[node_id]["state"]
        if node_state == "active" and action == SWITCH_OFF:
            nodes_to_switch_off.append(node_id)
        elif node_state == "sleeping" and action == SWITCH_ON:
            nodes_to_switch_on.append(node_id)

    log_trace(
        "Final actions: "
        f"switch_off={nodes_to_switch_off}, "
        f"switch_on={nodes_to_switch_on}"
    )

    buckets = power_manager.apply_rl_action(
        {"switch_off": nodes_to_switch_off, "switch_on": nodes_to_switch_on},
        simulator.current_time,
    )
    events = [
        {"time": bucket["timestamp"], "event": event}
        for bucket in buckets
        for event in bucket["events"]
    ]
    full_actions = T.full(
        (1, len(node_ids)),
        -1,
        dtype=T.long,
        device=actions.device,
    )
    full_actions[0, decision_node_indices] = actions.squeeze(0)

    return events, full_actions, logprobs
