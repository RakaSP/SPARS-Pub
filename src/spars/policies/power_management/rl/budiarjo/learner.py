from collections.abc import Sequence
from typing import Any

import torch as T

ACTION_KIND = "nodes"


def learn(
    model: Any,
    model_opt: T.optim.Optimizer,
    saved_experiences: tuple[
        Sequence[Any],
        Sequence[Any],
        Sequence[T.Tensor],
        Sequence[Any],
        Sequence[Any],
    ],
    next_value: T.Tensor,
    gamma: float = 0.99,
    grad_norm: float = 0.5,
) -> None:
    """Update the Budiarjo actor and critic from one rollout.

    Rollouts are collected without gradients, so recompute policy
    probabilities from each saved observation and its full node-action
    vector. ``-1`` marks nodes that were not decision candidates.
    """
    (
        observations,
        actions,
        _memory_logprobs,
        _memory_values,
        rewards,
    ) = saved_experiences

    if not rewards:
        return

    if not (len(observations) == len(actions) == len(rewards)):
        raise ValueError("Budiarjo rollout fields must have equal lengths")

    device = next(model.parameters()).device
    reward_values = T.stack(
        [
            reward.detach().to(device=device, dtype=T.float32).reshape(())
            if isinstance(reward, T.Tensor)
            else T.tensor(float(reward), dtype=T.float32, device=device)
            for reward in rewards
        ]
    )

    returns = T.empty_like(reward_values)
    running_return = (
        next_value.detach()
        .to(
            device=device,
            dtype=T.float32,
        )
        .reshape(())
    )
    for index in range(len(reward_values) - 1, -1, -1):
        running_return = reward_values[index] + gamma * running_return
        returns[index] = running_return

    actor_logprobs: list[T.Tensor] = []
    critic_values: list[T.Tensor] = []
    for observation, action in zip(observations, actions):
        state = T.as_tensor(
            observation,
            dtype=T.float32,
            device=device,
        )
        if state.ndim == 2:
            state = state.unsqueeze(0)

        probabilities, value = model.forward2(state)
        node_actions = T.as_tensor(
            action,
            dtype=T.long,
            device=probabilities.device,
        ).reshape(-1)
        node_probabilities = probabilities.reshape(-1, 2)
        if node_actions.numel() != node_probabilities.size(0):
            raise ValueError(
                "Budiarjo action vector must have one entry per node; "
                f"got {node_actions.numel()} actions for "
                f"{node_probabilities.size(0)} nodes"
            )

        selected_nodes = node_actions >= 0
        if selected_nodes.any():
            distribution = T.distributions.Categorical(
                probs=node_probabilities[selected_nodes]
            )
            selected_logprobs = distribution.log_prob(node_actions[selected_nodes])
            actor_logprobs.append(selected_logprobs.sum())
        else:
            actor_logprobs.append(probabilities.sum() * 0.0)

        critic_values.append(value.reshape(()))

    logprobs = T.stack(actor_logprobs)
    values = T.stack(critic_values)
    advantages = (returns - values).detach()

    actor_loss = -(logprobs * advantages).sum()
    critic_loss = T.nn.functional.mse_loss(values, returns)
    loss = actor_loss + critic_loss

    model_opt.zero_grad(set_to_none=True)
    loss.backward()
    T.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
    model_opt.step()
