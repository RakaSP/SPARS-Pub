import os
from typing import TYPE_CHECKING, Any

import torch as T

from spars.core.logging import log_trace
from spars.core.progress import set_progress_status
from spars.core.output import log_output

from .outputs import append_step_log, initialize_step_log

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator
    from spars.core.gym.env import HPCGymEnv


def run_evaluation(
    output_path: str,
    env: "HPCGymEnv",
    simulator: "Simulator",
    model: Any,
) -> float:
    """Run one argmax episode, write its outputs and return its episode return."""
    os.makedirs(
        output_path,
        exist_ok=True,
    )

    simulator.monitor.configure_state_hist_spill(
        output_folder=output_path,
        flush_every=1000,
    )

    env.reset(simulator)
    env.start_episode()

    observation = env.get_observation()

    model.eval()
    model.reset_episode()

    step_log_path = os.path.join(
        output_path,
        "step_log.csv",
    )
    initialize_step_log(step_log_path)

    episode_return = 0.0
    while env.simulator.is_running:
        with T.no_grad():
            logits, _ = model(observation)

        action_time = env.simulator.current_time
        next_observation, ppo_info, _ = env.step(
            observation,
            logits,
        )

        reward = ppo_info["reward"]
        actions = ppo_info["actions"]
        action_value = actions.cpu().tolist()
        reward_value = (
            reward.detach().cpu().item()
            if isinstance(reward, T.Tensor)
            else float(reward)
        )
        episode_return += reward_value

        append_step_log(
            step_log_path,
            action_time,
            action_value,
            reward_value,
        )

        log_trace(
            "actions: %s",
            actions,
        )

        observation = next_observation

    set_progress_status("saving")
    log_output(
        env.simulator,
        output_path,
    )
    return episode_return
