import copy
import csv
import importlib
import inspect
import os
import shutil
import sys
import time
from pathlib import Path

import torch as T
import yaml

from SPARS.Gym import config
from SPARS.Gym import utils as G
from SPARS.Logger import (
    log_info,
    log_trace,
    set_log_level,
)
from SPARS.Simulator.Simulator import (
    Simulator,
    run_simulation,
)
from SPARS.Utils import (
    _build_agent,
    _choose_device,
    _load_config,
    log_config_summary,
    log_output,
)


DEFAULT_CFG_PATH = "simulator_config.yaml"


def apply_gym_config(cfg):
    gym_cfg = cfg.get("gym")

    if gym_cfg is None:
        return

    if not isinstance(gym_cfg, dict):
        raise TypeError("gym must be a dictionary")

    config.apply_gym_config(gym_cfg)

    log_info(
        "Applied gym configuration:\n"
        f"  feature_extractor = {config.CFG['feature_extractor']}\n"
        f"  translator        = {config.CFG['translator']}\n"
        f"  reward            = {config.CFG['reward']}\n"
        f"  learner           = {config.CFG['learner']}"
    )

def load_hpc_gym_env():
    module_name = "SPARS.Gym.gym"

    if module_name in sys.modules:
        gym_module = importlib.reload(sys.modules[module_name])
    else:
        gym_module = importlib.import_module(module_name)

    return gym_module.HPCGymEnv


def dotted_name(obj):
    module = getattr(
        obj,
        "__module__",
        type(obj).__module__,
    )

    qualname = getattr(
        obj,
        "__qualname__",
        type(obj).__qualname__,
    )

    return f"{module}:{qualname}"


def save_metadata(cfg, output_path):
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "simulator_config_used.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False,
        )

    if cfg["rl"]["enabled"]:
        gym_metadata = {
            "config": copy.deepcopy(config.CFG),
            "resolved": {
                "feature_extractor": dotted_name(config.feature_extractor),
                "translator": dotted_name(config.translator),
                "learner": dotted_name(config.learner),
                "reward_factory": dotted_name(config.G.Reward),
                "reward_instance": dotted_name(config.reward_instance),
            },
        }

        with open(
            output_dir / "gym_config_used.yaml",
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                gym_metadata,
                file,
                sort_keys=False,
            )


def run_with_config(cfg):
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a configuration dictionary")

    set_log_level(cfg["logging"]["level"])

    apply_gym_config(cfg)

    # Import after applying the YAML reward configuration.
    HPCGymEnv = load_hpc_gym_env()

    log_config_summary(cfg)

    output_path = os.path.abspath(
        cfg["paths"]["output"]
    )

    save_metadata(cfg, output_path)

    rl_enabled = bool(cfg["rl"]["enabled"])
    rl_type = cfg["rl"]["type"]
    rl_dt = cfg["rl"]["dt"]
    device = _choose_device(cfg["rl"]["device"])

    epochs = int(cfg["rl"]["epochs"])
    episode_batch_size = int(cfg["rl"]["episode_batch_size"])

    if rl_enabled and rl_dt is None:
        raise RuntimeError(
            "Discrete RL requires rl.dt in the config file."
        )

    if rl_enabled:
        assigned_name = cfg["rl"]["assign"]
        agents_dict = cfg["rl"]["agents"]
        agent_cfg = agents_dict[assigned_name]

        checkpoint = cfg["rl"]["checkpoint"]
        learn = cfg["rl"]["learn"]

        simulator = Simulator.from_config(
            cfg,
            rl_kwargs={
                "rl_type": rl_type,
                "rl_dt": rl_dt,
            },
        )

        env = HPCGymEnv(
            simulator,
            training=learn,
            device=device,
        )

        model, model_opt = _build_agent(
            {
                "agent": agent_cfg,
                "device": cfg["rl"]["device"],
            },
            device,
        )

        if checkpoint is not None and learn:
            ckpt = T.load(
                checkpoint,
                map_location=device,
            )

            model.load_state_dict(
                ckpt["model_state_dict"]
            )

            model_opt.load_state_dict(
                ckpt["optimizer_state_dict"]
            )

            model.to(device).train()

        elif checkpoint is not None and not learn:
            ckpt = T.load(
                checkpoint,
                map_location=device,
            )

            model.load_state_dict(
                ckpt["model_state_dict"]
            )

            model_opt.load_state_dict(
                ckpt["optimizer_state_dict"]
            )

            model.to(device)
            model.eval()

        if learn:
            model.train()

            MAX_EPOCH_REWARD = -99999999999
            NO_IMPROVEMENT_COUNT = 0
            BEST_EPOCH = 0

            for epoch in range(epochs):
                log_info(
                    f"========== EPOCH {epoch} =========="
                )

                epoch_output_path = os.path.join(
                    output_path,
                    f"epoch_{epoch}",
                )

                os.makedirs(
                    epoch_output_path,
                    exist_ok=True,
                )

                simulator = Simulator.from_config(
                    cfg,
                    rl_kwargs={
                        "rl_type": rl_type,
                        "rl_dt": rl_dt,
                    },
                )

                simulator.monitor.configure_state_hist_spill(
                    output_folder=epoch_output_path,
                    flush_every=1000,
                )

                model.reset_episode()

                env.reset(simulator)
                env.simulator.start_simulator()

                obs = env.get_observation()

                epoch_rewards = []
                epoch_actions = []

                step_log_path = os.path.join(
                    epoch_output_path,
                    "step_log.csv",
                )

                with open(
                    step_log_path,
                    "w",
                    newline="",
                ) as file:
                    csv.writer(file).writerow([
                        "simulation_time",
                        "action",
                        "reward",
                    ])

                while env.simulator.is_running:
                    memory_observations = []
                    memory_actions = []
                    memory_logprobs = []
                    memory_values = []
                    memory_rewards = []

                    model.start_rollout()

                    for _ in range(episode_batch_size):
                        rollout_start = time.time()

                        with T.no_grad():
                            logits, value = model(obs)

                            action_time = env.simulator.current_time

                            next_obs, ppo_info, done = env.step(
                                obs,
                                logits,
                            )
                        
                        reward = ppo_info["reward"]
                        logprob = ppo_info["logprob"]
                        actions = ppo_info["actions"]

                        reward_value = (
                            reward.detach().cpu().item()
                            if isinstance(reward, T.Tensor)
                            else float(reward)
                        )

                        with open(
                            step_log_path,
                            "a",
                            newline="",
                        ) as file:
                            csv.writer(file).writerow([
                                action_time,
                                actions.detach().cpu().tolist(),
                                reward_value,
                            ])

                        log_trace(
                            f"Step reward: {reward}"
                        )

                        reward_tensor = (
                            reward.detach().to(device).float().squeeze()
                            if isinstance(reward, T.Tensor)
                            else T.tensor(
                                float(reward),
                                dtype=T.float32,
                                device=device,
                            )
                        )

                        memory_observations.append(obs)
                        memory_actions.append(actions.detach())
                        memory_logprobs.append(logprob.detach())
                        memory_values.append(value.detach())
                        memory_rewards.append(reward_tensor)

                        epoch_actions.append(
                            actions.detach().cpu()
                        )

                        epoch_rewards.append(
                            reward_tensor.detach().cpu()
                        )

                        obs = next_obs

                        rollout_finish = time.time()

                        log_trace(
                            "Rollout duration: "
                            f"{rollout_finish - rollout_start}"
                        )

                        if done:
                            break

                    saved_experiences = (
                        memory_observations,
                        memory_actions,
                        memory_logprobs,
                        memory_values,
                        memory_rewards,
                    )

                    with T.no_grad():
                        if env.simulator.is_running:
                            _, next_value = model.peek(obs)
                            next_value = next_value.detach()
                        else:
                            next_value = T.tensor(
                                0.0,
                                dtype=T.float32,
                                device=device,
                            )

                    G.learn(
                        model,
                        model_opt,
                        saved_experiences,
                        next_value=next_value,
                    )

                avg_epoch_reward = (
                    T.stack(epoch_rewards)
                    .float()
                    .mean()
                )
                
                ckpt = {
                    "agent_class": (
                        f"{model.__class__.__module__}:"
                        f"{model.__class__.__name__}"
                    ),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model_opt.state_dict(),
                    "rl_config": cfg.get("rl", {}),
                    "epochs_trained": epoch,
                }

                ckpt_path = os.path.join(
                    epoch_output_path,
                    "agent_checkpoint.pt",
                )

                T.save(
                    ckpt,
                    ckpt_path,
                )

                log_trace(
                    "Saved agent checkpoint to: "
                    f"{ckpt_path}"
                )

                log_output(
                    env.simulator,
                    epoch_output_path,
                )

                if avg_epoch_reward > MAX_EPOCH_REWARD:
                    MAX_EPOCH_REWARD = avg_epoch_reward
                    NO_IMPROVEMENT_COUNT = 0
                    BEST_EPOCH = epoch
                else:
                    NO_IMPROVEMENT_COUNT += 1

          

                log_info(
                    f"AVG REWARD: {avg_epoch_reward}"
                )

                log_info(
                    f"MAX REWARD: {MAX_EPOCH_REWARD}"
                )

                log_info(
                    "NO IMPROVEMENT COUNT: "
                    f"{NO_IMPROVEMENT_COUNT}"
                )

            best_src = os.path.join(
                output_path,
                f"epoch_{BEST_EPOCH}",
            )

            best_dst = os.path.join(
                output_path,
                f"best_epoch_{BEST_EPOCH}",
            )

            if os.path.exists(best_dst):
                shutil.rmtree(best_dst)

            shutil.copytree(
                best_src,
                best_dst,
            )

            log_info(
                f"Best epoch: {BEST_EPOCH}, "
                f"saved to: {best_dst}"
            )

        else:
            os.makedirs(
                output_path,
                exist_ok=True,
            )

            simulator.monitor.configure_state_hist_spill(
                output_folder=output_path,
                flush_every=1000,
            )

            env.reset(simulator)
            env.simulator.start_simulator()

            obs = env.get_observation()

            model.eval()
            model.reset_episode()

            step_log_path = os.path.join(
                output_path,
                "step_log.csv",
            )

            with open(
                step_log_path,
                "w",
                newline="",
            ) as file:
                csv.writer(file).writerow([
                    "simulation_time",
                    "action",
                    "reward",
                ])

            while env.simulator.is_running:
                with T.no_grad():
                    logits, value = model(obs)

                action_time = env.simulator.current_time

                next_obs, ppo_info, done = env.step(
                    obs,
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

                with open(
                    step_log_path,
                    "a",
                    newline="",
                ) as file:
                    csv.writer(file).writerow([
                        action_time,
                        action_value,
                        reward_value,
                    ])

                log_trace(
                    "actions:",
                    actions,
                )

                obs = next_obs

            log_output(
                env.simulator,
                output_path,
            )

    else:
        simulator = Simulator.from_config(cfg)

        run_simulation(
            simulator,
            output_path,
        )


def main():
    cfg = _load_config(
        DEFAULT_CFG_PATH
    )

    run_with_config(cfg)


if __name__ == "__main__":
    main()