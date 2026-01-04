# runner.py
from SPARS.Gym import config  # monkey patching the gym config
from SPARS.Gym import utils as G
from SPARS.Gym.gym import HPCGymEnv
from SPARS.Simulator.Simulator import Simulator, run_simulation
from SPARS.Logger import (
    set_log_level,
    log_info,
    log_trace,
)
import os
import time
import torch as T

from SPARS.Utils import log_output, _load_config, _choose_device, _build_agent, get_action, log_config_summary

DEFAULT_CFG_PATH = "simulator_config.yaml"

cfg = _load_config(DEFAULT_CFG_PATH)
set_log_level(cfg["logging"]["level"])
log_config_summary(cfg)

def main():
    output_path = cfg["paths"]["output"]

    rl_enabled = bool(cfg["rl"]["enabled"])
    rl_type = cfg["rl"]["type"] if rl_enabled else None
    rl_dt = cfg["rl"]["dt"] if rl_type == "discrete" else None
    device = _choose_device(cfg["rl"]["device"])

    # === RL parameters ===
    epochs = int(cfg["rl"]["epochs"])
    num_nodes = int(cfg["rl"]["num_nodes"])

    if rl_enabled and rl_type == "discrete" and rl_dt is None:
        raise RuntimeError("Discrete RL requires rl.dt in the config file.")

    if rl_enabled:
        assigned_name = cfg["rl"]["assign"]
        agents_dict = cfg["rl"]["agents"]
        agent_cfg = agents_dict[assigned_name]

        checkpoint = cfg["rl"]["checkpoint"]
        learn = cfg['rl']['learn']

        # Build simulator from config (no CLI/args)
        simulator = Simulator.from_config(
            cfg,
            rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
        )
        env = HPCGymEnv(simulator, device)

        model, model_opt = _build_agent(
            {"agent": agent_cfg, "device": cfg["rl"]["device"]}, device)

        if checkpoint is not None and learn:
            ckpt = T.load(checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model_opt.load_state_dict(ckpt["optimizer_state_dict"])
            model.to(device).train()
        
        elif checkpoint is not None and not learn:
            ckpt = T.load(checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model_opt.load_state_dict(ckpt["optimizer_state_dict"])
            model.eval()

        if learn:
            MAX_EPOCH_REWARD = -99999999999
            NO_IMPROVEMENT_COUNT = 0

            for _ in range(epochs):
                log_info(f'========== EPOCH {_} ==========')
                simulator = Simulator.from_config(
                    cfg,
                    rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
                )
                env.reset(simulator)
                env.simulator.start_simulator()
                observation = env.get_observation()
                batch_timesteps_size = 32

                while env.simulator.is_running:
                    
                    memory_features = []
                    memory_logprob = []
                    memory_actions = []
                    memory_rewards = []

                    # roll out
                    for i in range(batch_timesteps_size):
                        rollout_start = time.time()
                        features_ = observation
                        features_ = features_.to(device)

                        action, logprob = get_action(model, features_)

                        next_observation, reward, done = env.step(action)

                        log_trace(f"Step reward: {reward}")

                        memory_actions.append(action.detach())
                        memory_features.append(features_.detach())
                        memory_rewards.append(reward.detach() if isinstance(reward, T.Tensor)
                                              else T.tensor(float(reward)))

                        saved_experiences = (
                            memory_actions, memory_features, memory_rewards
                        )

                        observation = next_observation
                        if done == True:
                            break
                        rollout_finish = time.time()
                        log_info(
                            f'Rollout duration: {rollout_finish - rollout_start}')
                    G.learn(model, model_opt,
                            saved_experiences)

                avg_epoch_reward = sum(memory_rewards) / len(memory_rewards)
                avg_action = sum(memory_actions) / len(memory_actions)

                if avg_epoch_reward > MAX_EPOCH_REWARD:
                    MAX_EPOCH_REWARD = avg_epoch_reward
                    NO_IMPROVEMENT_COUNT = 0
                    # --- Save agent checkpoint ---
                    os.makedirs(output_path, exist_ok=True)
                    ckpt = {
                        "agent_class": f"{model.__class__.__module__}:{model.__class__.__name__}",
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model_opt.state_dict(),
                        "rl_config": cfg.get("rl", {}),
                        "epochs_trained": epochs,
                    }
                    ckpt_path = os.path.join(
                        output_path, "agent_checkpoint.pt")
                    T.save(ckpt, ckpt_path)
                    log_trace(f"Saved agent checkpoint to: {ckpt_path}")
                    log_output(env.simulator, output_path)
                else:
                    NO_IMPROVEMENT_COUNT += 1

                log_info(f"AVG ACTION: {avg_action}")
                log_info(f"AVG REWARD: {avg_epoch_reward}")
                log_info(f"MAX REWARD: {MAX_EPOCH_REWARD}")
                log_info(f"NO IMPROVEMENT COUNT: {NO_IMPROVEMENT_COUNT}")

                if NO_IMPROVEMENT_COUNT > 3:
                    # early stopping
                    break
        else:
            env.reset(simulator)
            env.simulator.start_simulator()
            observation = env.get_observation()

            while env.simulator.is_running:
                features_ = observation
                features_ = features_.to(device)
                action, logprob = get_action(model, features_)
                next_observation, reward, done = env.step(action)
                observation = next_observation

            log_output(env.simulator, output_path)

    else:
        simulator = Simulator.from_config(cfg)
        run_simulation(simulator, output_path)


if __name__ == "__main__":
    main()