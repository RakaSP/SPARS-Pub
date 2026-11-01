import copy
import csv
import os
import json
import shutil
import time
from collections.abc import Mapping
from typing import Any

import torch as T

from spars.core.gym import components as G
from spars.core.logging import log_info, log_trace
from spars.core.progress import set_progress_context, set_progress_status
from spars.core.runner.checkpoint import capture_rng_state
from spars.core.simulator.simulator import Simulator
from spars.core.output import log_output

from .evaluation import run_evaluation
from .outputs import append_step_log, initialize_step_log, append_learning_log


ExperienceBatch = tuple[
    list[Any],
    list[T.Tensor],
    list[T.Tensor],
    list[T.Tensor],
    list[T.Tensor],
]


def _reward_to_tensor(
    reward: Any,
    device: str | T.device,
) -> T.Tensor:
    if isinstance(reward, T.Tensor):
        return reward.detach().to(device).float().squeeze()

    return T.tensor(
        float(reward),
        dtype=T.float32,
        device=device,
    )


def _prepare_observation(model: Any, observation: Any) -> Any:
    """Let the model standardise a training observation as it is collected.

    The rollout then stores the inputs the policy acted on (gymnasium's
    NormalizeObservation in CleanRL's PPO). Models without the hook get the
    observation unchanged.
    """
    normalise = getattr(model, "normalise_observation", None)
    return observation if normalise is None else normalise(observation)


def _collect_rollout_batch(
    env: Any,
    model: Any,
    observation: Any,
    episode_batch_size: int,
    device: str | T.device,
    step_log_path: str,
) -> tuple[Any, ExperienceBatch, list[T.Tensor]]:
    memory_observations: list[Any] = []
    memory_actions: list[T.Tensor] = []
    memory_logprobs: list[T.Tensor] = []
    memory_values: list[T.Tensor] = []
    memory_rewards: list[T.Tensor] = []
    batch_rewards: list[T.Tensor] = []

    model.start_rollout()
    observation = _prepare_observation(model, observation)

    for _ in range(episode_batch_size):
        rollout_start = time.time()

        with T.no_grad():
            logits, value = model(observation)
            action_time = env.simulator.current_time
            next_observation, ppo_info, done = env.step(
                observation,
                logits,
            )
        next_observation = _prepare_observation(model, next_observation)

        reward = ppo_info["reward"]
        logprob = ppo_info["logprob"]
        actions = ppo_info["actions"]

        reward_value = (
            reward.detach().cpu().item()
            if isinstance(reward, T.Tensor)
            else float(reward)
        )
        append_step_log(
            step_log_path,
            action_time,
            actions.detach().cpu().tolist(),
            reward_value,
        )

        log_trace(f"Step reward: {reward}")

        reward_tensor = _reward_to_tensor(
            reward,
            device,
        )

        memory_observations.append(observation)
        memory_actions.append(actions.detach())
        memory_logprobs.append(logprob.detach())
        memory_values.append(value.detach())
        memory_rewards.append(reward_tensor)
        batch_rewards.append(reward_tensor.detach().cpu())

        observation = next_observation

        rollout_finish = time.time()
        log_trace(f"Rollout duration: {rollout_finish - rollout_start}")

        if done:
            break

    saved_experiences: ExperienceBatch = (
        memory_observations,
        memory_actions,
        memory_logprobs,
        memory_values,
        memory_rewards,
    )
    return observation, saved_experiences, batch_rewards


def _next_value(
    env: Any,
    model: Any,
    observation: Any,
    device: str | T.device,
) -> T.Tensor:
    with T.no_grad():
        if env.simulator.is_running:
            _, next_value = model.peek(observation)
            return next_value.detach()

        return T.tensor(
            0.0,
            dtype=T.float32,
            device=device,
        )


def _save_checkpoint(
    model: Any,
    model_opt: T.optim.Optimizer,
    cfg: Mapping[str, Any],
    epoch: int,
    epoch_output_path: str,
    selection: dict[str, Any] | None = None,
) -> str:
    checkpoint = {
        "agent_class": (f"{model.__class__.__module__}:{model.__class__.__name__}"),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_opt.state_dict(),
        "rl_config": cfg.get("rl", {}),
        "completed_epochs": epoch + 1,
        "contract": model.learning_contract,
        "selection": selection
        or {
            "metric": "undiscounted_episode_return",
            "value": getattr(model, "epoch_return", None),
        },
        "rng_state": capture_rng_state(),
    }
    if getattr(model, "critic_scale_stats", None) is not None:
        # Running critic-target statistics of a normalized critic.
        checkpoint["critic_scale_stats"] = list(model.critic_scale_stats)
    checkpoint_path = os.path.join(
        epoch_output_path,
        "agent_checkpoint.pt",
    )
    temporary = checkpoint_path + ".tmp"
    T.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    log_trace(f"Saved agent checkpoint to: {checkpoint_path}")
    return checkpoint_path


def _best_windowed_epoch(scores: dict[int, float], window: int, fallback: int) -> int:
    """Epoch with the highest mean score over the ``window`` epochs centred on it."""
    half = window // 2
    smoothed = {
        epoch: sum(scores[epoch + offset] for offset in range(-half, half + 1)) / window
        for epoch in scores
        if all(epoch + offset in scores for offset in range(-half, half + 1))
    }
    if not smoothed:
        return fallback
    best = max(smoothed, key=smoothed.get)
    log_info(f"Best epoch by {window}-epoch mean validation return: {best} ({smoothed[best]})")
    return best


def _copy_best_epoch(
    output_path: str,
    best_epoch: int,
) -> str:
    best_source = os.path.join(
        output_path,
        f"epoch_{best_epoch}",
    )
    best_destination = os.path.join(
        output_path,
        f"best_epoch_{best_epoch}",
    )

    if os.path.exists(best_destination):
        shutil.rmtree(best_destination)

    shutil.copytree(
        best_source,
        best_destination,
    )
    return best_destination


def _start_training_epoch(
    cfg: dict[str, Any],
    epoch_output_path: str,
    env: Any,
    model: Any,
    rl_type: str,
    rl_dt: int | float | None,
) -> tuple[Any, str]:
    """Create a fresh simulator and return its first observation and log path."""
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
    env.start_episode()

    step_log_path = os.path.join(
        epoch_output_path,
        "step_log.csv",
    )
    initialize_step_log(step_log_path)

    return env.get_observation(), step_log_path


def _run_training_epoch(
    env: Any,
    model: Any,
    model_opt: T.optim.Optimizer,
    observation: Any,
    episode_batch_size: int,
    device: str | T.device,
    step_log_path: str,
) -> T.Tensor:
    """Collect rollouts and update the model until the simulator finishes."""
    epoch_rewards: list[T.Tensor] = []

    while env.simulator.is_running:
        (
            observation,
            saved_experiences,
            batch_rewards,
        ) = _collect_rollout_batch(
            env=env,
            model=model,
            observation=observation,
            episode_batch_size=episode_batch_size,
            device=device,
            step_log_path=step_log_path,
        )
        epoch_rewards.extend(batch_rewards)

        next_value = _next_value(
            env,
            model,
            observation,
            device,
        )
        diagnostics = G.learn(
            model,
            model_opt,
            saved_experiences,
            next_value=next_value,
        )

        if diagnostics:
            append_learning_log(
                os.path.join(os.path.dirname(step_log_path), "learning_log.csv"),
                diagnostics,
            )

    if not epoch_rewards:
        raise RuntimeError("Training epoch completed without producing any rewards")

    values = T.stack(epoch_rewards).float()
    summary = {
        "selection_metric": "undiscounted_episode_return",
        "episode_return": float(values.sum()),
        "mean_step_reward": float(values.mean()),
        "steps": values.numel(),
    }
    with open(
        os.path.join(os.path.dirname(step_log_path), "training_summary.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)
    return values.sum()


def _persist_training_epoch(
    cfg: Mapping[str, Any],
    env: Any,
    model: Any,
    model_opt: T.optim.Optimizer,
    epoch: int,
    epoch_output_path: str,
    selection: dict[str, Any] | None = None,
) -> None:
    """Persist the checkpoint and simulator outputs for a completed epoch."""
    _save_checkpoint(
        model=model,
        model_opt=model_opt,
        cfg=cfg,
        epoch=epoch,
        epoch_output_path=epoch_output_path,
        selection=selection,
    )
    log_output(
        env.simulator,
        epoch_output_path,
    )


def _run_validation(
    cfg: Mapping[str, Any],
    env: Any,
    model: Any,
    device: str | T.device,
    rl_type: str,
    rl_dt: int | float | None,
    epoch_output_path: str,
) -> dict[str, float]:
    """Evaluate the current weights with argmax on the validation workload.

    Uses its own simulator and gym env, exactly as a test run would, so the
    training env is untouched. Argmax draws no random numbers and each
    simulator owns a fixed-seed generator, so training is unaffected.
    """
    validation_cfg = copy.deepcopy(dict(cfg))
    validation_cfg["paths"]["workload"] = cfg["rl"]["validation"]["workload"]
    simulator = Simulator.from_config(
        validation_cfg,
        rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt, "learn": False},
    )
    validation_env = type(env)(simulator, training=False, device=device)
    output_path = os.path.join(epoch_output_path, "validation")
    episode_return = run_evaluation(output_path, validation_env, simulator, model)
    model.train()
    with open(os.path.join(output_path, "metrics.csv"), encoding="utf-8") as stream:
        metrics = next(csv.DictReader(stream))
    return {
        "validation_return": float(episode_return),
        "validation_mean_waiting_time": float(metrics["mean_waiting_time"]),
        "validation_total_energy_waste": float(metrics["total_energy_waste"]),
    }


def _append_validation_log(output_path: str, row: dict[str, Any]) -> None:
    path = os.path.join(output_path, "validation_log.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(
    cfg: dict[str, Any],
    output_path: str,
    env: Any,
    model: Any,
    model_opt: T.optim.Optimizer,
    device: str | T.device,
    rl_type: str,
    rl_dt: int | float | None,
    epochs: int,
    episode_batch_size: int,
) -> None:
    """Train the configured RL agent and persist every epoch.

    With ``rl.validation`` the best epoch is the one whose saved weights score
    the highest argmax return on the validation workload, and
    ``rl.validation.patience`` optionally stops training early. Without it the
    best epoch is the highest training return, and training never stops early.

    ``rl.validation.window`` (odd, default 1) scores each epoch by the mean
    validation return of the ``window`` epochs centred on it, once training
    ends: one validation episode is a noisy estimate, and neighbouring epochs
    differ by one epoch of updates. Only epochs with a full window qualify.
    """
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if episode_batch_size <= 0:
        raise ValueError("episode_batch_size must be greater than zero")

    model.train()

    validation = cfg.get("rl", {}).get("validation")
    patience = validation.get("patience") if validation else None
    window = validation.get("window", 1) if validation else 1
    scores: dict[int, float] = {}
    best_epoch_return = float("-inf")
    no_improvement_count = 0
    start_epoch = int(getattr(model, "completed_epochs", 0))
    best_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + epochs):
        set_progress_context("train", epoch=epoch + 1, epochs=start_epoch + epochs)
        log_info(f"========== EPOCH {epoch} ==========")

        epoch_output_path = os.path.join(
            output_path,
            f"epoch_{epoch}",
        )
        os.makedirs(
            epoch_output_path,
            exist_ok=True,
        )

        observation, step_log_path = _start_training_epoch(
            cfg=cfg,
            epoch_output_path=epoch_output_path,
            env=env,
            model=model,
            rl_type=rl_type,
            rl_dt=rl_dt,
        )
        epoch_return = _run_training_epoch(
            env=env,
            model=model,
            model_opt=model_opt,
            observation=observation,
            episode_batch_size=episode_batch_size,
            device=device,
            step_log_path=step_log_path,
        )
        if not T.isfinite(T.as_tensor(epoch_return)).all():
            raise FloatingPointError(f"Non-finite training reward in epoch {epoch}")
        epoch_return = float(epoch_return)
        model.epoch_return = epoch_return
        score = epoch_return
        selection = None
        if validation:
            result = _run_validation(
                cfg=cfg,
                env=env,
                model=model,
                device=device,
                rl_type=rl_type,
                rl_dt=rl_dt,
                epoch_output_path=epoch_output_path,
            )
            score = result["validation_return"]
            selection = {
                "metric": "validation_episode_return",
                "value": score,
                "workload": validation["workload"],
                "training_episode_return": epoch_return,
            }
            _append_validation_log(
                output_path,
                {"epoch": epoch, "training_return": epoch_return, **result},
            )
        set_progress_status("saving")
        _persist_training_epoch(
            cfg=cfg,
            env=env,
            model=model,
            model_opt=model_opt,
            epoch=epoch,
            epoch_output_path=epoch_output_path,
            selection=selection,
        )

        model.completed_epochs = epoch + 1
        scores[epoch] = score

        if score > best_epoch_return:
            best_epoch_return = score
            no_improvement_count = 0
            best_epoch = epoch
        else:
            no_improvement_count += 1

        log_info(f"EPISODE RETURN: {epoch_return}")
        if validation:
            log_info(f"VALIDATION RETURN: {score}")
            log_info(f"BEST VALIDATION RETURN: {best_epoch_return}")
        else:
            log_info(f"BEST EPISODE RETURN: {best_epoch_return}")
        log_info(f"NO IMPROVEMENT COUNT: {no_improvement_count}")
        if patience is not None and no_improvement_count >= patience:
            log_info(
                f"Early stop after epoch {epoch}: no better validation return "
                f"for {patience} epochs"
            )
            break

    set_progress_status("finalizing")
    if window > 1:
        best_epoch = _best_windowed_epoch(scores, window, best_epoch)
    best_destination = _copy_best_epoch(
        output_path,
        best_epoch,
    )
    log_info(f"Best epoch: {best_epoch}, saved to: {best_destination}")
