"""Execute one effective configuration; RL dependencies load only for RL runs."""

import os
import random
from typing import Any

from spars.core.registry import resolve_run_components
from spars.core.logging import set_log_level
from spars.core.progress import set_progress_context, set_progress_status
from spars.core.simulator.simulator import Simulator, run_simulation
from spars.core.output import log_config_summary
from spars.core.runner.agent import resolve_device
from .metadata import save_metadata


def run_with_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise TypeError("Configuration must be a dictionary")
    scheduler, power = resolve_run_components(cfg["run"])
    cfg["run"].update(scheduler=scheduler, power_management=power)
    set_log_level(cfg["logging"]["level"])
    log_config_summary(cfg)
    output_path = os.path.abspath(cfg["paths"]["output"])
    rl = cfg["rl"]
    phase = (
        "train" if rl["enabled"] and rl["learn"] else "test" if rl["enabled"] else "run"
    )
    set_progress_context(phase)
    set_progress_status("loading")
    if not rl["enabled"]:
        save_metadata(cfg, output_path)
        simulator = Simulator.from_config(cfg)
        set_progress_status("running")
        run_simulation(simulator, output_path)
        return

    import numpy as np
    try:
        import torch as T
    except ImportError:
        raise RuntimeError(
            "This experiment uses reinforcement learning (rl.enabled), which needs PyTorch. "
            "Install it with: python install.py --torch"
        ) from None
    from .evaluation import run_evaluation
    from .process import apply_gym_config, load_hpc_gym_env, prepare_rl_process
    from .training import run_training

    # Each simulation owns one Simulation core, so torch uses one thread.
    T.set_num_threads(1)
    try:
        T.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = resolve_device(rl["device"])
    for key in ("epochs", "episode_batch_size"):
        if not isinstance(rl.get(key), int) or rl[key] <= 0:
            raise ValueError(f"rl.{key} must be a positive integer")
    if rl["type"] == "discrete" and (rl["dt"] is None or rl["dt"] <= 0):
        raise ValueError("Discrete RL requires a positive decision interval")
    seed = rl.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)
    apply_gym_config(cfg)
    simulator, env, model, optimizer, learn = prepare_rl_process(
        cfg,
        load_hpc_gym_env(),
        device,
        rl["type"],
        rl["dt"],
    )
    cfg["rl"]["starting_completed_epochs"] = model.completed_epochs
    cfg["rl"]["resolved_device"] = str(next(model.parameters()).device)
    save_metadata(cfg, output_path, model=model)
    set_progress_status("running")
    if learn:
        run_training(
            cfg,
            output_path,
            env,
            model,
            optimizer,
            device,
            rl["type"],
            rl["dt"],
            rl["epochs"],
            rl["episode_batch_size"],
        )
    else:
        run_evaluation(output_path, env, simulator, model)
