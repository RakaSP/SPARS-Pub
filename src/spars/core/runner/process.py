import importlib
import inspect
import sys
from collections.abc import Callable
from typing import Any

from spars.core.gym import config, components as G
from spars.core.simulator.simulator import Simulator
from spars.core.runner.agent import _build_agent
from spars.core.runner.checkpoint import load_checkpoint
from spars.core.gym.components import validate_component_contracts


def apply_gym_config(cfg: dict[str, Any]) -> None:
    gym_config = cfg.get("gym")

    if gym_config is None:
        return

    if not isinstance(gym_config, dict):
        raise TypeError("gym must be a dictionary")

    config.apply_gym_config(gym_config)


def load_hpc_gym_env() -> type:
    """Load or reload the configured HPC gym environment class."""
    module_name = "spars.core.gym.env"

    if module_name in sys.modules:
        gym_module = importlib.reload(sys.modules[module_name])
    else:
        gym_module = importlib.import_module(module_name)

    return gym_module.HPCGymEnv


def prepare_rl_process(
    cfg: dict[str, Any],
    hpc_gym_env: Callable[..., Any],
    device: str,
    rl_type: str,
    rl_dt: Any,
) -> tuple[Simulator, Any, Any, Any, bool]:
    rl_config = cfg["rl"]
    assigned_name = rl_config["assign"]
    agents = rl_config["agents"]

    if assigned_name not in agents:
        raise KeyError(f"RL agent not found: {assigned_name}")

    agent_config = agents[assigned_name]
    checkpoint_path = rl_config["checkpoint"]
    learn = bool(rl_config["learn"])

    simulator = Simulator.from_config(
        cfg,
        rl_kwargs={
            "rl_type": rl_type,
            "rl_dt": rl_dt,
        },
    )

    environment = hpc_gym_env(
        simulator,
        training=learn,
        device=device,
    )

    model, model_optimizer = _build_agent(
        {
            "agent": agent_config,
            "device": rl_config["device"],
        }
    )

    validate_component_contracts(model, simulator)
    reward = config.reward_instance
    learner_module = importlib.import_module(G.learn.__module__)
    model.learning_contract = {
        "version": 2,
        "observation": model.observation_kind,
        "action": model.action_kind,
        "observation_schema": getattr(model, "observation_schema", None),
        "feature_component": G.feature_extraction.__module__,
        "learner_component": G.learn.__module__,
        "learner_schema": getattr(learner_module, "LEARNER_SCHEMA", "original"),
        "learner_defaults": {
            key: parameter.default
            for key, parameter in inspect.signature(G.learn).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        },
        # Settings bound from gym.learner.params (already in the defaults).
        "learner_params": dict(getattr(G.learn, "keywords", None) or {}),
        "reward_schema": reward.schema,
        "reward_weights": [reward.weight1, reward.weight2],
        "reward_scales": reward.scales(simulator.platform_control.machine),
    }
    model.completed_epochs = 0
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, model_optimizer, device)

        model.to(device)

        if learn:
            model.train()
        else:
            model.eval()

    return (
        simulator,
        environment,
        model,
        model_optimizer,
        learn,
    )
