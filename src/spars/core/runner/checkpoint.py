"""Checkpoint validation and reproducible episode-boundary recovery."""

import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state():
    name, values, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, values.tolist(), position, gaussian, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state(
        (name, np.asarray(values, dtype=np.uint32), position, gaussian, cached)
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


# Component modules before the core/policies split. Contracts saved by older
# runs name these; they describe the same components as the current modules.
_RL = "spars.policies.power_management.rl"
LEGACY_COMPONENT_MODULES = {
    "spars.gym.features.Budiarjo": f"{_RL}.budiarjo.features",
    "spars.gym.learners.Budiarjo": f"{_RL}.budiarjo.learner",
}


def _current_contract(contract):
    if not isinstance(contract, dict):
        return contract
    contract = dict(contract)
    # Keys of options that no longer exist; a run that left them unset is still compatible.
    for removed in ("trigger", "reward_reference", "reward_branch"):
        if contract.get(removed) is None:
            contract.pop(removed, None)
    for key in ("feature_component", "learner_component"):
        if key in contract:
            contract[key] = LEGACY_COMPONENT_MODULES.get(contract[key], contract[key])
    return contract


def load_checkpoint(path, model, optimizer, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing {key}")
    expected = model.state_dict()
    actual = checkpoint["model_state_dict"]
    if expected.keys() != actual.keys() or any(
        expected[key].shape != actual[key].shape for key in expected
    ):
        raise ValueError(
            "Checkpoint architecture/platform does not match the configured agent; start fresh training for the updated features"
        )
    if _current_contract(checkpoint.get("contract")) != model.learning_contract:
        raise ValueError(
            "Checkpoint learning contract differs (features, reward or learner). "
            "Start fresh training for the updated formulation."
        )
    model.load_state_dict(actual)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.completed_epochs = int(
        checkpoint.get(
            "completed_epochs", int(checkpoint.get("epochs_trained", -1)) + 1
        )
    )
    if model.completed_epochs < 0:
        raise ValueError("Checkpoint completed epoch count must be nonnegative")
    if "critic_scale_stats" in checkpoint:
        model.critic_scale_stats = list(checkpoint["critic_scale_stats"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
