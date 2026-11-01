"""Build the configured RL agent and optimizer on the requested device."""

from collections.abc import Mapping
from typing import Any

from spars.core.registry import load_object


def resolve_device(device: str) -> str:
    """Map the config value "gpu" / "cpu" to a torch device string."""
    import torch as T
    name = str(device).lower()

    if name == "cpu":
        return "cpu"

    if name != "gpu":
        raise ValueError(
            f'device must be "gpu" or "cpu", got {device!r}'
        )

    # torch reports NVIDIA (CUDA) and AMD (ROCm) builds as "cuda".
    if T.cuda.is_available():
        return "cuda"

    xpu = getattr(T, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"

    mps = getattr(T.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    raise RuntimeError(
        'device is "gpu" but no GPU backend is available; '
        'set device to "cpu"'
    )


def _build_agent(rl_cfg: Mapping[str, Any]) -> tuple[Any, Any]:
    """Construct the explicitly configured neural agent and optimizer."""
    import torch as T
    spec = rl_cfg["agent"]
    parameters = dict(spec.get("params", {}))
    device = resolve_device(parameters.get("device", rl_cfg["device"]))
    parameters["device"] = device
    model = load_object(spec["class"])(**parameters)
    if not isinstance(model, T.nn.Module):
        raise TypeError("Configured agent must be a torch.nn.Module")
    model.to(device)
    optimizer = spec["optimizer"]
    model_optimizer = load_object(optimizer["class"])(model.parameters(), **optimizer.get("params", {}))
    return model, model_optimizer
