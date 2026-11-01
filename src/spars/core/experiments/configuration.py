import copy
from typing import Any

from spars.core.registry import resolve_run_components
from spars.core.logging import normalize_log_level


Config = dict[str, Any]


def deep_merge(base: Config, override: Config) -> Config:
    """Recursively merge ``override`` into a deep copy of ``base``."""
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def merge_overrides(base: Config, overrides: Config) -> Config:
    """A definition's settings: ``base`` with ``overrides`` merged in.

    The scheduler and power-manager configs are replaced, not merged, so a
    default config never adds keys to the one a definition chose.
    """
    cfg = deep_merge(base, overrides)
    run_overrides = overrides.get("run", {})
    for component_name in ("scheduler", "power_management"):
        component_override = run_overrides.get(component_name)
        if not isinstance(component_override, dict):
            continue
        if "config" in component_override:
            cfg["run"][component_name]["config"] = copy.deepcopy(
                component_override["config"]
            )
        elif "name" in component_override:
            # A different policy was named with no config of its own: it must not
            # inherit the base policy's settings (e.g. NoPowerManager inheriting
            # TimeoutPowerManager's "timeout").
            cfg["run"][component_name]["config"] = {}
    return cfg


def set_nested(cfg: Config, path: str, value: Any) -> None:
    """Set a dot-delimited configuration path, creating missing dictionaries."""
    if not path or not path.strip("."):
        raise ValueError("Configuration path must not be empty")

    keys = path.split(".")
    if any(not key for key in keys):
        raise ValueError(f"Invalid configuration path: {path!r}")

    current = cfg

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}

        if not isinstance(current[key], dict):
            raise TypeError(f"Cannot set {path}: {key} is not a dictionary")

        current = current[key]

    current[keys[-1]] = copy.deepcopy(value)


def validate_config(cfg: Config) -> None:
    """Validate fields required to execute an experiment configuration."""
    if not isinstance(cfg, dict):
        raise TypeError("Configuration must be a dictionary")

    required_fields = (
        ("paths", "workload"),
        ("paths", "platform"),
        ("paths", "output"),
        ("run", "overrun_policy"),
        ("rl", "enabled"),
        ("rl", "learn"),
        ("logging", "level"),
    )

    for section_name, field_name in required_fields:
        section = cfg.get(section_name)

        if not isinstance(section, dict):
            raise TypeError(
                f"Configuration section must be a dictionary: {section_name}"
            )

        if field_name not in section:
            raise KeyError(f"Missing configuration field: {section_name}.{field_name}")

        if section[field_name] is None:
            raise ValueError(
                f"Configuration field is None: {section_name}.{field_name}"
            )

    normalize_log_level(cfg["logging"]["level"])
    if not isinstance(cfg.get("run"), dict):
        raise TypeError("Configuration section must be a dictionary: run")
    resolve_run_components(cfg["run"])
    stall_timeout = cfg["run"].get("stall_timeout")
    if stall_timeout is not None and (
        isinstance(stall_timeout, bool)
        or not isinstance(stall_timeout, (int, float))
        or stall_timeout <= 0
    ):
        raise ValueError("run.stall_timeout must be a positive number or None")

    for field_name in ("enabled", "learn"):
        if not isinstance(cfg["rl"][field_name], bool):
            raise TypeError(f"Configuration field rl.{field_name} must be a boolean")

    if cfg["rl"]["enabled"]:
        agent_name = cfg["rl"].get("assign")
        agents = cfg["rl"].get("agents")

        if not isinstance(agents, dict):
            raise TypeError("Configuration field rl.agents must be a dictionary")

        if agent_name not in agents:
            raise KeyError(f"RL agent not found: {agent_name}")

        if cfg["rl"].get("type") not in {"discrete", "continuous"}:
            raise ValueError("rl.type must be discrete or continuous")
        if cfg["rl"]["type"] == "discrete" and (
            cfg["rl"].get("dt") is None or cfg["rl"]["dt"] <= 0
        ):
            raise ValueError("Discrete policies require rl.dt > 0")
        for key in ("epochs", "episode_batch_size"):
            if not isinstance(cfg["rl"].get(key), int) or cfg["rl"][key] <= 0:
                raise ValueError(f"rl.{key} must be a positive integer")
        checkpoint = cfg["rl"].get("checkpoint")
        if checkpoint is not None:
            from pathlib import Path

            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        validate_validation_config(cfg)
        learner = cfg.get("gym", {}).get("learner")
        if isinstance(learner, dict) and (
            not isinstance(learner.get("name"), str)
            or not isinstance(learner.get("params", {}), dict)
        ):
            # Param names are checked against the learner when it is loaded.
            raise TypeError("gym.learner must be a name or {'name': str, 'params': dict}")


def validate_validation_config(cfg: Config) -> None:
    """Check ``rl.validation``: a held-out workload, an optional patience and window."""
    from pathlib import Path

    validation = cfg["rl"].get("validation")
    if validation is None:
        return
    if not isinstance(validation, dict):
        raise TypeError("rl.validation must be a dictionary or None")
    unexpected = set(validation) - {"workload", "patience", "window"}
    if unexpected:
        raise ValueError(f"Unexpected rl.validation keys: {sorted(unexpected)}")
    workload = validation.get("workload")
    if not isinstance(workload, str) or not Path(workload).is_file():
        raise FileNotFoundError(f"Validation workload not found: {workload}")
    if Path(workload).resolve() == Path(cfg["paths"]["workload"]).resolve():
        raise ValueError("rl.validation.workload must differ from the training workload")
    patience = validation.get("patience")
    if patience is not None and (
        not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0
    ):
        raise ValueError("rl.validation.patience must be None or a positive integer")
    window = validation.get("window", 1)
    if not isinstance(window, int) or isinstance(window, bool) or window < 1 or window % 2 == 0:
        raise ValueError("rl.validation.window must be a positive odd integer")
