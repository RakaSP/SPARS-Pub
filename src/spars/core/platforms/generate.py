"""Platforms.

A platform is one or more node types; each type is a group of identical nodes
(one ``PlatformConfig``). Node ids run on from one type to the next. Every
DVFS profile scales its type's base compute power, idle power and compute
speed by its own factors; "base" has all factors 1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class DVFSProfile:
    name: str
    compute_power: float = 1.0
    idle_power: float = 1.0
    compute_speed: float = 1.0


@dataclass(frozen=True)
class PlatformConfig:
    num_nodes: int = 64
    node_name: str = "compute_node"
    base_compute_power: float = 2100.0
    base_idle_power: float = 600.0
    switching_on_power: float = 600.0
    switching_off_power: float = 600.0
    sleeping_power: float = 60.0
    switching_on_time: float = 1200.0
    switching_off_time: float = 600.0
    switching_on_std: float = 60.0
    switching_off_std: float = 60.0
    compute_speed_variation: float = 1.0
    # One base profile by default; add, edit or remove others as needed.
    dvfs_profiles: tuple[DVFSProfile, ...] = field(default_factory=lambda: (DVFSProfile("base"),))
    dvfs_mode: str = "base"


def compress_id_ranges(ids: Iterable[int]) -> list[int | str]:
    """[0, 1, 2, 3, 5, 8, 9, 10] -> ["0-3", 5, "8-10"]."""
    values = sorted(set(ids))
    out: list[int | str] = []
    start = end = values[0]
    for value in values[1:] + [None]:  # type: ignore[list-item]
        if value is not None and value == end + 1:
            end = value
            continue
        out.append(start if start == end else f"{start}-{end}")
        if value is not None:
            start = end = value
    return out


def validate_config(config: PlatformConfig) -> None:
    if config.num_nodes <= 0:
        raise ValueError("number of nodes must be positive")
    if not config.node_name.strip():
        raise ValueError("node name must not be empty")
    for label, value in (
        ("base compute power", config.base_compute_power),
        ("base idle power", config.base_idle_power),
        ("switching on power", config.switching_on_power),
        ("switching off power", config.switching_off_power),
        ("sleeping power", config.sleeping_power),
        ("switching on time", config.switching_on_time),
        ("switching off time", config.switching_off_time),
        ("switching on std", config.switching_on_std),
        ("switching off std", config.switching_off_std),
    ):
        if value < 0:
            raise ValueError(f"{label} cannot be negative")
    if config.compute_speed_variation <= 0:
        raise ValueError("compute speed must be positive")
    if not config.dvfs_profiles:
        raise ValueError("a platform needs at least one DVFS profile")
    names = [profile.name for profile in config.dvfs_profiles]
    if any(not name.strip() for name in names):
        raise ValueError("every DVFS profile needs a name")
    if len(set(names)) != len(names):
        raise ValueError("DVFS profile names must be different")
    for profile in config.dvfs_profiles:
        if min(profile.compute_power, profile.idle_power) < 0 or profile.compute_speed <= 0:
            raise ValueError(f"DVFS profile {profile.name}: factors must be positive")
    if config.dvfs_mode not in names:
        raise ValueError(f"the initial DVFS mode {config.dvfs_mode!r} is not one of the profiles")


def warnings(config: PlatformConfig) -> list[str]:
    """Settings that are allowed but probably not what was meant."""
    out = []
    if config.base_idle_power > config.base_compute_power:
        out.append(f"{config.node_name}: idle power is above compute power")
    if config.sleeping_power > config.base_idle_power:
        out.append(f"{config.node_name}: sleeping uses more power than idling, so switching off saves nothing")
    return out


def generate_platform(configs: PlatformConfig | Sequence[PlatformConfig]) -> dict[str, Any]:
    """The platform JSON for one node type or several (ids run on from type to type)."""
    configs = [configs] if isinstance(configs, PlatformConfig) else list(configs)
    if not configs:
        raise ValueError("a platform needs at least one node type")
    names = [config.node_name.strip() for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("node type names must be different")
    machines, first_id = [], 0
    for config in configs:
        machines.append(_machine(config, first_id))
        first_id += config.num_nodes
    return {"machines": machines}


def _machine(config: PlatformConfig, first_id: int) -> dict[str, Any]:
    validate_config(config)
    base_speed = 1.0 * config.compute_speed_variation
    profiles = {
        profile.name: {
            "power_compute": config.base_compute_power * profile.compute_power,
            "power_idle": config.base_idle_power * profile.idle_power,
            "compute_speed": base_speed * profile.compute_speed,
        }
        for profile in config.dvfs_profiles
    }

    def timed(state: str, seconds: float, std: float) -> list[dict[str, Any]]:
        return [{"state": state, "transition_time": seconds, "std": std}]

    fixed = [{"state": "", "transition_time": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}]
    machine = {
        "node_name": config.node_name.strip(),
        "id": compress_id_ranges(range(first_id, first_id + config.num_nodes)),
        "dvfs_profiles": profiles,
        "dvfs_mode": config.dvfs_mode,
        "states": {
            "active": {
                "power": "from_dvfs",
                "compute_speed": "from_dvfs",
                "can_run_jobs": True,
                "transitions": [{**fixed[0], "state": "switching_off"}],
            },
            "switching_off": {
                "power": config.switching_off_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": timed("sleeping", config.switching_off_time, config.switching_off_std),
            },
            "switching_on": {
                "power": config.switching_on_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": timed("active", config.switching_on_time, config.switching_on_std),
            },
            "sleeping": {
                "power": config.sleeping_power,
                "compute_speed": "from_dvfs",
                "can_run_jobs": False,
                "transitions": [{**fixed[0], "state": "switching_on"}],
            },
        },
    }
    return machine


def write_platform(platform: dict[str, Any], path: Path) -> None:
    """Write the JSON without a partial file and without overwriting; the platform must load."""
    from spars.core.simulator.machine import Machine

    if path.exists():
        raise FileExistsError(f"{path.name} already exists; it is not overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(platform, indent=4) + "\n", encoding="utf-8")
    try:
        Machine(str(temporary), 0)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
