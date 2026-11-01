"""Synthetic Markovian workloads.

The arrival rate follows from a utilization target: the exponential
interarrival scale is

    E[resources per job] * E[runtime] / (nb_res * target_utilization)

The random draws use a fixed order (resources, interarrivals, runtimes, requested
times), so one seed always gives the same workload.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeneratorConfig:
    num_jobs: int = 3000
    nb_res: int = 64
    target_utilization: float = 0.40
    mean_runtime: float = 7200.0
    resource_geometric_p: float = 0.35
    reqtime_bias: float = 1.20
    seed: int = 43
    # Every job stores this string in "profile", a key of the top-level "profiles".
    profile_name: str = "100"
    profile_cpu: int = 10_000_000_000_000_000_000_000
    profile_com: int = 0
    user_id: int = 0


def validate_config(config: GeneratorConfig) -> None:
    if config.num_jobs <= 0:
        raise ValueError("num_jobs must be positive")
    if config.nb_res <= 0:
        raise ValueError("nb_res must be positive")
    if config.target_utilization <= 0:
        raise ValueError("target_utilization must be positive")
    if config.mean_runtime <= 0:
        raise ValueError("mean_runtime must be positive")
    if not 0 < config.resource_geometric_p <= 1:
        raise ValueError("resource_geometric_p must be in (0, 1]")
    if config.reqtime_bias <= 0:
        raise ValueError("reqtime_bias must be positive")
    if not config.profile_name:
        raise ValueError("profile_name must not be empty")


def warnings(config: GeneratorConfig) -> list[str]:
    """Settings that are allowed but probably not what was meant."""
    out = []
    if config.target_utilization > 1:
        out.append(
            f"Target utilization {config.target_utilization:g} is above 1: jobs ask for more node-time "
            "than the platform has, so the queue keeps growing."
        )
    if config.reqtime_bias < 1:
        out.append(
            f"Requested time bias {config.reqtime_bias:g} is below 1: on average jobs ask for less time "
            "than they run, so many overrun their request."
        )
    return out


def expected_resources_per_job(config: GeneratorConfig) -> float:
    """E[min(Geometric(p), nb_res)]."""
    p = config.resource_geometric_p
    return (1.0 - (1.0 - p) ** config.nb_res) / p


def mean_interarrival(config: GeneratorConfig) -> float:
    return (
        expected_resources_per_job(config)
        * config.mean_runtime
        / (config.nb_res * config.target_utilization)
    )


def generate_workload(config: GeneratorConfig) -> dict[str, Any]:
    validate_config(config)
    rng = np.random.default_rng(config.seed)
    resources = np.minimum(
        rng.geometric(config.resource_geometric_p, size=config.num_jobs), config.nb_res
    ).astype(int)
    subtimes = np.cumsum(rng.exponential(scale=mean_interarrival(config), size=config.num_jobs))
    runtimes = rng.exponential(scale=config.mean_runtime, size=config.num_jobs)
    reqtimes = rng.exponential(scale=config.mean_runtime * config.reqtime_bias, size=config.num_jobs)
    return {
        "nb_res": config.nb_res,
        "jobs": [
            {
                "job_id": index + 1,
                "res": int(resources[index]),
                "subtime": float(subtimes[index]),
                "reqtime": float(reqtimes[index]),
                "runtime": float(runtimes[index]),
                "profile": config.profile_name,
                "user_id": config.user_id,
            }
            for index in range(config.num_jobs)
        ],
        "profiles": {
            config.profile_name: {
                "cpu": config.profile_cpu,
                "com": config.profile_com,
                "type": "parallel_homogeneous",
            }
        },
    }


def summarize(config: GeneratorConfig, workload: dict[str, Any]) -> str:
    """One line: the interarrival time asked for and the load that came out."""
    jobs = workload["jobs"]
    res = np.asarray([job["res"] for job in jobs], dtype=float)
    run = np.asarray([job["runtime"] for job in jobs], dtype=float)
    gaps = np.diff(np.concatenate(([0.0], [job["subtime"] for job in jobs])))
    load = res.mean() * run.mean() / (config.nb_res * gaps.mean())
    return (
        f"{len(jobs)} jobs, mean interarrival {gaps.mean():.0f} s "
        f"(asked {mean_interarrival(config):.0f} s), offered utilization {load:.2f}"
    )


def write_workload(workload: dict[str, Any], path: Path) -> None:
    """Write the JSON without ever leaving a partial file, and never overwrite."""
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; it is not overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(workload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, path)
