import re
from pathlib import Path
from typing import Any

from spars.core.registry import get_run_timeout, resolve_run_components

from .defaults import RESULTS_ROOT


def safe_name(value: Any) -> str:
    """Convert an arbitrary value into a filesystem-safe name fragment."""
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)

    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-")


def file_stem(path: str | Path) -> str:
    return safe_name(Path(path).stem)


def timeout_tag(timeout: Any) -> str:
    if timeout is None:
        return "no-timeout"

    return f"timeout-{safe_name(timeout)}"


def build_reward_weight_tags(cfg: dict[str, Any]) -> list[str]:
    reward_params = cfg.get("gym", {}).get("reward", {}).get("params", {})

    if not isinstance(reward_params, dict):
        raise TypeError("gym.reward.params must be a dictionary")

    weights: list[tuple[int, str]] = []

    for parameter_name, value in reward_params.items():
        match = re.fullmatch(
            r"weight(\d+)",
            str(parameter_name),
        )

        if match is None:
            continue

        weight_number = int(match.group(1))
        weights.append(
            (
                weight_number,
                f"w{weight_number}{safe_name(value)}",
            )
        )

    weights.sort(key=lambda item: item[0])
    return [tag for _, tag in weights]


def build_run_folder_name(cfg: dict[str, Any]) -> str:
    scheduler, _ = resolve_run_components(cfg["run"])
    parts = [safe_name(scheduler["name"])]
    if cfg["rl"]["enabled"]:
        parts.extend(
            [
                safe_name(cfg["rl"]["assign"]),
                *build_reward_weight_tags(cfg),
                "train" if cfg["rl"]["learn"] else "test",
            ]
        )
    else:
        parts.append(timeout_tag(get_run_timeout(cfg["run"])))
    represented = {
        "paths.platform",
        "paths.workload",
        "run.power_management.config.timeout",
    }
    for key, value in cfg.get("experiment", {}).get("sweep", {}).items():
        if key in represented or re.fullmatch(r"gym\.reward\.params\.weight\d+", key):
            continue
        parts.append(f"{safe_name(key)}-{safe_name(value)}")
    return "_".join(parts)


def build_result_run_name(research_name: Any, run_number: Any) -> str:
    research = safe_name(research_name)
    if not research:
        raise ValueError("research_name must not be empty")

    try:
        number = int(run_number)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"run_number must be a positive integer: {run_number!r}"
        ) from error

    if number < 1:
        raise ValueError(f"run_number must be a positive integer: {run_number!r}")

    return f"{research}-{number:06d}"


def build_workload_root(
    cfg: dict[str, Any],
    research_name: Any,
    run_number: Any,
    experiment_name: Any,
) -> Path:
    platform_name = file_stem(cfg["paths"]["platform"])
    workload_name = file_stem(cfg["paths"]["workload"])

    return (
        RESULTS_ROOT
        / build_result_run_name(research_name, run_number)
        / safe_name(experiment_name)
        / platform_name
        / workload_name
    )


def build_output_path(
    cfg: dict[str, Any],
    research_name: Any,
    run_number: Any,
    experiment_name: Any,
) -> Path:
    return build_workload_root(
        cfg,
        research_name,
        run_number,
        experiment_name,
    ) / build_run_folder_name(cfg)


def build_generated_config_path(
    cfg: dict[str, Any],
    research_name: Any,
    run_number: Any,
    experiment_name: Any,
    suffix: Any = None,
) -> Path:
    file_name = build_run_folder_name(cfg)

    if suffix is not None:
        file_name = f"{file_name}_{safe_name(suffix)}"

    return (
        build_workload_root(
            cfg,
            research_name,
            run_number,
            experiment_name,
        )
        / "generated_configs"
        / f"{file_name}.yaml"
    )
