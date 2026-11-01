import copy
from pathlib import Path
from typing import Any

from spars.core.registry import get_run_timeout, resolve_run_components
from spars.core.logging import is_quiet_level

from .naming import build_result_run_name, build_reward_weight_tags


def reserve_output_path(
    output_path: Path, used_outputs: set[str] | None = None
) -> None:
    resolved_output = output_path.resolve()
    resolved_text = str(resolved_output)
    if used_outputs is not None and resolved_text in used_outputs:
        raise ValueError(
            f"Duplicate result path generated:\n{resolved_output}\n\nTwo experiments generated the same output path."
        )
    if resolved_output.exists():
        raise FileExistsError(
            f"Result path already exists; refusing to overwrite existing results:\n{resolved_output}"
        )
    if used_outputs is not None:
        used_outputs.add(resolved_text)


def print_generated_run(run_spec: dict[str, Any]) -> None:
    cfg = run_spec["cfg"]
    if is_quiet_level(cfg.get("logging", {}).get("level", "INFO")):
        return
    scheduler, power_management = resolve_run_components(cfg["run"])
    print("\nGenerated run:")
    print(
        "run:", build_result_run_name(run_spec["research_name"], run_spec["run_number"])
    )
    print("experiment:", run_spec["name"])
    print("phase:", run_spec["phase"])
    print("config:", run_spec["config_path"])
    print("platform:", cfg["paths"]["platform"])
    print("workload:", cfg["paths"]["workload"])
    print("scheduler:", scheduler["name"])
    print("power manager:", power_management["name"])
    print("timeout:", get_run_timeout(cfg["run"]))
    print("output:", run_spec["output_path"])
    print("run parameters:", cfg["run"])
    if cfg["rl"]["enabled"]:
        print("reward weights:", build_reward_weight_tags(cfg))
    if run_spec["sweep"]:
        print("sweep:", run_spec["sweep"])
    if cfg["rl"].get("checkpoint") is not None:
        print("checkpoint:", cfg["rl"]["checkpoint"])


def make_run_record(
    run_spec: dict[str, Any], output_path: str | Path | None = None
) -> dict[str, Any]:
    cfg = run_spec["cfg"]
    scheduler, power_management = resolve_run_components(cfg["run"])
    effective_output_path = (
        output_path if output_path is not None else run_spec["output_path"]
    )
    record = {
        "run_number": run_spec["run_number"],
        "name": run_spec["name"],
        "display_name": cfg.get("experiment", {}).get("display_name", run_spec["name"]),
        "research_name": run_spec["research_name"],
        "run_root": cfg["experiment"]["run_root"],
        "phase": run_spec["phase"],
        "output": str(effective_output_path),
        "platform": cfg["paths"]["platform"],
        "workload": cfg["paths"]["workload"],
        "algorithm": scheduler["name"],
        "timeout": get_run_timeout(cfg["run"]),
        "run_parameters": copy.deepcopy(cfg["run"]),
        "sweep": copy.deepcopy(run_spec["sweep"]),
        "config_path": str(run_spec["config_path"]),
    }

    from spars.core.analysis.automatic import record_run

    record_run(record)
    return record
