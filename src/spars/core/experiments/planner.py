import copy
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from spars.core.registry import resolve_run_components
from spars.core.logging import is_quiet_level

from .configuration import deep_merge, merge_overrides, set_nested, validate_config
from .defaults import BASE_CONFIG, PROJECT_ROOT, RESULTS_ROOT
from .naming import (
    build_generated_config_path,
    build_output_path,
    build_result_run_name,
    build_run_folder_name,
    build_workload_root,
    safe_name,
)
from .records import print_generated_run, reserve_output_path
from .sweep import expand_sweep
from .worker_config import verify_generated_config, write_generated_config


RunSpec = dict[str, Any]
ExperimentConfig = dict[str, Any]


def _resolve_input_path(value: str, path_key: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        project_candidate = PROJECT_ROOT / path
        data_candidate = PROJECT_ROOT / "data" / path
        if project_candidate.is_file():
            path = project_candidate
        elif data_candidate.is_file():
            path = data_candidate
        else:
            raise FileNotFoundError(f"Experiment {path_key} file not found: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Experiment {path_key} file not found: {path}")
    return str(path.resolve())


def resolve_input_paths(cfg: ExperimentConfig) -> None:
    for path_key in ("workload", "platform"):
        cfg["paths"][path_key] = _resolve_input_path(cfg["paths"][path_key], path_key)
    validation = cfg.get("rl", {}).get("validation")
    if validation:
        validation["workload"] = _resolve_input_path(
            validation["workload"], "validation workload"
        )


def _phase_for_config(cfg: ExperimentConfig) -> tuple[bool, bool, str]:
    is_rl = bool(cfg["rl"]["enabled"])
    is_rl_training = is_rl and bool(cfg["rl"]["learn"])
    return (
        is_rl,
        is_rl_training,
        "train" if is_rl_training else ("test" if is_rl else "run"),
    )


def _configure_output(cfg: ExperimentConfig, output_path: Path) -> None:
    cfg["paths"]["output"] = str(output_path)


def _normalize_run_config(cfg: ExperimentConfig) -> None:
    run_config = cfg["run"]
    scheduler, power_management = resolve_run_components(run_config)
    run_config["scheduler"] = scheduler
    run_config["power_management"] = power_management


@contextmanager
def _exclusive_lock(path: Path):
    """Hold an exclusive lock on a file until the block ends (POSIX and Windows)."""
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _allocate_run_number() -> int:
    """Allocate a never-reused monotonic result number."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    counter_path = RESULTS_ROOT / ".run_counter"

    with _exclusive_lock(RESULTS_ROOT / ".run_counter.lock"):
        raw_value = (
            counter_path.read_text(encoding="utf-8").strip()
            if counter_path.exists()
            else ""
        )
        stored = int(raw_value) if raw_value else 0

        highest_existing = 0
        pattern = re.compile(r".+-(\d+)$")
        for candidate in RESULTS_ROOT.iterdir():
            if not candidate.is_dir():
                continue
            match = pattern.fullmatch(candidate.name)
            if match:
                highest_existing = max(highest_existing, int(match.group(1)))

        run_number = max(stored, highest_existing) + 1
        counter_path.write_text(str(run_number), encoding="utf-8")
        return run_number


def _require_experiment_identity(experiment: ExperimentConfig) -> str:
    name = str(experiment["name"])
    if not name.strip():
        raise ValueError("Experiment name must not be empty")
    return name


def _build_curriculum_spec(
    experiment,
    research_name,
    run_number,
    experiment_name,
    cfg,
    curriculum,
    sweep_values,
) -> RunSpec:
    if not curriculum:
        raise ValueError(f"Curriculum is empty: {experiment_name}")
    if not cfg["rl"]["enabled"]:
        raise ValueError(f"Curriculum requires rl.enabled=True: {experiment_name}")

    spec: RunSpec = {
        "research_name": research_name,
        "run_number": run_number,
        "name": experiment_name,
        "phase": "curriculum",
        "is_rl": True,
        "is_rl_training": True,
        "is_curriculum": True,
        "base_cfg": cfg,
        "curriculum": copy.deepcopy(curriculum),
        "test_overrides": copy.deepcopy(experiment.get("test_overrides") or {}),
        "sweep": copy.deepcopy(sweep_values),
    }
    if not is_quiet_level(cfg.get("logging", {}).get("level", "INFO")):
        print("\nGenerated curriculum:")
        print("run:", build_result_run_name(research_name, run_number))
        print("experiment:", experiment_name)
        print("stages:", len(curriculum))
    return spec


def _build_standard_run_spec(
    experiment,
    research_name,
    run_number,
    experiment_name,
    cfg,
    sweep_values,
    used_outputs,
) -> RunSpec:
    output_path = build_output_path(
        cfg, research_name, run_number, experiment_name
    ).resolve()
    _configure_output(cfg, output_path)
    validate_config(cfg)
    reserve_output_path(output_path, used_outputs)
    is_rl, is_rl_training, phase = _phase_for_config(cfg)
    config_path = build_generated_config_path(
        cfg, research_name, run_number, experiment_name
    ).resolve()
    write_generated_config(cfg, config_path)
    spec: RunSpec = {
        "research_name": research_name,
        "run_number": run_number,
        "name": experiment_name,
        "phase": phase,
        "is_rl": is_rl,
        "is_rl_training": is_rl_training,
        "is_curriculum": False,
        "cfg": cfg,
        "config_path": config_path,
        "output_path": output_path,
        "sweep": copy.deepcopy(sweep_values),
    }
    if is_rl_training:
        spec["test_overrides"] = copy.deepcopy(experiment.get("test_overrides") or {})
    print_generated_run(spec)
    return spec


def build_curriculum_stage_spec(
    curriculum_spec: RunSpec,
    stage: dict[str, Any],
    stage_index: int,
    checkpoint_path: str | Path | None,
) -> RunSpec:
    stage_number = stage_index + 1
    stage_name = safe_name(stage.get("name", f"stage-{stage_number}"))
    research_name = curriculum_spec["research_name"]
    run_number = curriculum_spec["run_number"]
    experiment_name = curriculum_spec["name"]

    cfg = deep_merge(curriculum_spec["base_cfg"], stage.get("overrides", {}))
    _normalize_run_config(cfg)
    cfg["paths"]["workload"] = stage["workload"]
    resolve_input_paths(cfg)
    cfg["rl"]["enabled"] = True
    cfg["rl"]["learn"] = True
    cfg["rl"]["epochs"] = int(stage["epochs"])
    cfg["rl"]["checkpoint"] = (
        str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
    )

    run_folder = f"{build_run_folder_name(cfg)}_{stage_number:02d}-{stage_name}"
    output_path = (
        build_workload_root(cfg, research_name, run_number, experiment_name)
        / run_folder
    ).resolve()
    _configure_output(cfg, output_path)
    reserve_output_path(output_path)
    config_path = build_generated_config_path(
        cfg,
        research_name,
        run_number,
        experiment_name,
        suffix=f"{stage_number:02d}-{stage_name}",
    ).resolve()
    validate_config(cfg)
    write_generated_config(cfg, config_path)

    spec: RunSpec = {
        "research_name": research_name,
        "run_number": run_number,
        "name": f"{experiment_name} - {stage_name}",
        "experiment_name": experiment_name,
        "phase": f"train-{stage_name}",
        "is_rl": True,
        "is_rl_training": True,
        "is_curriculum": False,
        "cfg": cfg,
        "config_path": config_path,
        "output_path": output_path,
        "sweep": copy.deepcopy(curriculum_spec["sweep"]),
        "test_overrides": copy.deepcopy(curriculum_spec.get("test_overrides") or {}),
    }
    print_generated_run(spec)
    return spec


def build_run_specs(
    experiments: list[ExperimentConfig],
    research_name: str,
    reporting: dict | None = None,
) -> list[RunSpec]:
    enabled_experiments = [
        experiment for experiment in experiments if experiment.get("enabled", True)
    ]
    if not enabled_experiments:
        raise RuntimeError("No enabled experiments were generated")

    run_specs: list[RunSpec] = []
    used_outputs: set[str] = set()
    run_number = _allocate_run_number()

    for experiment in enabled_experiments:
        experiment_name = _require_experiment_identity(experiment)
        experiment_cfg = merge_overrides(BASE_CONFIG, experiment.get("overrides", {}))

        for sweep_values in expand_sweep(experiment.get("sweep")):
            cfg = copy.deepcopy(experiment_cfg)
            for config_path, value in sweep_values.items():
                set_nested(cfg, config_path, value)

            cfg["reporting"] = copy.deepcopy(
                reporting or {"metrics": False, "gantt": False}
            )
            cfg["experiment"] = {
                "name": experiment_name,
                "display_name": experiment.get("display_name", experiment_name),
                "research_name": research_name,
                "run_number": run_number,
                "run_root": str(
                    (
                        RESULTS_ROOT / build_result_run_name(research_name, run_number)
                    ).resolve()
                ),
                "sweep": copy.deepcopy(sweep_values),
            }
            curriculum = experiment.get("curriculum")
            if curriculum is None:
                resolve_input_paths(cfg)
            if curriculum is not None:
                run_specs.append(
                    _build_curriculum_spec(
                        experiment,
                        research_name,
                        run_number,
                        experiment_name,
                        cfg,
                        curriculum,
                        sweep_values,
                    )
                )
            else:
                _normalize_run_config(cfg)
                run_specs.append(
                    _build_standard_run_spec(
                        experiment,
                        research_name,
                        run_number,
                        experiment_name,
                        cfg,
                        sweep_values,
                        used_outputs,
                    )
                )

    if not run_specs:
        raise RuntimeError("No enabled experiments were generated")
    return run_specs


def build_automatic_test_spec(
    training_spec: RunSpec, checkpoint_path: str | Path
) -> RunSpec:
    research_name = training_spec["research_name"]
    run_number = training_spec["run_number"]
    experiment_name = training_spec.get("experiment_name", training_spec["name"])
    test_overrides = training_spec.get("test_overrides") or {}

    test_cfg = deep_merge(copy.deepcopy(training_spec["cfg"]), test_overrides)
    # The test is a plain evaluation; validation only selects the checkpoint.
    validation = test_cfg["rl"].pop("validation", None)
    resolve_input_paths(test_cfg)
    if validation:
        # Train / validation / test split: the test must be held out of both.
        held_out = {
            "training": training_spec["cfg"]["paths"]["workload"],
            "validation": validation["workload"],
        }
        for role, workload in held_out.items():
            if Path(workload).resolve() == Path(test_cfg["paths"]["workload"]):
                raise ValueError(f"Test workload must differ from the {role} workload")
    test_cfg["rl"]["enabled"] = True
    test_cfg["rl"]["learn"] = False
    test_cfg["rl"]["checkpoint"] = str(Path(checkpoint_path).resolve())

    test_output_path = build_output_path(
        test_cfg, research_name, run_number, experiment_name
    ).resolve()
    _configure_output(test_cfg, test_output_path)
    reserve_output_path(test_output_path)
    test_config_path = build_generated_config_path(
        test_cfg, research_name, run_number, experiment_name
    ).resolve()

    expected_workload = test_overrides.get("paths", {}).get("workload")
    if expected_workload is not None:
        expected_path = Path(expected_workload)
        if not expected_path.is_absolute():
            project_candidate = PROJECT_ROOT / expected_path
            data_candidate = PROJECT_ROOT / "data" / expected_path
            expected_path = (
                project_candidate if project_candidate.is_file() else data_candidate
            )
        expected_workload = str(expected_path.resolve())
        if test_cfg["paths"]["workload"] != expected_workload:
            raise RuntimeError(
                "Test workload override was not applied: "
                f"expected {expected_workload!r}, got {test_cfg['paths']['workload']!r}"
            )

    validate_config(test_cfg)
    write_generated_config(test_cfg, test_config_path)
    spec: RunSpec = {
        "research_name": research_name,
        "run_number": run_number,
        "name": experiment_name,
        "phase": "test",
        "is_rl": True,
        "is_rl_training": False,
        "cfg": test_cfg,
        "config_path": test_config_path,
        "output_path": test_output_path,
        "sweep": copy.deepcopy(training_spec["sweep"]),
    }
    verify_generated_config(spec)
    print_generated_run(spec)
    return spec
