from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psutil

from spars.core.analysis.automatic import generate_all_plots
from spars.core.logging import is_quiet_level
from spars.core.runner.cores import (
    DEFAULT_SUPPORT_CORES,
    CoreAllocation,
    resolve_core_allocation,
)
from spars.core.runner.worker import EventDashboard, ProgressDashboard, WorkerBatchFailed, runner_main

from .checkpoints import find_best_epoch_checkpoint
from .defaults import (
    DEFAULT_CORES,
    GENERATE_GANTT_PLOTS,
    GENERATE_METRICS_PLOT,
    PROJECT_ROOT,
    WORKER_SCRIPT,
)
from .discovery import SELECTED_DEFINITIONS_ENV, SELECTED_RESEARCH_ENV, load_experiments
from .planner import (
    build_automatic_test_spec,
    build_curriculum_stage_spec,
    build_run_specs,
)
from .records import make_run_record
from .worker_config import verify_generated_config
from spars.core.analysis.automatic import build_run_label
from spars.core.registry import get_run_timeout


# JSON {"level": ..., "file": ...} set by the dashboard for every experiment.
LOGGING_OVERRIDE_ENV = "SPARS_LOGGING_OVERRIDE"

RunSpec = dict[str, Any]
RunRecord = dict[str, Any]
# Set by main(): every worker runs on one of these Simulation cores.
_core_allocation: CoreAllocation | None = None


def _log_file(cfg: dict[str, Any]) -> Path | None:
    """The run's log file; a relative path is inside the batch folder (results/<Set>-NNNNNN/)."""
    name = cfg.get("logging", {}).get("file")
    if not name:
        return None
    path = Path(name)
    return path if path.is_absolute() else Path(cfg["experiment"]["run_root"]) / path


def _run_spec_config(run_spec: RunSpec) -> dict[str, Any]:
    cfg = run_spec.get("cfg")
    if isinstance(cfg, dict):
        return cfg

    base_cfg = run_spec.get("base_cfg")
    if isinstance(base_cfg, dict):
        return base_cfg

    return {}


def _is_quiet_run_spec(run_spec: RunSpec) -> bool:
    return is_quiet_level(
        _run_spec_config(run_spec).get("logging", {}).get("level", "INFO")
    )


def _expected_worker_count(run_spec: RunSpec) -> int:
    if run_spec.get("is_curriculum", False):
        return len(run_spec.get("curriculum", [])) + 1
    if run_spec.get("is_rl_training", False):
        return 2
    return 1


def _prepare_progress_dashboard(
    run_specs: list[RunSpec],
    events: bool = False,
) -> ProgressDashboard | None:
    if not run_specs:
        return None

    # The TUI dashboard draws progress at any log level.
    if not events:
        quiet_flags = [_is_quiet_run_spec(run_spec) for run_spec in run_specs]
        if any(quiet_flags) and not all(quiet_flags):
            raise ValueError(
                "Selected experiments mix QUIET and non-QUIET logging levels. "
                "Use QUIET for all selected experiments or none of them."
            )
        if not all(quiet_flags):
            return None

    group_keys = [str(run_spec["name"]) for run_spec in run_specs]
    group_totals = Counter(group_keys)
    group_indices: defaultdict[str, int] = defaultdict(int)
    experiment_totals: defaultdict[str, int] = defaultdict(int)
    total_simulations = 0

    for run_spec, group_key in zip(run_specs, group_keys, strict=True):
        cfg = _run_spec_config(run_spec)
        name = build_run_label(
            {
                "name": run_spec["name"],
                "display_name": cfg.get("experiment", {}).get(
                    "display_name", run_spec["name"]
                ),
                "timeout": get_run_timeout(cfg["run"]),
                "sweep": run_spec.get("sweep", {}),
            }
        )
        group_indices[group_key] += 1
        run_spec["_progress_group"] = group_key
        run_spec["_progress_label"] = (
            f"{name} #{group_indices[group_key]}"
            if group_totals[group_key] > 1
            else name
        )

        expected = _expected_worker_count(run_spec)
        experiment_totals[group_key] += expected
        total_simulations += expected

    return (EventDashboard if events else ProgressDashboard)(
        total_simulations=total_simulations,
        experiment_totals=experiment_totals,
    )


def _inherit_progress_metadata(child: RunSpec, parent: RunSpec) -> None:
    child["_progress_group"] = parent.get(
        "_progress_group",
        parent["name"],
    )
    child["_progress_label"] = parent.get("_progress_label", parent["name"])


async def run_worker_batch(
    run_specs: list[RunSpec],
    concurrency: int,
    progress_dashboard: ProgressDashboard | None = None,
) -> None:
    if not run_specs:
        return

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    config_paths = [verify_generated_config(run_spec) for run_spec in run_specs]

    await runner_main(
        script_path=str(Path(WORKER_SCRIPT).resolve()),
        args_=[[str(config_path)] for config_path in config_paths],
        THREAD_LIMIT=concurrency,
        simulation_cpus=(
            _core_allocation.simulation if _core_allocation is not None else None
        ),
        progress_dashboard=progress_dashboard,
        log_files=[_log_file(_run_spec_config(run_spec)) for run_spec in run_specs],
        progress_labels=(
            [
                str(run_spec.get("_progress_label", run_spec["name"]))
                for run_spec in run_specs
            ]
            if progress_dashboard is not None
            else None
        ),
        progress_groups=(
            [
                str(run_spec.get("_progress_group", run_spec["name"]))
                for run_spec in run_specs
            ]
            if progress_dashboard is not None
            else None
        ),
    )


def _resolve_curriculum_start_checkpoint(
    curriculum_spec: RunSpec,
) -> Path | None:
    configured_checkpoint = curriculum_spec["base_cfg"].get("rl", {}).get("checkpoint")
    if configured_checkpoint is None:
        return None

    checkpoint_path = Path(configured_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Curriculum starting checkpoint was not found:\n{checkpoint_path}"
        )

    return checkpoint_path


class _PartialBatchFailure(Exception):
    """Some runs of a concurrent group failed; ``records`` holds the ones that did not."""

    def __init__(self, original: Exception, records: list[RunRecord]) -> None:
        super().__init__(str(original))
        self.original = original
        self.records = records


async def _run_pending_non_rl(
    pending_run_specs: list[RunSpec],
    cores: int,
    progress_dashboard: ProgressDashboard | None,
) -> list[RunRecord]:
    if not pending_run_specs:
        return []

    if progress_dashboard is None:
        print("\nRunning non-RL experiments:")
        print("runs:", len(pending_run_specs))
        print("concurrency:", cores)

    run_specs = list(pending_run_specs)
    pending_run_specs.clear()  # also when the batch fails, so it is not run twice
    try:
        await run_worker_batch(
            run_specs,
            concurrency=cores,
            progress_dashboard=progress_dashboard,
        )
    except WorkerBatchFailed as error:
        # One failed run in the group must not cost the plots of the others that succeeded.
        succeeded = [spec for index, spec in enumerate(run_specs) if index not in error.failed_indices]
        raise _PartialBatchFailure(error, [make_run_record(spec) for spec in succeeded]) from error

    return [make_run_record(run_spec) for run_spec in run_specs]


async def _run_curriculum(
    curriculum_spec: RunSpec,
    progress_dashboard: ProgressDashboard | None,
) -> list[RunRecord]:
    run_records: list[RunRecord] = []
    checkpoint_path = _resolve_curriculum_start_checkpoint(curriculum_spec)

    if checkpoint_path is not None and progress_dashboard is None:
        print(
            "\nCurriculum starting checkpoint:",
            checkpoint_path,
        )

    final_stage_spec: RunSpec | None = None

    for stage_index, stage in enumerate(curriculum_spec["curriculum"]):
        stage_spec = build_curriculum_stage_spec(
            curriculum_spec,
            stage,
            stage_index,
            checkpoint_path,
        )

        _inherit_progress_metadata(stage_spec, curriculum_spec)

        if progress_dashboard is None:
            print("\nRunning curriculum stage:")
            print("stage:", stage_index + 1)
            print("name:", stage_spec["name"])
            print(
                "workload:",
                stage_spec["cfg"]["paths"]["workload"],
            )
            print(
                "epochs:",
                stage_spec["cfg"]["rl"]["epochs"],
            )
            print(
                "checkpoint:",
                stage_spec["cfg"]["rl"]["checkpoint"],
            )
            print("concurrency: 1")

        await run_worker_batch(
            [stage_spec],
            concurrency=1,
            progress_dashboard=progress_dashboard,
        )

        checkpoint_path = find_best_epoch_checkpoint(stage_spec["output_path"])
        if progress_dashboard is None:
            print("saved checkpoint:", checkpoint_path)

        run_records.append(
            make_run_record(
                stage_spec,
                output_path=checkpoint_path.parent,
            )
        )
        final_stage_spec = stage_spec

    if final_stage_spec is None or checkpoint_path is None:
        raise RuntimeError(
            f"Curriculum produced no checkpoint: {curriculum_spec['name']}"
        )

    test_spec = build_automatic_test_spec(
        final_stage_spec,
        checkpoint_path,
    )
    _inherit_progress_metadata(test_spec, curriculum_spec)

    if progress_dashboard is None:
        print("\nRunning curriculum test:")
        print(
            "workload:",
            test_spec["cfg"]["paths"]["workload"],
        )
        print("checkpoint:", checkpoint_path)
        print("concurrency: 1")

    await run_worker_batch(
        [test_spec],
        concurrency=1,
        progress_dashboard=progress_dashboard,
    )
    run_records.append(make_run_record(test_spec))

    return run_records


async def _run_rl_experiment(
    run_spec: RunSpec,
    progress_dashboard: ProgressDashboard | None,
) -> list[RunRecord]:
    if progress_dashboard is None:
        print("\nRunning RL experiment exclusively:")
        print("experiment:", run_spec["name"])
        print("phase:", run_spec["phase"])

        print("concurrency: 1")

    await run_worker_batch(
        [run_spec],
        concurrency=1,
        progress_dashboard=progress_dashboard,
    )

    if not run_spec["is_rl_training"]:
        return [make_run_record(run_spec)]

    checkpoint_path = find_best_epoch_checkpoint(run_spec["output_path"])
    run_records = [
        make_run_record(
            run_spec,
            output_path=checkpoint_path.parent,
        )
    ]

    if progress_dashboard is None:
        print("\nBest epoch checkpoint:", checkpoint_path)

    test_spec = build_automatic_test_spec(
        run_spec,
        checkpoint_path,
    )
    _inherit_progress_metadata(test_spec, run_spec)

    if progress_dashboard is None:
        print("\nRunning RL test immediately after training:")
        print("experiment:", test_spec["name"])
        print("concurrency: 1")

    await run_worker_batch(
        [test_spec],
        concurrency=1,
        progress_dashboard=progress_dashboard,
    )
    run_records.append(make_run_record(test_spec))

    return run_records


async def execute_run_specs(
    run_specs: list[RunSpec],
    cores: int,
    progress_dashboard: ProgressDashboard | None = None,
) -> list[RunRecord]:
    if cores < 1:
        raise ValueError("cores must be at least 1")

    plot_records: list[RunRecord] = []
    pending_non_rl: list[RunSpec] = []
    failures: list[Exception] = []

    async def attempt(step) -> None:
        # A failed group must not stop the groups after it.
        try:
            plot_records.extend(await step)
        except _PartialBatchFailure as error:
            plot_records.extend(error.records)  # keep the plots of the runs that did succeed
            failures.append(error.original)
        except Exception as error:
            failures.append(error)

    for run_spec in run_specs:
        if run_spec.get("is_curriculum", False):
            await attempt(_run_pending_non_rl(pending_non_rl, cores, progress_dashboard))
            await attempt(_run_curriculum(run_spec, progress_dashboard))
            continue

        if not run_spec["is_rl"]:
            pending_non_rl.append(run_spec)
            continue

        await attempt(_run_pending_non_rl(pending_non_rl, cores, progress_dashboard))
        await attempt(_run_rl_experiment(run_spec, progress_dashboard))

    await attempt(_run_pending_non_rl(pending_non_rl, cores, progress_dashboard))
    if failures:
        error = RuntimeError("\n".join(str(failure) for failure in failures))
        error.partial_records = plot_records  # so the runs that did succeed can still be plotted
        raise error
    return plot_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute SPARS experiment definitions",
    )
    parser.add_argument(
        "cores",
        metavar="WORKERS",
        help="Maximum non-RL workers; RL jobs run exclusively",
        nargs="?",
        type=int,
        default=DEFAULT_CORES,
    )
    parser.add_argument(
        "--support-cores",
        type=int,
        default=DEFAULT_SUPPORT_CORES,
        help="Cores for everything other than simulation (logging, plotting)",
    )
    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=GENERATE_METRICS_PLOT,
    )
    parser.add_argument(
        "--gantt", action=argparse.BooleanOptionalAction, default=GENERATE_GANTT_PLOTS
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Report progress as JSON event lines for the TUI dashboard",
    )
    args = parser.parse_args()

    if args.cores < 1:
        parser.error("cores must be at least 1")
    if args.support_cores < 1:
        parser.error("support cores must be at least 1")
    return args


def _report_support_usage(started: float, support_cores: int) -> None:
    """Print how busy the Support cores were, to size --support-cores."""
    usage = psutil.Process().cpu_times()
    elapsed = time.monotonic() - started
    if elapsed <= 0:
        return
    busy = (usage.user + usage.system) / (elapsed * support_cores)
    print(f"Support core utilisation: {busy:.1%}", flush=True)


def _selected_definition_keys() -> list[str] | None:
    raw_value = os.environ.get(SELECTED_DEFINITIONS_ENV)
    if raw_value is None:
        return None

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {SELECTED_DEFINITIONS_ENV} value") from error

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{SELECTED_DEFINITIONS_ENV} must contain a JSON string list")

    return value


def _selected_research_name(definition_keys: list[str] | None) -> str:
    explicit = os.environ.get(SELECTED_RESEARCH_ENV)
    if explicit:
        return explicit

    if definition_keys:
        research_names = {key.split("/", 1)[0] for key in definition_keys if "/" in key}
        if len(research_names) == 1:
            return next(iter(research_names))

    return "all"


def main() -> None:
    global _core_allocation
    args = _parse_args()
    cores = args.cores
    started = time.monotonic()
    _core_allocation = resolve_core_allocation(cores, args.support_cores)

    selected_definition_keys = _selected_definition_keys()
    research_name = _selected_research_name(selected_definition_keys)
    experiments = load_experiments(selected_definition_keys)
    logging_override = os.environ.get(LOGGING_OVERRIDE_ENV)
    if logging_override:
        for experiment in experiments:
            experiment.setdefault("overrides", {})["logging"] = json.loads(
                logging_override
            )
    reporting = {"metrics": args.metrics, "gantt": args.gantt}
    run_specs = build_run_specs(
        experiments, research_name=research_name, reporting=reporting
    )
    progress_dashboard = _prepare_progress_dashboard(run_specs, events=args.events)
    if isinstance(progress_dashboard, EventDashboard):
        progress_dashboard.announce(
            str(_run_spec_config(run_specs[0])["experiment"]["run_root"])
        )
    non_rl_count = sum(not run_spec["is_rl"] for run_spec in run_specs)
    rl_count = len(run_specs) - non_rl_count

    if progress_dashboard is None:
        print("\nSPARS outer runner:")
        print("runs:", len(run_specs))
        print("non-RL runs:", non_rl_count)
        print("RL runs:", rl_count)
        print("non-RL concurrency:", cores)
        print("RL concurrency: 1")
        print("Simulation cores:", [list(c) for c in _core_allocation.simulation])
        print("Support cores:", list(_core_allocation.support))
        print("python:", sys.executable)

    try:
        run_records = asyncio.run(
            execute_run_specs(run_specs, cores, progress_dashboard)
        )
        if args.metrics or args.gantt:
            if progress_dashboard is not None:
                progress_dashboard.set_activity("Plotting")
            generate_all_plots(
                runs=run_records,
                generate_gantt=args.gantt,
                generate_metrics=args.metrics,
            )
    except BaseException as error:
        if progress_dashboard is not None:
            progress_dashboard.set_activity("Failed")
        # The batch still failed overall, but the runs that did succeed keep their plots.
        partial_records = getattr(error, "partial_records", None)
        if partial_records and (args.metrics or args.gantt):
            generate_all_plots(
                runs=partial_records,
                generate_gantt=args.gantt,
                generate_metrics=args.metrics,
            )
        raise
    if progress_dashboard is not None:
        progress_dashboard.set_activity("Completed")
    _report_support_usage(started, args.support_cores)


if __name__ == "__main__":
    main()
