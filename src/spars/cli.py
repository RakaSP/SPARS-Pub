"""SPARS command line: ``spars`` (or ``python -m spars.cli``).

In a terminal it opens the dashboard, where the cores, plots and runs are
chosen. Without a terminal (a script, a server, CI) it runs the sets named
with ``--set`` (or every set with ``--all``) using the settings the dashboard
remembered, which ``--cores``, ``--support-cores``, ``--metrics`` and
``--gantt`` override.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence

from spars.core.runner.cores import allocate_cores, set_affinity
from spars.core.runner.worker import RUNNER_THREAD_LIMIT_ENV, terminate_process


async def run_module(
    module_name: str,
    args: Sequence[object] | None = None,
    thread_limit: int = 1,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Run one Python module in a child process and wait for it to finish."""
    if thread_limit < 1:
        raise ValueError("thread_limit must be at least 1")

    command = [sys.executable, "-m", module_name, *(str(argument) for argument in (args or []))]
    child_environment = os.environ.copy()
    child_environment[RUNNER_THREAD_LIMIT_ENV] = str(thread_limit)
    if env_overrides:
        child_environment.update(env_overrides)

    print(f"Starting: {' '.join(command)}", flush=True)
    process = await asyncio.create_subprocess_exec(*command, env=child_environment)
    try:
        return_code = await process.wait()
    except asyncio.CancelledError:
        await terminate_process(process)
        raise
    if return_code != 0:
        raise RuntimeError(f"{module_name} failed with exit code {return_code}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spars",
        description="SPARS: open the dashboard, or run experiment sets without a terminal.",
    )
    parser.add_argument("--set", action="append", dest="sets", metavar="NAME", default=[],
                        help="Run this experiment set without the dashboard (repeatable)")
    parser.add_argument("--all", action="store_true", help="Run every experiment set without the dashboard")
    parser.add_argument("--cores", type=int, default=None, help="Simulation cores (default: the dashboard's choice)")
    parser.add_argument("--support-cores", type=int, default=None,
                        help="Cores for everything other than simulation (default: the dashboard's choice)")
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=None, help="Make metrics plots")
    parser.add_argument("--gantt", action=argparse.BooleanOptionalAction, default=None, help="Make Gantt plots")
    parser.add_argument("--plot-only", metavar="RESULT_FOLDER", help="Plot saved runs without running anything")
    args = parser.parse_args(argv)
    if args.cores is not None and args.cores < 1:
        parser.error("--cores must be at least 1")
    if args.support_cores is not None and args.support_cores < 1:
        parser.error("--support-cores must be at least 1")
    if args.all and args.sets:
        parser.error("give --set or --all, not both")
    return args


def _settings(args: argparse.Namespace):
    from spars.core.dashboard.settings import DashboardSettings

    settings = DashboardSettings.load()
    for name, value in (
        ("simulation_cores", args.cores),
        ("support_cores", args.support_cores),
        ("metrics", args.metrics),
        ("gantt", args.gantt),
    ):
        if value is not None:
            setattr(settings, name, value)
    return settings


def _selection(args: argparse.Namespace) -> tuple[str, list[str], list[str]]:
    """(result name, definition keys to run, why the others were left out) of the sets to run."""
    from spars.core.experiments import store
    from spars.core.experiments.discovery import discover_experiment_definitions, discover_research_names

    names = discover_research_names()
    chosen = names if args.all else args.sets
    unknown = [name for name in chosen if name not in names]
    if unknown:
        raise ValueError(f"Unknown set(s) {', '.join(unknown)}. Sets: {', '.join(names) or 'none yet'}")
    broken: list[str] = []
    definitions = [d for name in chosen for d in discover_experiment_definitions(name, broken)]
    blocked = store.blocked_definitions(definitions)
    prevented = broken + [problem for problems in blocked.values() for problem in problems]
    keys = [d.key for d in definitions if d.key not in blocked]
    if not keys:
        raise ValueError(
            f"No experiment can run in {', '.join(chosen) or 'any set'}"
            + ("".join(f"\n  {line}" for line in prevented) or ": no experiment definitions")
        )
    return (chosen[0] if len(chosen) == 1 else "all"), keys, prevented


def _warn_prevented(prevented: list[str]) -> None:
    if prevented:
        print(f"WARNING: {len(prevented)} problem(s) kept experiments from running:", file=sys.stderr, flush=True)
        for line in prevented:
            print(f"WARNING:   {line}", file=sys.stderr, flush=True)


async def _run_sets(args: argparse.Namespace) -> None:
    from spars.core.experiments.discovery import SELECTED_DEFINITIONS_ENV, SELECTED_RESEARCH_ENV
    from spars.core.experiments.executor import LOGGING_OVERRIDE_ENV

    settings = _settings(args)
    research, keys, prevented = _selection(args)
    _warn_prevented(prevented)
    allocation = allocate_cores(settings.simulation_cores, settings.support_cores)
    set_affinity(allocation.support)  # this process is Support work
    print(f"Simulation cores: {[list(core) for core in allocation.simulation]}", flush=True)
    print(f"Support cores: {list(allocation.support)}", flush=True)
    await run_module(
        "spars.core.experiments.executor",
        [
            settings.simulation_cores,
            "--support-cores", settings.support_cores,
            "--metrics" if settings.metrics else "--no-metrics",
            "--gantt" if settings.gantt else "--no-gantt",
        ],
        thread_limit=settings.simulation_cores,
        env_overrides={
            **allocation.to_environment(),
            SELECTED_RESEARCH_ENV: research,
            SELECTED_DEFINITIONS_ENV: json.dumps(keys),
            # Harmonizes every selected experiment to one log level, same as the
            # dashboard: mixed per-file levels would otherwise fail at startup.
            LOGGING_OVERRIDE_ENV: json.dumps({"level": settings.log_level, "file": settings.log_file or None}),
        },
    )
    print("All runs finished.", flush=True)
    _warn_prevented(prevented)


NO_TERMINAL = """No terminal found, so the dashboard cannot open.
Run sets without it:  spars --set NAME   (or --all for every set)
Sets: {sets}
In Git Bash on Windows, open the dashboard with:  winpty python -m spars.cli"""


def _check_project_root() -> None:
    """SPARS runs from its repository folder (data/, results/, the experiment sets)."""
    from spars.core.experiments.defaults import PROJECT_ROOT

    if not (PROJECT_ROOT / "data").is_dir():
        raise FileNotFoundError(
            f"No data/ folder in {PROJECT_ROOT}. SPARS runs from its repository folder: "
            "install it there with python install.py or pip install -e <repository>, "
            "or set SPARS_ROOT to the repository folder."
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        _check_project_root()
        if args.plot_only is not None:
            from spars.core.analysis.automatic import generate_all_plots, load_saved_runs

            generate_all_plots(
                load_saved_runs(args.plot_only),
                generate_metrics=True if args.metrics is None else args.metrics,
                generate_gantt=bool(args.gantt),
            )
            return 0
        if args.sets or args.all:
            asyncio.run(_run_sets(args))
            return 0
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            from spars.core.experiments.discovery import discover_research_names

            print(NO_TERMINAL.format(sets=", ".join(discover_research_names()) or "none yet"), file=sys.stderr)
            return 2
        from spars.core.dashboard.app import run_dashboard

        return run_dashboard(_settings(args))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
