from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from spars.core.logging import is_quiet_level
from spars.core.progress import PROGRESS_ENV, PROGRESS_PREFIX
from spars.core.runner.cores import set_affinity


RUNNER_THREAD_LIMIT_ENV = "RUNNER_THREAD_LIMIT"
# Prefix of the JSON event lines EventDashboard writes for the TUI dashboard.
EVENT_PREFIX = "__SPARS_EVENT__:"
# Lines from a child process may be long (TRACE logs); asyncio's default limit is 64 KiB.
PIPE_LIMIT = 16 * 1024 * 1024
_PROGRESS_BAR_WIDTH = 16
_SIMULATION_WIDTH = 28


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))

    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes:02d}:{secs:02d}"

    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


@dataclass
class _WorkerProgress:
    label: str = "-"
    group: str | None = None
    completed: int = 0
    total: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    phase: str = "run"
    epoch: int = 0
    epochs: int = 0
    status: str = "running"
    config_label: str = ""


class ProgressDashboard:
    """Render one stable progress row per active worker slot."""

    def __init__(
        self,
        *,
        total_simulations: int,
        experiment_totals: Mapping[str, int],
    ) -> None:
        self.total_simulations = max(0, int(total_simulations))
        self.experiment_totals = {
            str(name): int(total) for name, total in experiment_totals.items()
        }
        self.experiment_completed = {name: 0 for name in self.experiment_totals}
        self.completed_simulations = 0
        self.failed_simulations = 0
        self.started_at = time.monotonic()
        self.worker_count = 0
        self.workers: dict[int, _WorkerProgress] = {}
        self.activity = "Running"
        self._last_render = 0.0
        self._tty = sys.stdout.isatty()
        self._rendered_line_count = 0
        self._last_non_tty_summary: str | None = None

    def set_activity(self, activity: str) -> None:
        self.activity = activity
        self.render(force=True)

    def configure_workers(self, count: int) -> None:
        self.worker_count = max(1, int(count))
        self.workers = {
            slot: _WorkerProgress() for slot in range(1, self.worker_count + 1)
        }

    def start(self, slot: int, label: str, group: str) -> None:
        self.workers[slot] = _WorkerProgress(
            label=label,
            group=group,
            started_at=time.monotonic(),
            phase="loading",
            status="loading",
        )
        self.render(force=True)

    def update(
        self,
        slot: int,
        completed: int,
        total: int,
        *,
        phase: str | None = None,
        epoch: int | None = None,
        epochs: int | None = None,
        status: str | None = None,
        config_label: str | None = None,
    ) -> None:
        worker = self.workers[slot]
        previous_stage = (
            worker.phase,
            worker.epoch,
            worker.status,
            worker.config_label,
        )
        worker.completed = max(0, int(completed))
        worker.total = max(0, int(total))
        if phase is not None:
            worker.phase = phase
        if epoch is not None:
            worker.epoch = max(0, int(epoch))
        if epochs is not None:
            worker.epochs = max(0, int(epochs))
        if status is not None:
            worker.status = status
        if config_label is not None:
            worker.config_label = config_label
        changed = previous_stage != (
            worker.phase,
            worker.epoch,
            worker.status,
            worker.config_label,
        )
        self.render(force=changed)

    def finish(self, slot: int, *, success: bool) -> None:
        worker = self.workers[slot]
        worker.finished_at = time.monotonic()
        worker.status = "done" if success else "failed"

        if success:
            if worker.total > 0:
                worker.completed = worker.total
            self.completed_simulations += 1

            if worker.group in self.experiment_completed:
                self.experiment_completed[worker.group] += 1

        if not success:
            self.failed_simulations += 1
        self.render(force=True)

    def _completed_experiments(self) -> int:
        return sum(
            self.experiment_completed[name] >= expected
            for name, expected in self.experiment_totals.items()
        )

    def _progress_field(self, worker: _WorkerProgress) -> str:
        if worker.status == "done":
            fraction = 1.0
        elif worker.total <= 0 and not (worker.phase == "train" and worker.epochs > 0):
            return "-"
        else:
            fraction = (
                min(1.0, max(0.0, worker.completed / worker.total))
                if worker.total
                else 0.0
            )
            if worker.phase == "train" and worker.epochs > 0:
                fraction = (max(0, worker.epoch - 1) + fraction) / worker.epochs
            fraction = min(1.0, max(0.0, fraction))
        filled = round(fraction * _PROGRESS_BAR_WIDTH)
        bar = "█" * filled + "░" * (_PROGRESS_BAR_WIDTH - filled)
        return f"{bar} {fraction * 100:5.1f}%"

    def _jobs_field(self, worker: _WorkerProgress) -> str:
        if worker.total <= 0:
            return "-"
        return f"{worker.completed}/{worker.total}"

    def _elapsed_field(self, worker: _WorkerProgress) -> str:
        if worker.started_at is None:
            return "-"
        end = worker.finished_at or time.monotonic()
        return _format_elapsed(end - worker.started_at)

    def _lines(self) -> list[str]:
        elapsed = _format_elapsed(time.monotonic() - self.started_at)
        experiment_total = len(self.experiment_totals)
        active = sum(
            w.started_at is not None and w.finished_at is None
            for w in self.workers.values()
        )
        queued = max(
            0,
            self.total_simulations
            - self.completed_simulations
            - self.failed_simulations
            - active,
        )
        columns = shutil.get_terminal_size((132, 24)).columns if self._tty else 132
        failed = (
            f" | {self.failed_simulations} failed" if self.failed_simulations else ""
        )
        lines = [
            (
                f"{self.completed_simulations}/{self.total_simulations} simulations"
                f" | {self._completed_experiments()}/{experiment_total} experiments"
                f" | {queued} queued{failed} | {self.activity} | {elapsed}"
            ),
            "",
            (
                f"{'Worker':<6}  "
                f"{'Simulation':<{_SIMULATION_WIDTH}}  "
                f"{'Phase':<15}  "
                f"{'Epoch':<7}  "
                f"{'Overall progress':<24}  "
                f"{'Current jobs':<12}  "
                "Elapsed"
            ),
        ]

        compact = columns < 128
        if compact:
            lines = [
                lines[0],
                "",
                "Worker / Simulation | Phase | Epoch | Overall | Current jobs | Elapsed",
            ]

        for slot in range(1, self.worker_count + 1):
            worker = self.workers.get(slot, _WorkerProgress())
            label = _truncate(worker.label, _SIMULATION_WIDTH)
            phase = worker.phase.capitalize() if worker.started_at is not None else "-"
            if worker.status in {"loading", "saving", "finalizing", "done", "failed"}:
                phase = worker.status.capitalize()
            epoch = f"{worker.epoch}/{worker.epochs}" if worker.epochs else "-"
            if compact:
                lines.append(f"W{slot}  {worker.label}")
                percentage = self._progress_field(worker).rsplit(" ", 1)[-1]
                lines.append(
                    f"    {phase} | {epoch} | {percentage} overall"
                    f" | {self._jobs_field(worker)} jobs | {self._elapsed_field(worker)}"
                )
                continue
            lines.append(
                f"{f'W{slot}':<6}  "
                f"{label:<{_SIMULATION_WIDTH}}  "
                f"{phase:<15}  "
                f"{epoch:<7}  "
                f"{self._progress_field(worker):<24}  "
                f"{self._jobs_field(worker):<12}  "
                f"{self._elapsed_field(worker)}"
            )

        # Reserve the last column: writing it can auto-wrap and break cursor anchoring.
        return [_truncate(line, max(1, columns - 1)) for line in lines]

    def render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.10:
            return
        self._last_render = now

        lines = self._lines()
        if self._tty:
            if self._rendered_line_count:
                sys.stdout.write(f"\033[{self._rendered_line_count}F")

            line_count = max(len(lines), self._rendered_line_count)
            for index in range(line_count):
                line = lines[index] if index < len(lines) else ""
                sys.stdout.write("\033[2K" + line + "\n")

            self._rendered_line_count = line_count
            sys.stdout.flush()
        elif force:
            active = [line.strip() for line in lines[3:] if line.strip()]
            summary = "\n".join([lines[0], *active])
            if summary != self._last_non_tty_summary:
                print(summary, flush=True)
                self._last_non_tty_summary = summary

    def print_messages(self, messages: Sequence[str]) -> None:
        clean_messages = [message for message in messages if message]
        if not clean_messages:
            return

        # Messages are exceptional in QUIET mode. Print them below the current
        # dashboard and start a fresh cursor anchor for the next refresh.
        stream = sys.stderr
        for message in clean_messages:
            print(message, file=stream, flush=True)
        self._rendered_line_count = 0
        self.render(force=True)

    def log(self, label: str, line: str) -> None:
        """A worker's log line; the terminal dashboard keeps only errors."""


class EventDashboard(ProgressDashboard):
    """Report progress as JSON event lines for the TUI dashboard.

    The TUI process draws the screen; this only describes the state, so every
    log level can run with a dashboard.
    """

    def __init__(self, *, total_simulations: int, experiment_totals: Mapping[str, int]) -> None:
        super().__init__(
            total_simulations=total_simulations, experiment_totals=experiment_totals
        )
        self._tty = False

    def _emit(self, event: dict[str, Any]) -> None:
        sys.stdout.write(EVENT_PREFIX + json.dumps(event, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def announce(self, run_root: str) -> None:
        self._emit({"type": "run", "run_root": run_root})

    def render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.25:
            return
        self._last_render = now
        workers = []
        for slot in range(1, self.worker_count + 1):
            worker = self.workers.get(slot, _WorkerProgress())
            percentage = self._progress_field(worker).rsplit(" ", 1)[-1]
            workers.append(
                {
                    "slot": slot,
                    "label": worker.label,
                    "phase": worker.phase if worker.started_at is not None else "-",
                    "status": worker.status,
                    "epoch": worker.epoch,
                    "epochs": worker.epochs,
                    "progress": percentage,
                    "jobs": self._jobs_field(worker),
                    "elapsed": self._elapsed_field(worker),
                }
            )
        self._emit(
            {
                "type": "state",
                "activity": self.activity,
                "completed": self.completed_simulations,
                "failed": self.failed_simulations,
                "total": self.total_simulations,
                "elapsed": _format_elapsed(now - self.started_at),
                "workers": workers,
            }
        )

    def print_messages(self, messages: Sequence[str]) -> None:
        """Errors were already sent line by line through log()."""

    def log(self, label: str, line: str) -> None:
        if line:
            self._emit({"type": "log", "label": label, "line": line})


def resolve_thread_limit(requested_limit: int | None) -> int:
    """Determine the number of scripts allowed to run concurrently."""
    environment_value = os.environ.get(RUNNER_THREAD_LIMIT_ENV)

    if environment_value is not None:
        try:
            thread_limit = int(environment_value)
        except ValueError as error:
            raise ValueError(
                f"{RUNNER_THREAD_LIMIT_ENV} must be an integer, "
                f"not {environment_value!r}"
            ) from error
    elif requested_limit is not None:
        thread_limit = requested_limit
    else:
        thread_limit = 1

    if thread_limit < 1:
        raise ValueError("THREAD_LIMIT must be at least 1")

    return thread_limit


async def terminate_process(
    process: asyncio.subprocess.Process,
) -> None:
    """Terminate a subprocess cleanly, killing it if necessary."""
    if process.returncode is not None:
        return

    try:
        process.terminate()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return

        await process.wait()


def _parse_progress_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
        result: dict[str, Any] = {
            "completed": int(payload["completed"]),
            "total": int(payload["total"]),
        }
        for key in ("epoch", "epochs"):
            if key in payload:
                result[key] = int(payload[key])
        for key in ("phase", "status", "config_label"):
            if key in payload:
                if not isinstance(payload[key], str):
                    return None
                result[key] = payload[key]
        return result
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return None


class _LogSink:
    """Write one worker's output lines from a Support core.

    The simulation only writes lines into its pipes; this sink, running in
    the parent, echoes them to the terminal and appends them to the log file.
    """

    def __init__(self, log_file: str | Path | None, label: str) -> None:
        self.label = label
        self._file = None
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, line: str, echo: Any = None) -> None:
        if echo is not None:
            print(line, file=echo, flush=True)
        if self._file is not None and line:
            self._file.write(f"[{self.label}] {line}\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


async def _consume_stdout(
    stream: asyncio.StreamReader,
    *,
    dashboard: ProgressDashboard | None,
    slot: int,
    captured: deque[str],
    sink: _LogSink,
) -> None:
    while True:
        raw_line = await stream.readline()
        if not raw_line:
            return

        line = raw_line.decode(errors="replace").rstrip("\r\n")
        if dashboard is None:
            sink.write(line, sys.stdout)
            continue

        progress = _parse_progress_line(line)
        if progress is None:
            captured.append(line)
            sink.write(line)
            dashboard.log(sink.label, line)
            continue

        dashboard.update(slot, **progress)


async def _consume_stderr(
    stream: asyncio.StreamReader,
    captured: deque[str],
    sink: _LogSink,
    dashboard: ProgressDashboard | None,
) -> None:
    while True:
        raw_line = await stream.readline()
        if not raw_line:
            return
        line = raw_line.decode(errors="replace").rstrip("\r\n")
        captured.append(line)
        if dashboard is None:
            sink.write(line, sys.stderr)
        else:
            sink.write(line)
            dashboard.log(sink.label, line)


async def _dashboard_heartbeat(dashboard: ProgressDashboard) -> None:
    while True:
        await asyncio.sleep(0.5)
        dashboard.render()


def _pin_to(cpus: Sequence[int] | None) -> Any:
    """Return a pre-exec hook that pins the new process before it runs (Linux only)."""
    if cpus is None or not hasattr(os, "sched_setaffinity"):
        return None
    selected = set(cpus)
    return lambda: os.sched_setaffinity(0, selected)


async def run_script_async(
    semaphore: asyncio.Semaphore,
    script_path: str | Path,
    args: Sequence[object] | None,
    iterator: int,
    *,
    slot_queue: asyncio.PriorityQueue[int],
    simulation_cpus: Sequence[Sequence[int]] | None = None,
    dashboard: ProgressDashboard | None = None,
    progress_label: str | None = None,
    progress_group: str | None = None,
    log_file: str | Path | None = None,
) -> list[str]:
    """Run one Python script instance on its own Simulation core."""
    script = Path(script_path)

    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script.resolve()}")

    command = [
        sys.executable,
        str(script),
        *(str(argument) for argument in (args or [])),
    ]

    if dashboard is not None and (progress_label is None or progress_group is None):
        raise RuntimeError("Quiet progress runner is missing dashboard metadata")

    async with semaphore:
        # The lowest free slot is taken first, so the fastest cores are reused
        # before slower ones.
        slot = await slot_queue.get()
        cpus = simulation_cpus[slot - 1] if simulation_cpus is not None else None
        environment = os.environ.copy()
        # One core per simulation: keep numeric libraries single-threaded.
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if dashboard is None:
            print(
                f"\nStarting instance {iterator} on CPU {list(cpus or [])}: {script}",
                flush=True,
            )
            print(f"Command: {' '.join(command)}", flush=True)
        else:
            dashboard.start(slot, progress_label, progress_group)
            environment[PROGRESS_ENV] = "1"
        sink = _LogSink(log_file, progress_label or f"instance {iterator}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=PIPE_LIMIT,
                preexec_fn=_pin_to(cpus),
            )
            if cpus is not None and not hasattr(os, "sched_setaffinity"):
                set_affinity(cpus, process.pid)
        except OSError:
            if dashboard is not None:
                dashboard.finish(slot, success=False)
            slot_queue.put_nowait(slot)
            sink.close()
            raise
        assert process.stdout is not None
        assert process.stderr is not None

        captured_stdout: deque[str] = deque(maxlen=50)
        captured_stderr: deque[str] = deque(maxlen=50)
        stdout_task = asyncio.create_task(
            _consume_stdout(
                process.stdout,
                dashboard=dashboard,
                slot=slot,
                captured=captured_stdout,
                sink=sink,
            )
        )
        stderr_task = asyncio.create_task(
            _consume_stderr(process.stderr, captured_stderr, sink, dashboard)
        )

        try:
            return_code = await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            await terminate_process(process)
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(
                stdout_task,
                stderr_task,
                return_exceptions=True,
            )
            if dashboard is not None:
                dashboard.finish(slot, success=False)
            slot_queue.put_nowait(slot)
            raise
        finally:
            sink.close()

        success = return_code == 0
        if dashboard is not None:
            dashboard.finish(slot, success=success)
        slot_queue.put_nowait(slot)

        if dashboard is None:
            if not success:
                raise RuntimeError(
                    f"{script} instance {iterator} failed with exit code {return_code}"
                )
            print(f"Finished instance {iterator}: {script}", flush=True)
            return []

        if not success:
            output_tail = [
                *list(captured_stdout)[-20:],
                *list(captured_stderr)[-20:],
            ]
            details = "\n".join(line for line in output_tail if line)
            suffix = f"\n{details}" if details else ""
            raise RuntimeError(
                f"{script} instance {iterator} failed "
                f"with exit code {return_code}{suffix}"
            )

        return list(captured_stderr)


class WorkerBatchFailed(RuntimeError):
    """Some runs in the batch failed; ``failed_indices`` are 0-based positions in ``args_`` order."""

    def __init__(self, message: str, failed_indices: set[int]) -> None:
        super().__init__(message)
        self.failed_indices = failed_indices


async def runner_main(
    script_path: str | Path,
    args_: Sequence[Sequence[object]],
    THREAD_LIMIT: int | None = None,
    *,
    simulation_cpus: Sequence[Sequence[int]] | None = None,
    progress_dashboard: ProgressDashboard | None = None,
    progress_labels: Sequence[str] | None = None,
    progress_groups: Sequence[str] | None = None,
    log_files: Sequence[str | Path | None] | None = None,
) -> None:
    """Run multiple instances of the same script, one per Simulation core."""
    thread_limit = resolve_thread_limit(THREAD_LIMIT)
    quiet = progress_dashboard is not None
    if simulation_cpus is not None and len(simulation_cpus) < thread_limit:
        raise ValueError(
            f"{thread_limit} concurrent instances need {thread_limit} "
            f"Simulation cores, but only {len(simulation_cpus)} were allocated"
        )
    if log_files is not None and len(log_files) != len(args_):
        raise ValueError("Log files must match the worker count")

    if not quiet:
        print(
            f"\nRunner configuration for {script_path}:",
            flush=True,
        )
        print(
            f"  Instances: {len(args_)}",
            flush=True,
        )
        print(
            f"  Concurrent instances: {thread_limit}",
            flush=True,
        )

    if quiet:
        if progress_labels is None or progress_groups is None:
            raise ValueError("Quiet progress requires labels and groups")
        if len(progress_labels) != len(args_) or len(progress_groups) != len(args_):
            raise ValueError("Quiet progress metadata must match worker count")

    semaphore = asyncio.Semaphore(thread_limit)
    active_slots = min(thread_limit, len(args_))
    slot_queue: asyncio.PriorityQueue[int] = asyncio.PriorityQueue()
    for slot in range(1, active_slots + 1):
        slot_queue.put_nowait(slot)
    if quiet:
        progress_dashboard.configure_workers(active_slots)

    tasks = [
        asyncio.create_task(
            run_script_async(
                semaphore=semaphore,
                script_path=script_path,
                args=args,
                iterator=iterator,
                slot_queue=slot_queue,
                simulation_cpus=simulation_cpus,
                dashboard=progress_dashboard,
                progress_label=(
                    progress_labels[iterator - 1]
                    if progress_labels is not None
                    else None
                ),
                progress_group=(
                    progress_groups[iterator - 1]
                    if progress_groups is not None
                    else None
                ),
                log_file=log_files[iterator - 1] if log_files is not None else None,
            )
        )
        for iterator, args in enumerate(args_, start=1)
    ]

    heartbeat_task = (
        asyncio.create_task(_dashboard_heartbeat(progress_dashboard))
        if progress_dashboard is not None
        else None
    )

    try:
        # A failed run must not stop the others: collect every outcome.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    failures = [o for o in outcomes if isinstance(o, BaseException)]
    if quiet:
        errors = [
            line
            for group in outcomes
            if not isinstance(group, BaseException)
            for line in group
            if "[ERROR]" in line
        ]
        if errors:
            progress_dashboard.print_messages(errors)
    if failures:
        first_lines = "\n".join(str(failure).splitlines()[0] for failure in failures)
        failed_indices = {index for index, outcome in enumerate(outcomes) if isinstance(outcome, BaseException)}
        raise WorkerBatchFailed(f"{len(failures)} of {len(tasks)} runs failed:\n{first_lines}", failed_indices)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <config.yaml>")

    config_path = Path(sys.argv[1])

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise TypeError(f"Invalid configuration in {config_path}")

    quiet = is_quiet_level(cfg.get("logging", {}).get("level", "INFO"))
    if not quiet:
        print("\nRunning configuration:")
        print("config:", config_path)
        print("scheduler:", cfg["run"].get("scheduler", {}).get("name"))
        print("power manager:", cfg["run"].get("power_management", {}).get("name"))
        print("workload:", cfg["paths"]["workload"])
        print("output:", cfg["paths"]["output"])
        print("run parameters:", cfg["run"])

    from spars.core.runner.run import run_with_config

    run_with_config(cfg)


if __name__ == "__main__":
    main()
