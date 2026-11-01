"""Queue of experiment batches started from the dashboard.

Batches run one at a time. Each batch is one executor process with its own
core allocation; the executor reports progress as JSON event lines.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from spars.core.experiments.defaults import PROJECT_ROOT
from spars.core.experiments.discovery import (
    SELECTED_DEFINITIONS_ENV,
    SELECTED_RESEARCH_ENV,
)
from spars.core.experiments.executor import LOGGING_OVERRIDE_ENV
from spars.core.runner.cores import allocate_cores, set_affinity
from spars.core.runner.worker import (
    EVENT_PREFIX,
    PIPE_LIMIT,
    RUNNER_THREAD_LIMIT_ENV,
)


def _stop_tree(pid: int, force: bool = False) -> None:
    """Terminate (or kill) a process and everything it started, on any OS."""
    try:
        root = psutil.Process(pid)
        tree = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for process in tree:
        try:
            process.kill() if force else process.terminate()
        except psutil.NoSuchProcess:
            pass


@dataclass
class Batch:
    number: int
    research: str
    definitions: list[str]
    metrics: bool
    gantt: bool
    log_level: str
    log_file: str
    simulation_cores: int
    support_cores: int
    status: str = "queued"  # queued, running, done, failed, cancelled
    run_root: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    failed_runs: list[str] = field(default_factory=list)
    prevented: list[str] = field(default_factory=list)  # why experiments left out of this batch did not start

    @property
    def name(self) -> str:
        return Path(self.run_root).name if self.run_root else f"batch {self.number}"


class BatchRunner:
    def __init__(
        self,
        available_cpus: set[int],
        on_log: Callable[[str, str], None],
        on_finish: Callable[[Batch], None],
        on_run_failed: Callable[[Batch, str], None] = lambda batch, label: None,
        python_path: str | None = None,
    ) -> None:
        self.python_path = python_path or str(PROJECT_ROOT / "src")
        self.available_cpus = available_cpus
        self.on_log = on_log
        self.on_finish = on_finish
        self.on_run_failed = on_run_failed
        self.batches: list[Batch] = []
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._cancelled: set[int] = set()

    @property
    def current(self) -> Batch | None:
        return next((b for b in self.batches if b.status == "running"), None)

    @property
    def waiting(self) -> list[Batch]:
        return [b for b in self.batches if b.status == "queued"]

    @property
    def busy(self) -> bool:
        return self.current is not None or bool(self.waiting)

    def submit(self, **options: Any) -> Batch:
        batch = Batch(number=len(self.batches) + 1, **options)
        self.batches.append(batch)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())
        return batch

    async def _drain(self) -> None:
        while self.waiting:
            await self._run(self.waiting[0])

    def _command(self, batch: Batch) -> list[str]:
        return [
            sys.executable,
            "-m",
            "spars.core.experiments.executor",
            str(batch.simulation_cores),
            "--support-cores",
            str(batch.support_cores),
            "--metrics" if batch.metrics else "--no-metrics",
            "--gantt" if batch.gantt else "--no-gantt",
            "--events",
        ]

    async def _run(self, batch: Batch) -> None:
        """Run one batch; it always ends done, failed or cancelled, so the queue goes on."""
        batch.status = "running"
        try:
            return_code = await self._execute(batch)
        except Exception as error:  # a batch that cannot start or be read must not hang the queue
            self.on_log("dashboard", f"Batch {batch.number} stopped: {type(error).__name__}: {error}")
            if self._process is not None and self._process.returncode is None:
                _stop_tree(self._process.pid, force=True)
            return_code = None
        self._process = None
        if batch.number in self._cancelled:
            batch.status = "cancelled"
        else:
            batch.status = "done" if return_code == 0 else "failed"
        self.on_finish(batch)

    async def _execute(self, batch: Batch) -> int:
        allocation = allocate_cores(
            batch.simulation_cores, batch.support_cores, self.available_cpus
        )
        # The dashboard is Support work too.
        set_affinity(allocation.support)
        environment = os.environ.copy()
        environment.update(allocation.to_environment())
        environment.update(
            {
                SELECTED_DEFINITIONS_ENV: json.dumps(batch.definitions),
                SELECTED_RESEARCH_ENV: batch.research,
                LOGGING_OVERRIDE_ENV: json.dumps(
                    {"level": batch.log_level, "file": batch.log_file or None}
                ),
                RUNNER_THREAD_LIMIT_ENV: str(batch.simulation_cores),
                "PYTHONPATH": self.python_path,
                "SPARS_ROOT": str(PROJECT_ROOT),
            }
        )

        self._process = await asyncio.create_subprocess_exec(
            *self._command(batch),
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=PIPE_LIMIT,
            # Own session on POSIX, so Ctrl+C in the terminal does not reach the executor.
            start_new_session=True,
        )
        await asyncio.gather(
            self._read_events(batch, self._process.stdout),
            self._read_lines(self._process.stderr),
        )
        return await self._process.wait()

    async def _read_events(self, batch: Batch, stream: Any) -> None:
        while raw := await stream.readline():
            line = raw.decode(errors="replace").rstrip("\r\n")
            if not line.startswith(EVENT_PREFIX):
                self.on_log("executor", line)
                continue
            try:
                event = json.loads(line[len(EVENT_PREFIX) :])
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "state":
                batch.state = event
                for worker in event.get("workers", []):
                    label = worker.get("label", "")
                    if worker.get("status") == "failed" and label not in batch.failed_runs:
                        batch.failed_runs.append(label)
                        self.on_run_failed(batch, label)
            elif kind == "run":
                batch.run_root = event.get("run_root")
            elif kind == "log":
                self.on_log(event.get("label") or "executor", event.get("line", ""))

    async def _read_lines(self, stream: Any) -> None:
        while raw := await stream.readline():
            self.on_log("executor", raw.decode(errors="replace").rstrip("\r\n"))

    def cancel(self, batch: Batch) -> bool:
        """Take a queued batch out of the queue, or stop the running one."""
        if batch.status == "queued":
            batch.status = "cancelled"
            self.on_finish(batch)
            return True
        if batch.status == "running" and batch is self.current and self._process is not None:
            self._cancelled.add(batch.number)
            _stop_tree(self._process.pid)
            return True
        return False

    def cancel_current(self) -> Batch | None:
        batch = self.current
        return batch if batch is not None and self.cancel(batch) else None

    def cancel_all(self) -> None:
        for batch in self.waiting:
            batch.status = "cancelled"
        self.cancel_current()

    async def shutdown(self) -> None:
        self.cancel_all()
        process = self._process
        if process is None:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            _stop_tree(process.pid, force=True)
