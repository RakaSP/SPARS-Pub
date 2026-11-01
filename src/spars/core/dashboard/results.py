"""Read the mean waiting time, energy waste and time per node state of every run folder."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from spars.core.experiments.defaults import RESULTS_ROOT


_RUN_FOLDER = re.compile(r".+-\d{6}$")


@dataclass(frozen=True)
class RunFolder:
    path: Path
    modified: datetime
    metrics_files: int


@dataclass(frozen=True)
class MetricsRow:
    experiment: str
    workload: str
    variant: str
    mean_wait: float | None
    waste_gj: float | None
    # Node-seconds spent in each state, summed over all nodes (metrics.csv total_*).
    state_seconds: tuple[float | None, ...] = ()
    # Jobs that never ran (rows of unfinished_jobs_log.csv) and, for a run the
    # simulator stopped early, why (stalled.json).
    unfinished_jobs: int | None = None
    stalled: str | None = None


STATE_COLUMNS = (
    ("Idle", "total_active_idle"),
    ("Computing", "total_active_compute"),
    ("Switching off", "total_switching_off"),
    ("Switching on", "total_switching_on"),
    ("Sleeping", "total_sleeping"),
)


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def list_run_folders(root: Path = RESULTS_ROOT) -> list[RunFolder]:
    """Run folders such as <Set>-000061, newest first."""
    if not root.is_dir():
        return []
    folders = []
    for path in root.iterdir():
        if path.is_dir() and _RUN_FOLDER.fullmatch(path.name):
            folders.append(
                RunFolder(
                    path=path,
                    modified=datetime.fromtimestamp(path.stat().st_mtime),
                    metrics_files=sum(1 for _ in path.rglob("metrics.csv")),
                )
            )
    return sorted(folders, key=lambda folder: _natural_key(folder.path.name), reverse=True)


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _unfinished_jobs(folder: Path) -> int | None:
    """Jobs that did not finish: from stalled.json when the run stopped early (it counts jobs never submitted), else the rows of unfinished_jobs_log.csv."""
    try:
        return int(json.loads((folder / "stalled.json").read_text(encoding="utf-8"))["unfinished_jobs"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        with open(folder / "unfinished_jobs_log.csv", encoding="utf-8") as file:
            return max(0, sum(1 for _ in file) - 1)  # header; a run with none has an empty file
    except OSError:
        return None


def _stall_reason(folder: Path) -> str | None:
    try:
        return str(json.loads((folder / "stalled.json").read_text(encoding="utf-8"))["reason"])
    except (OSError, ValueError, KeyError):
        return None


def read_metrics(run_folder: Path) -> list[MetricsRow]:
    """One row per metrics.csv in the run, labelled by its relative path."""
    rows = []
    for path in sorted(
        run_folder.rglob("metrics.csv"),
        key=lambda item: _natural_key(str(item.relative_to(run_folder))),
    ):
        parts = path.relative_to(run_folder).parent.parts
        try:
            with open(path, newline="", encoding="utf-8") as file:
                record = next(csv.DictReader(file), {})
        except OSError:
            record = {}
        waste = _number(record.get("total_energy_waste"))
        rows.append(
            MetricsRow(
                experiment=parts[0] if parts else "",
                # experiment / platform / workload / variant...
                workload=parts[2] if len(parts) > 2 else "",
                variant="/".join(parts[3:]),
                mean_wait=_number(record.get("mean_waiting_time")),
                waste_gj=waste / 1e9 if waste is not None else None,
                state_seconds=tuple(_number(record.get(key)) for _, key in STATE_COLUMNS),
                unfinished_jobs=_unfinished_jobs(path.parent),
                stalled=_stall_reason(path.parent),
            )
        )
    return rows
