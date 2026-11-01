"""Workloads screen: list the converted workloads and convert a CS HUJI SWF trace.

The trace is picked from the .swf / .swf.gz files under data/, or typed. A
typed path may be absolute, relative to data/ (like every path in an
experiment), relative to the project folder, or just a file name under data/.
Quotes around it and Windows back-slashes are fine.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, Static

from spars.core.experiments.defaults import PROJECT_ROOT
from spars.core.simulator.simulator import validate_workload
from spars.core.workloads.swf import convert_swf

from .fields import DATA_ROOT, FIELD_KEYS, Picker, data_files, field_row

WORKLOAD_DIR = DATA_ROOT / "workloads" / "json"
SWF_PATTERNS = ("*.swf", "*.swf.gz")
_MAX_PROCS: dict[tuple[str, float], int | None] = {}
_SIZES: dict[tuple[str, int], tuple[str, str]] = {}  # (path, mtime) -> (jobs, nb_res)


def swf_files() -> list[str]:
    """The SWF traces under data/, as data paths (workloads/cshuji/x.swf)."""
    return data_files("", SWF_PATTERNS)


def max_procs(path: Path) -> int | None:
    """The MaxProcs header of an SWF trace: the size of the machine it came from."""
    key = (str(path), path.stat().st_mtime)
    if key not in _MAX_PROCS:
        import gzip

        opener = gzip.open if path.name.endswith(".gz") else open
        found = None
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if not line.startswith(";"):
                        break
                    match = re.match(r";\s*MaxProcs:\s*(\d+)", line)
                    if match:
                        found = int(match.group(1))
                        break
        except OSError:
            pass
        _MAX_PROCS[key] = found
    return _MAX_PROCS[key]


def swf_details() -> dict[str, tuple[str, str]]:
    details = {}
    for value in swf_files():
        size = max_procs(DATA_ROOT / value)
        details[value] = (value, f"needs a platform of {size} nodes" if size else "")
    return details


def resolve_swf(text: str) -> Path:
    """The trace a typed path means; raises FileNotFoundError naming what was tried."""
    cleaned = text.strip().strip("\"'").strip()
    if not cleaned:
        raise FileNotFoundError("Choose an SWF trace: type part of its name, or press Enter for the list.")
    if os.sep == "/":
        cleaned = cleaned.replace("\\", "/")  # a Windows path typed on Linux/macOS
    path = Path(cleaned).expanduser()
    candidates = [path] if path.is_absolute() else [DATA_ROOT / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    named = [DATA_ROOT / value for value in swf_files() if Path(value).name == path.name]
    if len(named) == 1:
        return named[0]
    available = ", ".join(swf_files()) or "none"
    where = " or ".join(str(c) for c in candidates)
    if len(named) > 1:
        raise FileNotFoundError(f"{path.name} is in several folders; give one of: {', '.join(map(str, named))}")
    raise FileNotFoundError(f"No file {cleaned} (looked for {where}). SWF traces under data/: {available}")


def output_name(source: Path, first: int, last: int | None) -> str:
    stem = re.sub(r"\.swf(\.gz)?$", "", source.name)
    if first == 1 and last is None:
        return f"{stem}.json"
    return f"{stem}-{first}-{last if last is not None else 'last'}.json"  # 'last' only in the preview


def _job_number(text: str, label: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{label} must be a whole number of at least 1.")
    return int(text)


class WorkloadsScreen(Screen):
    """Converted workloads, and a form that converts an SWF trace."""

    BINDINGS = [
        Binding("ctrl+s", "convert", "Convert", priority=True),
        Binding("escape", "app.pop_screen", "Back"),
        *FIELD_KEYS,
    ]
    HELP = {
        "swf_path": "Required. The trace to convert: type part of its name, or press Enter for the list. "
        "A path may be absolute, relative to data/, or just a file name.",
        "swf_first": "First job to keep, counted in submit order from 1. Empty = 1.",
        "swf_last": "Last job to keep (inclusive). Empty = up to the last job of the trace.",
    }

    def compose(self) -> ComposeResult:
        yield Static("Workloads   ·   convert a CS HUJI .swf / .swf.gz trace   ·   Ctrl+S converts", id="title")
        yield DataTable(id="workload-list", show_cursor=False)
        yield Static("Convert an SWF trace", classes="section")
        yield Picker("SWF trace *", swf_files, "", id="swf_path", describe=swf_details)
        yield field_row("First job", "1", id="swf_first", classes="kind-int", placeholder="empty = 1")
        yield field_row("Last job", "", id="swf_last", classes="kind-int", placeholder="empty = last job")
        yield Static(id="swf-output", classes="hint")
        yield Static(id="help", classes="hint")
        yield Static(id="import-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#workload-list", DataTable)
        table.can_focus = False
        table.add_columns("File", "Jobs", "nb_res")
        self.refresh_list()
        self.query_one("#swf_path", Picker).entry.focus()

    def refresh_list(self) -> None:
        # Reading big workloads takes seconds: fill the table from a thread.
        self.run_worker(self._read_list, thread=True, exclusive=True, group="list")

    def _read_list(self) -> None:
        rows = []
        for path in sorted(WORKLOAD_DIR.glob("*.json")) if WORKLOAD_DIR.is_dir() else []:
            key = (str(path), path.stat().st_mtime_ns)
            if key not in _SIZES:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    _SIZES[key] = (str(len(data["jobs"])), str(data["nb_res"]))
                except (OSError, ValueError, KeyError, TypeError):
                    _SIZES[key] = ("invalid", "-")
            rows.append((path.name, *_SIZES[key]))
        self.app.call_from_thread(self._show_list, rows)

    def _show_list(self, rows) -> None:
        try:
            table = self.query_one("#workload-list", DataTable)
        except NoMatches:  # the screen (or the app) closed while the list loaded
            return
        table.clear()
        for row in rows:
            table.add_row(*row)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        for node in (event.widget, *event.widget.ancestors):
            if (node.id or "") in self.HELP:
                self.query_one("#help", Static).update(self.HELP[node.id])
                return

    def status(self, message: str, error: bool = False) -> None:
        self.query_one("#import-status", Static).update(Text(message, style="red" if error else "green"))

    def _form(self) -> tuple[Path, int, int | None]:
        source = resolve_swf(self.query_one("#swf_path", Picker).value)
        first = _job_number(self.query_one("#swf_first", Input).value, "First job") or 1
        last = _job_number(self.query_one("#swf_last", Input).value, "Last job")
        if last is not None and last < first:
            raise ValueError("Last job must not be before the first job.")
        return source, first, last

    def on_input_changed(self, message: Input.Changed) -> None:
        self._show_output()

    def on_picker_changed(self, message: Picker.Changed) -> None:
        self._show_output()

    def _show_output(self) -> None:
        try:
            source, first, last = self._form()
            size = max_procs(source)
            text = f"Writes workloads/json/{output_name(source, first, last)}" + (
                f"; it needs a platform of at least {size} nodes" if size else ""
            )
        except (FileNotFoundError, ValueError, OSError):
            text = ""
        self.query_one("#swf-output", Static).update(text)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        self.focus_next()

    def action_convert(self) -> None:
        try:
            source, first, last = self._form()
        except (FileNotFoundError, ValueError) as error:
            self.status(str(error), error=True)
            return
        self.status("Converting...")
        self.run_worker(lambda: self.convert(source, first, last), thread=True)

    def convert(self, source: Path, first: int, last: int | None) -> None:
        # Runs in a thread: big traces have hundreds of thousands of jobs.
        try:
            workload = convert_swf(source, last, first=first)
            count = len(workload["jobs"])
            if not count:
                raise ValueError(f"{source.name} has no job number {first} or later.")
            # Named after the jobs it really holds: asking for more than the trace has shortens the
            # range. "Whole trace, no suffix" only when no upper bound was asked for at all.
            whole = first == 1 and last is None
            target = WORKLOAD_DIR / output_name(source, first, None if whole else first + count - 1)
            if target.exists():
                raise FileExistsError(f"{target.name} already exists; it is not overwritten.")
            validate_workload(workload, workload["nb_res"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")  # a crash leaves no partial JSON
            temporary.write_text(json.dumps(workload, indent=4) + "\n", encoding="utf-8")
            os.replace(temporary, target)
            message = (
                f"Wrote {len(workload['jobs'])} jobs to workloads/json/{target.name} "
                f"(made for {workload['nb_res']} nodes); choose it as the workload of an experiment."
            )
            self.app.call_from_thread(self.finish, message, False)
        except Exception as error:  # shown to the user, not swallowed
            self.app.call_from_thread(self.finish, f"{type(error).__name__}: {error}", True)

    def finish(self, message: str, error: bool) -> None:
        try:
            self.status(message, error)
        except NoMatches:  # closed while converting: the file is written, tell the app instead
            self.app.notify(message, severity="error" if error else "information", timeout=12)
            return
        self.refresh_list()
