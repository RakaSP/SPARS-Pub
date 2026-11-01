"""Frozen copy of the code that batches run from.

Every run is a new Python process that imports spars from disk when it starts,
so editing a file mid-batch would mix two code versions in one sweep. The
dashboard copies the code once when it starts and runs all batches from that
copy; changed_files tells the dashboard when the live code has moved on.
Experiment definitions stay live (a symlink), because the dashboard edits them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import psutil

from spars.core.experiments.defaults import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "src"
SNAPSHOTS = PROJECT_ROOT / ".spars"
LIVE_PACKAGES = {"experiments", "__pycache__"}


def _link_directory(link: Path, target: Path) -> None:
    """Symlink a directory; Windows without developer mode needs a junction instead."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _remove_snapshot(path: Path) -> None:
    """Delete a snapshot. Its experiments link points at the live definitions:
    remove the link itself first, so nothing can follow it into them."""
    link = path / "src" / "spars" / "experiments"
    try:
        if link.is_symlink():
            link.unlink()
        elif getattr(link, "is_junction", lambda: False)() or (os.name == "nt" and link.exists()):
            os.rmdir(link)  # a junction (Windows): rmdir removes the link, not its contents
    except OSError:
        return  # leave the folder rather than risk the definitions
    shutil.rmtree(path, ignore_errors=True)


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _python_files(source: Path):
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        if relative.parts[1:2] and relative.parts[1] in LIVE_PACKAGES:
            continue
        if "__pycache__" not in relative.parts:
            yield relative


class Snapshot:
    def __init__(self, source: Path = SOURCE, root: Path = SNAPSHOTS) -> None:
        self.source = source
        self.path = root / f"snapshot-{os.getpid()}"
        self._remove_stale(root)
        shutil.rmtree(self.path, ignore_errors=True)
        self.signatures: dict[Path, tuple[int, int]] = {}
        for relative in _python_files(source):
            target = self.path / "src" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
            self.signatures[relative] = _signature(source / relative)
        (self.path / "src" / "spars").mkdir(parents=True, exist_ok=True)
        _link_directory(
            self.path / "src" / "spars" / "experiments",
            (source / "spars" / "experiments").resolve(),
        )

    @property
    def python_path(self) -> str:
        return str(self.path / "src")

    def changed_files(self) -> list[str]:
        """Files edited, added or removed since the snapshot, relative to src/."""
        now = {}
        for relative in _python_files(self.source):
            try:
                now[relative] = _signature(self.source / relative)
            except FileNotFoundError:  # deleted while scanning
                continue
        changed = {p for p in now.keys() | self.signatures.keys() if now.get(p) != self.signatures.get(p)}
        return sorted(p.as_posix() for p in changed)

    def remove(self) -> None:
        _remove_snapshot(self.path)

    @staticmethod
    def _remove_stale(root: Path) -> None:
        """Delete snapshots left by dashboards that crashed."""
        for old in root.glob("snapshot-*"):
            pid = old.name.removeprefix("snapshot-")
            if pid.isdigit() and not psutil.pid_exists(int(pid)):
                _remove_snapshot(old)
