from __future__ import annotations

import json
import os
from typing import Any

PROGRESS_ENV = "SPARS_PROGRESS_PROTOCOL"
PROGRESS_PREFIX = "__SPARS_PROGRESS__:"
_context: dict[str, Any] = {}
_last_counts = (0, 0)


def progress_enabled() -> bool:
    return os.getenv(PROGRESS_ENV, "0") == "1"


def set_progress_context(
    phase: str, *, epoch: int = 0, epochs: int = 0, config_label: str | None = None
) -> None:
    """Start a phase/epoch; epoch numbers shown to users are one-based."""
    global _context
    if phase not in {"train", "test", "run"}:
        raise ValueError(f"Invalid progress phase: {phase}")
    if phase == "train" and not (epoch == epochs == 0 or 1 <= epoch <= epochs):
        raise ValueError("Training progress requires 1 <= epoch <= epochs")
    previous_label = _context.get("config_label", "")
    _context = {
        "phase": phase,
        "epoch": epoch,
        "epochs": epochs,
        "status": "running",
        "config_label": previous_label if config_label is None else config_label,
    }
    emit_simulation_progress(0, 0)


def set_progress_status(status: str) -> None:
    if status not in {"loading", "running", "saving", "finalizing"}:
        raise ValueError(f"Invalid progress status: {status}")
    _context["status"] = status
    emit_simulation_progress(*_last_counts)


def emit_simulation_progress(completed: int, total: int) -> None:
    global _last_counts
    _last_counts = (max(0, int(completed)), max(0, int(total)))
    if not progress_enabled():
        return
    payload: dict[str, Any] = {
        **_context,
        "completed": _last_counts[0],
        "total": _last_counts[1],
    }
    print(PROGRESS_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)
