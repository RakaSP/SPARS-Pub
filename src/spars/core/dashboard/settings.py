"""Dashboard choices remembered between sessions.

They live in .spars/cli_settings.json (git-ignored), never in the experiment
definitions or defaults.py, so published experiments do not change when an
option is toggled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from spars.core.experiments.defaults import (
    BASE_CONFIG,
    DEFAULT_CORES,
    GENERATE_GANTT_PLOTS,
    GENERATE_METRICS_PLOT,
    PROJECT_ROOT,
)
from spars.core.logging import _ALLOWED_LEVELS
from spars.core.runner.cores import DEFAULT_SUPPORT_CORES


SETTINGS_PATH = PROJECT_ROOT / ".spars" / "cli_settings.json"
LOG_LEVELS = tuple(_ALLOWED_LEVELS)


@dataclass
class DashboardSettings:
    metrics: bool = GENERATE_METRICS_PLOT
    gantt: bool = GENERATE_GANTT_PLOTS
    log_level: str = str(BASE_CONFIG["logging"]["level"])
    log_file: str = str(BASE_CONFIG["logging"]["file"])
    simulation_cores: int = DEFAULT_CORES
    support_cores: int = DEFAULT_SUPPORT_CORES
    research: str | None = None
    selected: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "DashboardSettings":
        path = path or SETTINGS_PATH
        settings = cls()
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return settings
        if not isinstance(stored, dict):
            return settings
        for item in fields(cls):
            if item.name not in stored:
                continue
            value = stored[item.name]
            default = getattr(settings, item.name)
            # Ignore values of the wrong type, e.g. from an older version.
            if default is None or isinstance(value, type(default)):
                setattr(settings, item.name, value)
        if settings.log_file == "results/simulation.log":  # the old default, now inside each batch folder
            settings.log_file = cls.log_file
        if settings.log_level not in LOG_LEVELS:
            settings.log_level = cls.log_level
        settings.simulation_cores = max(1, int(settings.simulation_cores))
        settings.support_cores = max(1, int(settings.support_cores))
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def reset(self) -> None:
        """Restore the option defaults; the run selection is kept."""
        defaults = DashboardSettings()
        for name in (
            "metrics",
            "gantt",
            "log_level",
            "log_file",
            "simulation_cores",
            "support_cores",
        ):
            setattr(self, name, getattr(defaults, name))
