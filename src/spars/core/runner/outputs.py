import csv
from pathlib import Path
from typing import Any


_STEP_LOG_HEADER = (
    "simulation_time",
    "action",
    "reward",
)


def initialize_step_log(step_log_path: str | Path) -> None:
    """Create a step log and write its header."""
    step_log_path = Path(step_log_path)
    step_log_path.parent.mkdir(parents=True, exist_ok=True)

    with step_log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        csv.writer(file).writerow(_STEP_LOG_HEADER)


def append_step_log(
    step_log_path: str | Path,
    action_time: Any,
    action_value: Any,
    reward_value: Any,
) -> None:
    """Append one simulation step to an existing step log."""
    step_log_path = Path(step_log_path)
    step_log_path.parent.mkdir(parents=True, exist_ok=True)

    with step_log_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        csv.writer(file).writerow(
            [
                action_time,
                action_value,
                reward_value,
            ]
        )


def append_learning_log(path: str | Path, diagnostics: dict[str, float]) -> None:
    """Save learner diagnostics without printing them into the dashboard."""
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics))
        if not exists:
            writer.writeheader()
        writer.writerow(diagnostics)
