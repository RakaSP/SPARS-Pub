import re
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def find_best_epoch_checkpoint(training_output_path: PathLike) -> Path:
    """Return the most recently modified best-epoch checkpoint."""
    training_output_dir = Path(training_output_path)
    candidates: list[Path] = []

    for checkpoint_path in training_output_dir.glob(
        "best_epoch_*/agent_checkpoint.pt"
    ):
        match = re.fullmatch(
            r"best_epoch_(\d+)",
            checkpoint_path.parent.name,
        )

        if match is None or not checkpoint_path.is_file():
            continue

        candidates.append(checkpoint_path)

    if not candidates:
        expected_path = (
            training_output_dir
            / "best_epoch_<epoch>"
            / "agent_checkpoint.pt"
        )
        raise FileNotFoundError(
            "RL training finished, but the best epoch checkpoint "
            "was not found. Expected:\n"
            f"{expected_path}"
        )

    checkpoint_path = max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )

    if len(candidates) > 1:
        print(
            "Found multiple best_epoch directories; using the "
            "most recently modified checkpoint:",
            checkpoint_path,
        )

    return checkpoint_path.resolve()
