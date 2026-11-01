"""One effective runtime configuration and its resolved components."""

import json
from pathlib import Path
from typing import Any


def dotted_name(value: Any) -> str:
    return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', type(value).__qualname__)}"


def save_metadata(
    cfg: dict[str, Any], output_path: str | Path, *, model: Any = None
) -> None:
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "config": cfg}
    if cfg["rl"]["enabled"]:
        from spars.core.gym import config

        metadata["resolved"] = {
            "feature_extractor": dotted_name(config.feature_extractor),
            "translator": dotted_name(config.translator),
            "learner": dotted_name(config.learner),
            "reward": dotted_name(config.reward_instance),
        }
    if model is not None:
        metadata["learning_contract"] = model.learning_contract
        metadata["resolved"]["agent"] = dotted_name(model)
        metadata["resolved"]["device"] = str(next(model.parameters()).device)
    temporary = output_dir / "run_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "run_metadata.json")
