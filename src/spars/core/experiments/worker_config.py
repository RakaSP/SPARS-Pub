from pathlib import Path
import os
import tempfile
from typing import Any

import yaml

from spars.core.registry import get_run_timeout
from spars.core.logging import is_quiet_level

from .naming import build_reward_weight_tags


def write_generated_config(
    cfg: dict[str, Any],
    config_path: str | Path,
) -> None:
    config_path = Path(config_path)
    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Install a complete YAML file; interruption must not leave a partial config.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            yaml.safe_dump(cfg, file, sort_keys=False)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(config_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def verify_generated_config(
    run_spec: dict[str, Any],
) -> Path:
    config_path = Path(run_spec["config_path"]).resolve()

    # The run spec owns the configuration. Recreate a missing generated artifact
    # at dispatch, including after a long preceding training run.
    if not config_path.exists():
        write_generated_config(run_spec["cfg"], config_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        written_cfg = yaml.safe_load(file)

    if not isinstance(written_cfg, dict):
        raise TypeError(f"Generated config must contain a dictionary: {config_path}")

    expected_cfg = run_spec["cfg"]
    if written_cfg != expected_cfg:
        changed_sections = sorted(
            key
            for key in written_cfg.keys() | expected_cfg.keys()
            if written_cfg.get(key) != expected_cfg.get(key)
        )
        raise RuntimeError(
            "Generated config does not match the run specification: "
            + ", ".join(changed_sections)
            + f"\n{config_path}"
        )

    quiet = is_quiet_level(written_cfg.get("logging", {}).get("level", "INFO"))
    if not quiet:
        print("\nVerified worker config:")
        print("config:", config_path)
        print("workload:", written_cfg["paths"]["workload"])
        print("output:", written_cfg["paths"]["output"])
        print("rl.enabled:", written_cfg["rl"]["enabled"])
        print("rl.learn:", written_cfg["rl"]["learn"])
        print("checkpoint:", written_cfg["rl"].get("checkpoint"))
        print(
            "timeout:",
            get_run_timeout(written_cfg["run"]),
        )

        if written_cfg["rl"]["enabled"]:
            print(
                "reward weights:",
                build_reward_weight_tags(written_cfg),
            )

    return config_path
