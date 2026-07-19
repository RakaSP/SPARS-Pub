import sys
from pathlib import Path

import yaml

from Runner import run_with_config


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <config.yaml>"
        )

    config_path = Path(sys.argv[1])

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path.resolve()}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Invalid configuration in {config_path}"
        )

    print("\nRunning configuration:")
    print("config:", config_path)
    print("algorithm:", cfg["run"]["algorithm"])
    print("workload:", cfg["paths"]["workload"])
    print("output:", cfg["paths"]["output"])
    print("run parameters:", cfg["run"])

    run_with_config(cfg)


if __name__ == "__main__":
    main()