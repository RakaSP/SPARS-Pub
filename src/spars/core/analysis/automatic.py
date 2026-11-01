"""Generic run indexing, grouping and optional plotting."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RunRecord = dict[str, Any]


def build_run_label(run: RunRecord) -> str:
    from spars.core.analysis.metrics import display_label

    return display_label(run).replace("\n", " | ")


def group_runs(runs: Sequence[RunRecord]) -> list[list[RunRecord]]:
    groups = defaultdict(list)
    for run in runs:
        phase = "evaluation" if run["phase"] in {"test", "run"} else "training"
        key = (
            str(Path(run["platform"]).resolve()),
            str(Path(run["workload"]).resolve()),
            phase,
        )
        groups[key].append(run)
    return list(groups.values())


def record_run(record: RunRecord) -> None:
    root = Path(record["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    index = root / "run_index.json"
    records = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    records = [item for item in records if item["output"] != record["output"]]
    records.append(record)
    temporary = index.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(index)


def load_saved_runs(root: str | Path) -> list[RunRecord]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    indexes = sorted(root.rglob("run_index.json"))
    for index in indexes:
        for record in json.loads(index.read_text(encoding="utf-8")):
            old_root = Path(record["run_root"])
            for field in ("output", "config_path"):
                value = record.get(field)
                if value is not None:
                    record[field] = str(
                        index.parent / Path(value).relative_to(old_root)
                    )
            record["run_root"] = str(index.parent)
            records.append(record)
    # Metadata also exposes runs completed before their parent indexed them.
    known = {item["output"] for item in records}
    metadata_files = sorted(root.rglob("run_metadata.json"))
    for path in metadata_files:
        cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
        directory = path.parent
        rl = cfg["rl"]
        phase = (
            "train"
            if rl["enabled"] and rl["learn"]
            else "test"
            if rl["enabled"]
            else "run"
        )
        outputs = (
            sorted(directory.glob("best_epoch_*/metrics.csv"))
            if phase == "train"
            else [directory / "metrics.csv"]
        )
        for metrics in outputs:
            if not metrics.is_file() or str(metrics.parent) in known:
                continue
            identity = cfg["experiment"]
            records.append(
                {
                    "name": identity["name"],
                    "display_name": identity.get("display_name", identity["name"]),
                    "phase": phase,
                    "platform": cfg["paths"]["platform"],
                    "workload": cfg["paths"]["workload"],
                    "output": str(metrics.parent),
                    "run_root": str(root),
                    "timeout": cfg["run"]["power_management"]["config"].get("timeout"),
                    "sweep": identity.get("sweep", {}),
                }
            )
    return records


def plot_gantt(run: RunRecord) -> None:
    from spars.core.analysis.gantt import plot_run

    directory = Path(run["output"])
    if not (directory / "node_log.csv").is_file():
        return
    plot_run(
        experiment_name=build_run_label(run),
        platform_name=Path(run["platform"]).stem,
        workload_name=Path(run["workload"]).stem,
        run_dir=directory,
        node_log_path=directory / "node_log.csv",
    )


def generate_all_plots(
    runs: Sequence[RunRecord],
    generate_gantt: bool = False,
    generate_metrics: bool = False,
) -> None:
    if generate_gantt:
        for run in runs:
            plot_gantt(run)
    if generate_metrics:
        from spars.core.analysis.metrics import plot_comparison

        for index, group in enumerate(group_runs(runs), start=1):
            phase = "evaluation" if group[0]["phase"] in {"test", "run"} else "training"
            output = (
                Path(group[0]["run_root"])
                / "plots"
                / Path(group[0]["platform"]).stem
                / Path(group[0]["workload"]).stem
                / phase
            )
            # A numeric group suffix prevents basename collisions without
            # embedding policy-specific knowledge in reporting.
            plot_comparison(group, output / f"comparison-{index}")
