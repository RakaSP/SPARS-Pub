import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import PlotGantt
import PlotMetrics


def safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))
    return text.strip("-")


def file_stem(path):
    return safe_name(Path(path).stem)


def build_run_label(run):
    return Path(run["output"]).name

def get_root_uid(run):
    root_uid = run.get("root_uid")

    if root_uid is None:
        root_uid = run.get("uid")

    if root_uid is None:
        raise KeyError("Run record is missing both 'root_uid' and 'uid'")

    return safe_name(root_uid)


def find_uid_root(run):
    output_path = Path(run["output"]).resolve()
    root_uid = get_root_uid(run)

    for candidate in [output_path, *output_path.parents]:
        if candidate.name == root_uid:
            return candidate

    raise ValueError(
        "Could not find the UID directory in the run output path.\n"
        f"UID: {root_uid}\n"
        f"Output: {output_path}"
    )


def build_comparison_root(run):
    uid_root = find_uid_root(run)
    experiment_name = safe_name(run["name"])
    platform_name = file_stem(run["platform"])
    workload_name = file_stem(run["workload"])

    return (
        uid_root
        / experiment_name
        / platform_name
        / workload_name
    )


def write_experiment_list(runs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "experiments.txt"
    json_path = output_dir / "experiments.json"

    lines = [
        "Experiments included in this metrics comparison",
        "=" * 60,
        "",
    ]

    json_records = []

    for index, run in enumerate(runs, start=1):
        record = {
            "name": run["name"],
            "phase": run.get("phase"),
            "uid": run.get("uid"),
            "root_uid": run.get("root_uid", run.get("uid")),
            "label": build_run_label(run),
            "algorithm": run["algorithm"],
            "timeout": run.get("timeout"),
            "platform": run["platform"],
            "workload": run["workload"],
            "output": run["output"],
            "config_path": run.get("config_path"),
            "run_parameters": run.get("run_parameters", {}),
            "sweep": run.get("sweep", {}),
        }

        lines.extend([
            f"{index}. {record['name']}",
            f"   Label: {record['label']}",
            f"   Phase: {record['phase']}",
            f"   UID: {record['uid']}",
            f"   Root UID: {record['root_uid']}",
            f"   Algorithm: {record['algorithm']}",
            f"   Timeout: {record['timeout']}",
            f"   Platform: {record['platform']}",
            f"   Workload: {record['workload']}",
            f"   Output: {record['output']}",
            f"   Config: {record['config_path']}",
            f"   Run parameters: {record['run_parameters']}",
            f"   Sweep values: {record['sweep']}",
            "",
        ])

        json_records.append(record)

    text_path.write_text("\n".join(lines), encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_records, file, indent=4)

    print("experiment list:", text_path)
    print("experiment JSON:", json_path)


def plot_gantt(run):
    run_dir = Path(run["output"])
    node_log_path = run_dir / "node_log.csv"

    if not node_log_path.exists():
        print(
            f"Skipping Gantt plot, missing: "
            f"{node_log_path}"
        )
        return

    print("\nGenerating Gantt charts:")
    print("run:", run["name"])
    print("input:", node_log_path)
    print("output:", run_dir / "plots")

    generated_count = PlotGantt.plot_run(
        experiment_name=run["name"],
        platform_name=file_stem(run["platform"]),
        workload_name=file_stem(run["workload"]),
        run_dir=run_dir,
        node_log_path=node_log_path,
    )

    print(
        f"Generated {generated_count} "
        "Gantt images for this run."
    )

    plt.close("all")


def plot_combined_metrics(runs, comparison_root):
    valid_runs = []

    for run in runs:
        metrics_path = Path(run["output"]) / "metrics.csv"

        if not metrics_path.exists():
            print(f"Skipping metrics entry, missing: {metrics_path}")
            continue

        valid_runs.append(run)

    if not valid_runs:
        print(f"No metrics found for comparison root: {comparison_root}")
        return

    comparison_dir = Path(comparison_root) / "metrics_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    write_experiment_list(
        runs=valid_runs,
        output_dir=comparison_dir,
    )

    algo_config = {}
    label_counts = defaultdict(int)

    for run in valid_runs:
        base_label = build_run_label(run)
        label_counts[base_label] += 1
        occurrence = label_counts[base_label]

        if occurrence == 1:
            label = base_label
        else:
            label = f"{base_label} #{occurrence}"

        algo_config[label] = {
            "type": "single",
            "base_dir": run["output"],
            "tick_label": label,
        }


    print("\nGenerating metrics comparison:")
    print("platform:", file_stem(valid_runs[0]["platform"]))
    print("workload:", file_stem(valid_runs[0]["workload"]))
    print("runs:", len(valid_runs))
    print("output:", comparison_dir)

    for run in valid_runs:
        print(" -", build_run_label(run))

    PlotMetrics.generate_comparison(
        algo_config=algo_config,
        plot_dir=comparison_dir,
        output_name="combined_run_metrics",
        x_label="Experiment",
        x_tick_rotation=35,
        bar_width=0.55,
        figure_size=(
            max(12, len(valid_runs) * 2.5),
            5,
        ),
    )
    
    plt.close("all")

    generated_files = sorted(
        path
        for path in comparison_dir.iterdir()
        if path.is_file()
    )

    if generated_files:
        print("\nMetrics comparison files:")

        for path in generated_files:
            print(" -", path)


def group_runs_by_comparison_root(runs):
    grouped_runs = defaultdict(list)

    for run in runs:
        comparison_root = build_comparison_root(run)
        grouped_runs[comparison_root].append(run)

    return grouped_runs


def generate_all_plots(
    runs,
    generate_gantt=True,
    generate_metrics=True,
):
    if not runs:
        print("No completed runs were supplied to the plot generator")
        return

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        if generate_gantt:
            for run in runs:
                plot_gantt(run)

        if generate_metrics:
            grouped_runs = group_runs_by_comparison_root(runs)

            for comparison_root, comparison_runs in grouped_runs.items():
                plot_combined_metrics(
                    runs=comparison_runs,
                    comparison_root=comparison_root,
                )
    finally:
        plt.show = original_show
        plt.close("all")