import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scienceplots


OUTPUT_NAME = "combined_run_metrics"
OUTPUT_DPI = 1000
X_TICK_ROTATION = 25
BAR_WIDTH = 0.8

COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#999999",
]

HATCHES = [
    "///",
    "\\\\\\",
    "xxx",
    "---",
    "+++",
    "...",
    "**",
    "oo",
]

# Preferred ordering in legends and grouped bars.
ALGORITHM_ORDER = [
    "FCFS/B+IPM",
    "FCFS/B PSAS RL Budiarjo",
    "FCFS/B PSUS RL Budiarjo",
]


def safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))
    return text.strip("-")


def resolve_uid_root(value):
    path = Path(value).expanduser()

    if path.is_dir():
        return path.resolve()

    path = Path("results") / value

    if path.is_dir():
        return path.resolve()

    raise FileNotFoundError(
        f"Results UID directory not found: {value}"
    )


def epoch_number(path):
    match = re.fullmatch(
        r"(?:best_)?epoch_(\d+)",
        path.parent.name,
    )

    if match is None:
        return -1

    return int(match.group(1))


def find_run_metrics(run_dir):
    direct_metrics = run_dir / "metrics.csv"

    if direct_metrics.is_file():
        return direct_metrics

    best_metrics = sorted(
        run_dir.glob("best_epoch_*/metrics.csv"),
        key=epoch_number,
    )

    if best_metrics:
        return best_metrics[-1]

    candidates = []

    for metrics_path in run_dir.rglob("metrics.csv"):
        relative_parts = metrics_path.relative_to(
            run_dir
        ).parts

        if any(
            part in {
                "plots",
                "metrics_comparison",
                "generated_configs",
            }
            for part in relative_parts
        ):
            continue

        if any(
            re.fullmatch(r"epoch_\d+", part)
            for part in relative_parts
        ):
            continue

        candidates.append(metrics_path)

    if not candidates:
        return None

    candidates.sort()
    return candidates[0]


def get_metric(row, columns, metrics_path):
    for column in columns:
        if column in row.index:
            return float(row[column])

    print(
        f"Missing columns {columns} in {metrics_path}"
    )

    return float("nan")


def read_metrics(metrics_path):
    try:
        dataframe = pd.read_csv(metrics_path)
    except Exception as exc:
        print(
            f"Could not read {metrics_path}: {exc}"
        )
        return None

    if dataframe.empty:
        print(f"Empty metrics file: {metrics_path}")
        return None

    row = dataframe.iloc[0]

    return {
        "waiting_time": get_metric(
            row,
            [
                "mean_waiting_time",
            ],
            metrics_path,
        ),
        "energy_waste": get_metric(
            row,
            [
                "energy_waste",
                "total_energy_waste",
                "wasted_joules",
            ],
            metrics_path,
        ),
    }


def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def format_weight(value):
    number = float(value)

    if number.is_integer():
        return str(int(number))

    return f"{number:g}"


def extract_weights(run_name):
    match = re.search(
        r"(?:^|_)w1(?P<w1>-?\d+(?:\.\d+)?)"
        r"_w2(?P<w2>-?\d+(?:\.\d+)?)(?:_|$)",
        run_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return (
        format_weight(match.group("w1")),
        format_weight(match.group("w2")),
    )


def extract_timeout(run_name):
    match = re.search(
        r"(?:^|_)timeout-(?P<timeout>\d+(?:\.\d+)?)"
        r"(?:_|$)",
        run_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    timeout = float(match.group("timeout"))

    if timeout.is_integer():
        return int(timeout)

    return timeout


def timeout_tick_label(timeout):
    if timeout is None:
        return "No timeout"

    return f"{timeout:g} s" if isinstance(timeout, float) else f"{timeout} s"


def algorithm_base_label(run_name):
    """Map a directory name to a publication-facing algorithm label."""
    lower_name = run_name.lower()

    # Check the most specific names before their shorter prefixes.
    if lower_name.startswith("easy_ipm"):
        return "FCFS/B+IPM"

    if (
        lower_name.startswith("easy_psas_budiarjo")
        or lower_name.startswith("easy_baseline_psas_budiarjo")
    ):
        return "FCFS/B PSAS RL Budiarjo"

    if lower_name.startswith("easy_psus_budiarjo"):
        return "FCFS/B PSUS RL Budiarjo"

    if lower_name.startswith("easy_psas"):
        return "FCFS/B+IPM"

    # Fallback formatting for any other experiment directory.
    cleaned = re.sub(
        r"(?:^|_)timeout-\d+(?:\.\d+)?(?:_|$)",
        "_",
        run_name,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:^|_)w1-?\d+(?:\.\d+)?"
        r"_w2-?\d+(?:\.\d+)?(?:_|$)",
        "_",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"easy",
        "FCFS/B",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"_+", " ", cleaned).strip()

    return cleaned or run_name


def parse_run_label(run_name):
    base_label = algorithm_base_label(run_name)
    weights = extract_weights(run_name)
    timeout = extract_timeout(run_name)

    if weights is None:
        display_label = base_label
        style_key = base_label
    else:
        display_label = (
            f"{base_label} "
            f"(W1={weights[0]}, W2={weights[1]})"
        )
        style_key = display_label

    return {
        "raw_label": run_name,
        "algorithm_base": base_label,
        "algorithm_label": display_label,
        "style_key": style_key,
        "timeout": timeout,
        "timeout_label": timeout_tick_label(timeout),
    }


def algorithm_sort_key(label):
    for index, preferred in enumerate(ALGORITHM_ORDER):
        if label == preferred or label.startswith(f"{preferred} "):
            return (index, natural_sort_key(label))

    return (len(ALGORITHM_ORDER), natural_sort_key(label))


def timeout_sort_key(timeout):
    # Keep numeric timeout values first, then standalone no-timeout runs.
    if timeout is None:
        return (1, float("inf"))

    return (0, float(timeout))


def discover_runs(workload_dir):
    records = []

    ignored_names = {
        "generated_configs",
        "metrics_comparison",
        "plots",
    }

    for run_dir in sorted(workload_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        if run_dir.name in ignored_names:
            continue

        metrics_path = find_run_metrics(run_dir)

        if metrics_path is None:
            continue

        metrics = read_metrics(metrics_path)

        if metrics is None:
            continue

        label_info = parse_run_label(run_dir.name)

        records.append({
            "label": label_info["algorithm_label"],
            **label_info,
            "run_dir": str(run_dir.resolve()),
            "metrics_path": str(
                metrics_path.resolve()
            ),
            **metrics,
        })

    records.sort(
        key=lambda record: (
            timeout_sort_key(record["timeout"]),
            algorithm_sort_key(record["algorithm_label"]),
        )
    )

    return records


def format_short_number(value):
    if pd.isna(value):
        return ""

    value = float(value)
    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}T"

    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if absolute_value >= 1_000:
        return f"{value / 1_000:.1f}K"

    if absolute_value >= 100:
        return f"{value:.0f}"

    if absolute_value >= 10:
        return f"{value:.1f}"

    if absolute_value >= 1:
        return f"{value:.2f}"

    if absolute_value == 0:
        return "0"

    return f"{value:.3g}"


def annotate_bars(ax, bars):
    for bar in bars:
        value = bar.get_height()

        if pd.isna(value):
            continue

        if value >= 0:
            offset = 3
            vertical_alignment = "bottom"
        else:
            offset = -3
            vertical_alignment = "top"

        ax.annotate(
            format_short_number(value),
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=7,
        )


def build_styles(records):
    style_keys = sorted(
        {
            record["style_key"]
            for record in records
        },
        key=algorithm_sort_key,
    )

    return {
        style_key: (
            COLORS[index % len(COLORS)],
            HATCHES[index % len(HATCHES)],
        )
        for index, style_key in enumerate(style_keys)
    }


def grouped_dimensions(records):
    """Return hybrid x-axis groups and all algorithm style keys.

    Timeout runs are grouped first by numeric timeout value. Runs without a
    timeout remain standalone and appear afterward using their full algorithm
    labels as x-axis ticks.
    """
    timeout_values = sorted(
        {
            record["timeout"]
            for record in records
            if record["timeout"] is not None
        },
        key=float,
    )

    groups = []

    for timeout in timeout_values:
        timeout_records = sorted(
            (
                record
                for record in records
                if record["timeout"] == timeout
            ),
            key=lambda record: algorithm_sort_key(
                record["algorithm_label"]
            ),
        )
        groups.append({
            "key": ("timeout", timeout),
            "tick_label": timeout_tick_label(timeout),
            "records": timeout_records,
            "is_timeout_group": True,
        })

    no_timeout_records = sorted(
        (
            record
            for record in records
            if record["timeout"] is None
        ),
        key=lambda record: algorithm_sort_key(
            record["algorithm_label"]
        ),
    )

    groups.extend(
        {
            "key": ("no_timeout", record["raw_label"]),
            "tick_label": record["algorithm_label"],
            "records": [record],
            "is_timeout_group": False,
        }
        for record in no_timeout_records
    )

    algorithms = sorted(
        {
            record["style_key"]
            for record in records
        },
        key=algorithm_sort_key,
    )

    return groups, algorithms

def plot_metric(
    ax,
    records,
    metric_name,
    ylabel,
    title,
    styles,
):
    groups, _ = grouped_dimensions(records)
    group_positions = np.arange(len(groups), dtype=float)

    maximum_timeout_algorithms = max(
        [
            len(group["records"])
            for group in groups
            if group["is_timeout_group"]
        ]
        or [1]
    )
    timeout_bar_width = min(0.84, BAR_WIDTH) / max(
        1,
        maximum_timeout_algorithms,
    )
    standalone_bar_width = min(0.60, BAR_WIDTH)

    for group_index, group in enumerate(groups):
        group_records = group["records"]

        if group["is_timeout_group"]:
            current_bar_width = timeout_bar_width
        else:
            # No-timeout methods retain one standalone bar and their own
            # algorithm label as the x tick, matching the original layout.
            current_bar_width = standalone_bar_width

        for record_index, record in enumerate(group_records):
            if group["is_timeout_group"]:
                offset = (
                    record_index
                    - (len(group_records) - 1) / 2
                ) * current_bar_width
            else:
                offset = 0.0

            color, hatch = styles[record["style_key"]]

            bars = ax.bar(
                group_positions[group_index] + offset,
                record[metric_name],
                width=current_bar_width,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            )

            annotate_bars(ax, bars)

    ax.set_xticks(group_positions)
    tick_labels = ax.set_xticklabels(
        [group["tick_label"] for group in groups],
    )

    # Numeric timeout ticks are short, so keep them horizontal. Long
    # standalone no-timeout algorithm labels retain the configured rotation.
    for tick_label, group in zip(tick_labels, groups):
        if group["is_timeout_group"]:
            tick_label.set_rotation(0)
            tick_label.set_ha("center")
        else:
            tick_label.set_rotation(X_TICK_ROTATION)
            tick_label.set_ha(
                "right" if X_TICK_ROTATION else "center"
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.margins(y=0.17)

def add_shared_legend(fig, records, styles):
    _, algorithms = grouped_dimensions(records)

    handles = []

    for style_key in algorithms:
        color, hatch = styles[style_key]
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
                label=style_key,
                alpha=0.85,
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(3, max(1, len(handles))),
        frameon=True,
        fontsize=8,
    )


def write_experiment_list(
    records,
    comparison_dir,
    experiment_name,
    platform_name,
    workload_name,
):
    text_path = (
        comparison_dir
        / "experiments.txt"
    )

    json_path = (
        comparison_dir
        / "experiments.json"
    )

    lines = [
        f"Experiment: {experiment_name}",
        f"Platform: {platform_name}",
        f"Workload: {workload_name}",
        "=" * 60,
        "",
    ]

    for index, record in enumerate(
        records,
        start=1,
    ):
        lines.extend([
            f"{index}. {record['algorithm_label']}",
            f"   Original name: {record['raw_label']}",
            f"   Timeout: {record['timeout_label']}",
            f"   Run directory: {record['run_dir']}",
            f"   Metrics: {record['metrics_path']}",
            f"   Mean waiting time: {record['waiting_time']}",
            f"   Energy waste: {record['energy_waste']}",
            "",
        ])

    text_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=4,
        )

    print("experiment list:", text_path)
    print("experiment JSON:", json_path)


def plot_workload(
    experiment_dir,
    platform_dir,
    workload_dir,
):
    experiment_name = experiment_dir.name
    records = discover_runs(
        workload_dir
    )

    if not records:
        print(
            "No metrics found:",
            workload_dir,
        )
        return

    comparison_dir = (
        workload_dir
        / "metrics_comparison"
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    platform_name = (
        platform_dir.name
    )

    workload_name = (
        workload_dir.name
    )

    write_experiment_list(
        records,
        comparison_dir,
        experiment_name,
        platform_name,
        workload_name,
    )
    styles = build_styles(
        records
    )

    groups, algorithms = grouped_dimensions(records)

    figure_width = max(
        12,
        len(groups) * 2.8,
    )

    figure_height = 6.3 + max(
        0,
        (len(algorithms) - 3) * 0.25,
    )

    fig, (
        waiting_ax,
        waste_ax,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            figure_width,
            figure_height,
        ),
    )

    plot_metric(
        waiting_ax,
        records,
        "waiting_time",
        "Mean Waiting Time (s)",
        "Mean Waiting Time",
        styles,
    )

    plot_metric(
        waste_ax,
        records,
        "energy_waste",
        "Energy Waste (J)",
        "Energy Waste",
        styles,
    )

    fig.suptitle(
        f"{experiment_name} / {platform_name} / {workload_name}",
        y=0.91,
    )
    add_shared_legend(
        fig,
        records,
        styles,
    )

    fig.tight_layout(
        rect=[
            0,
            0.0,
            1,
            0.87,
        ]
    )

    png_path = (
        comparison_dir
        / f"{OUTPUT_NAME}.png"
    )

    pdf_path = (
        comparison_dir
        / f"{OUTPUT_NAME}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nGenerated metrics comparison:")
    print("platform:", platform_name)
    print("workload:", workload_name)
    print("runs:", len(records))
    print("PNG:", png_path)
    print("PDF:", pdf_path)


def generate_uid_metrics(uid_root):
    uid_root = resolve_uid_root(uid_root)

    plt.style.use([
        "science",
        "no-latex",
        "grid",
    ])

    generated_count = 0

    for experiment_dir in sorted(uid_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        for platform_dir in sorted(experiment_dir.iterdir()):
            if not platform_dir.is_dir():
                continue

            for workload_dir in sorted(platform_dir.iterdir()):
                if not workload_dir.is_dir():
                    continue

                if workload_dir.name in {
                    "generated_configs",
                    "metrics_comparison",
                    "plots",
                }:
                    continue

                records = discover_runs(workload_dir)

                if not records:
                    continue

                plot_workload(
                    experiment_dir,
                    platform_dir,
                    workload_dir,
                )

                generated_count += 1

    if generated_count == 0:
        raise RuntimeError(
            "No experiment/platform/workload directories "
            f"containing metrics.csv were found in {uid_root}"
        )

    print(
        f"\nGenerated {generated_count} "
        "experiment/platform/workload metrics comparisons."
    )

def generate_comparison(
    algo_config,
    plot_dir,
    output_name="combined_run_metrics",
    x_label=None,
    x_tick_rotation=25,
    bar_width=0.8,
    figure_size=None,
):
    records = []

    for label, config in algo_config.items():
        metrics_path = Path(config["base_dir"]) / "metrics.csv"

        if not metrics_path.is_file():
            print(f"Skipping metrics entry, missing: {metrics_path}")
            continue

        metrics = read_metrics(metrics_path)

        if metrics is None:
            continue

        source_label = config.get("tick_label", label)
        label_info = parse_run_label(source_label)

        if "timeout" in config:
            label_info["timeout"] = config["timeout"]
            label_info["timeout_label"] = timeout_tick_label(
                config["timeout"]
            )

        if "algorithm_label" in config:
            label_info["algorithm_label"] = config["algorithm_label"]
            label_info["style_key"] = config["algorithm_label"]

        records.append({
            "label": label_info["algorithm_label"],
            **label_info,
            "run_dir": str(Path(config["base_dir"]).resolve()),
            "metrics_path": str(metrics_path.resolve()),
            **metrics,
        })

    if not records:
        print(f"No valid metrics found for: {plot_dir}")
        return

    records.sort(
        key=lambda record: (
            timeout_sort_key(record["timeout"]),
            algorithm_sort_key(record["algorithm_label"]),
        )
    )

    output_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = build_styles(records)
    groups, algorithms = grouped_dimensions(records)

    if figure_size is None:
        figure_size = (
            max(11, len(groups) * 2.8),
            6.2 + max(0, (len(algorithms) - 3) * 0.25),
        )

    fig, (
        waiting_ax,
        waste_ax,
    ) = plt.subplots(
        1,
        2,
        figsize=figure_size,
    )

    old_rotation = globals()["X_TICK_ROTATION"]
    old_bar_width = globals()["BAR_WIDTH"]

    globals()["X_TICK_ROTATION"] = x_tick_rotation
    globals()["BAR_WIDTH"] = bar_width

    try:
        plot_metric(
            waiting_ax,
            records,
            "waiting_time",
            "Mean Waiting Time (s)",
            "Mean Waiting Time",
            styles,
        )

        plot_metric(
            waste_ax,
            records,
            "energy_waste",
            "Energy Waste (J)",
            "Energy Waste",
            styles,
        )
    finally:
        globals()["X_TICK_ROTATION"] = old_rotation
        globals()["BAR_WIDTH"] = old_bar_width

    if x_label:
        fig.supxlabel(x_label)

    add_shared_legend(fig, records, styles)
    bottom_margin = 0.03 if x_label else 0.0
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.87])

    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"

    fig.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved PNG:", png_path)
    print("Saved PDF:", pdf_path)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} results/<UID>"
        )

    generate_uid_metrics(
        sys.argv[1]
    )


if __name__ == "__main__":
    main()
