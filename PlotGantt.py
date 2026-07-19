import colorsys
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


# ============================================================
# Parameters
# ============================================================

OUTPUT_NAME = "gantt"
OUTPUT_DPI = 180

GENERATE_OVERVIEW = True
GENERATE_WINDOWS = True

# Each window shows 100,000 simulation-time units.
WINDOW_SIZE = 100_000

FIGURE_WIDTH = 24
MIN_FIGURE_HEIGHT = 8
MAX_FIGURE_HEIGHT = 30
HEIGHT_PER_NODE = 0.12

SHOW_JOB_LABELS = True
JOB_LABEL_FONTSIZE = 6
THIN_JOB_LABEL_FONTSIZE = 5

# A job narrower than this fraction of the displayed x-range
# gets a vertical label.
THIN_JOB_LABEL_THRESHOLD = 0.002

SHOW_ZERO_RUNTIME_JOBS = True
ZERO_RUNTIME_LINEWIDTH = 0.8
ZERO_RUNTIME_LABEL_FONTSIZE = 5

SHOW_JOB_ARRIVALS = True
ARRIVAL_LABEL_FONTSIZE = 8
ARRIVAL_MARKER_SIZE = 32
ARRIVAL_BAND_HEIGHT = 32
ARRIVAL_LABEL_LEVELS = 5

MAX_X_TICKS = 14
MAX_Y_TICKS = 20

STATE_COLORS = {
    -1: "#A9B7C6",
    -2: "#C0392B",
    -3: "#27AE60",
    -4: "#2C3E7A",
}

TERMINATED_COLOR = "#3D3535"

RESERVED_HUES = [
    (0.00, 0.06),
    (0.33, 0.13),
    (0.55, 0.07),
    (0.60, 0.06),
    (0.65, 0.06),
]

IGNORED_DIRECTORY_NAMES = {
    "generated_configs",
    "metrics_comparison",
    "plots",
}


# ============================================================
# Path helpers
# ============================================================

def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


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


def find_run_node_log(run_dir):
    direct_log = run_dir / "node_log.csv"

    if direct_log.is_file():
        return direct_log

    best_logs = sorted(
        run_dir.glob("best_epoch_*/node_log.csv"),
        key=epoch_number,
    )

    if best_logs:
        return best_logs[-1]

    candidates = []

    for node_log_path in run_dir.rglob("node_log.csv"):
        relative_parts = node_log_path.relative_to(
            run_dir
        ).parts

        if any(
            part in IGNORED_DIRECTORY_NAMES
            for part in relative_parts
        ):
            continue

        if any(
            re.fullmatch(r"epoch_\d+", part)
            for part in relative_parts
        ):
            continue

        candidates.append(node_log_path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda path: natural_sort_key(str(path))
    )

    return candidates[0]


# ============================================================
# Color helpers
# ============================================================

def hue_is_reserved(hue):
    for center, half_width in RESERVED_HUES:
        difference = abs(hue - center)
        difference = min(
            difference,
            1.0 - difference,
        )

        if difference < half_width:
            return True

    return False


def build_job_palette(number_of_colors=100):
    palette = []
    hue = 0.0
    step = 1.0 / (number_of_colors * 1.5)

    saturation_cycle = [
        0.75,
        0.55,
    ]

    lightness_cycle = [
        0.45,
        0.60,
    ]

    while len(palette) < number_of_colors:
        hue = (hue + step) % 1.0

        if hue_is_reserved(hue):
            continue

        index = len(palette)

        saturation = saturation_cycle[
            index % len(saturation_cycle)
        ]

        lightness = lightness_cycle[
            index % len(lightness_cycle)
        ]

        red, green, blue = colorsys.hls_to_rgb(
            hue,
            lightness,
            saturation,
        )

        palette.append(
            (
                red,
                green,
                blue,
            )
        )

    return palette


JOB_PALETTE = build_job_palette(100)


def build_job_colors(timeline):
    job_ids = sorted({
        event["job_id"]
        for event in timeline
        if event["job_id"] > 0
    })

    return {
        job_id: JOB_PALETTE[
            index % len(JOB_PALETTE)
        ]
        for index, job_id in enumerate(job_ids)
    }


def get_text_color(background_color):
    if isinstance(background_color, str):
        hexadecimal = background_color.lstrip("#")

        red, green, blue = (
            int(
                hexadecimal[index:index + 2],
                16,
            ) / 255.0
            for index in (
                0,
                2,
                4,
            )
        )
    else:
        red, green, blue = background_color[:3]

    luminance = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return (
        "white"
        if luminance < 0.5
        else "black"
    )


def get_event_color(event, job_colors):
    job_id = event["job_id"]

    if event["terminated"]:
        return TERMINATED_COLOR

    if job_id in STATE_COLORS:
        return STATE_COLORS[job_id]

    return job_colors[job_id]


# ============================================================
# Timeline parsing
# ============================================================

def parse_nodes(value):
    if value is None:
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):
        return [
            int(node_id)
            for node_id in value
        ]

    if pd.isna(value):
        return []

    return [
        int(item)
        for item in re.findall(
            r"-?\d+",
            str(value),
        )
    ]


def parse_boolean(value):
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def read_timeline(file_path):
    dataframe = pd.read_csv(file_path)

    node_column = (
        "allocated_resources"
        if "allocated_resources" in dataframe.columns
        else "nodes"
    )

    start_column = (
        "starting_time"
        if "starting_time" in dataframe.columns
        else "start_time"
    )

    type_column = (
        "type"
        if "type" in dataframe.columns
        else "state"
    )

    required_columns = {
        node_column,
        start_column,
        "finish_time",
        "job_id",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    has_submission_time = (
        "submission_time"
        in dataframe.columns
    )

    timeline = []

    for _, row in dataframe.iterrows():
        nodes = parse_nodes(
            row[node_column]
        )

        if not nodes:
            continue

        start_time = float(
            row[start_column]
        )

        finish_time = float(
            row["finish_time"]
        )

        # Keep zero-runtime jobs.
        if finish_time < start_time:
            continue

        event = {
            "starting_time": start_time,
            "finish_time": finish_time,
            "allocated_resources": nodes,
            "type": row.get(type_column),
            "job_id": int(row["job_id"]),
            "terminated": parse_boolean(
                row.get(
                    "terminated",
                    False,
                )
            ),
        }

        if (
            has_submission_time
            and pd.notna(
                row["submission_time"]
            )
        ):
            event["submission_time"] = float(
                row["submission_time"]
            )

        timeline.append(event)

    return timeline


def contiguous_groups(nodes):
    nodes = sorted(set(nodes))

    if not nodes:
        return []

    groups = []

    group_start = nodes[0]
    previous_node = nodes[0]

    for node_id in nodes[1:]:
        if node_id == previous_node + 1:
            previous_node = node_id
            continue

        groups.append(
            (
                group_start,
                previous_node - group_start + 1,
            )
        )

        group_start = node_id
        previous_node = node_id

    groups.append(
        (
            group_start,
            previous_node - group_start + 1,
        )
    )

    return groups


# ============================================================
# Window helpers
# ============================================================

def event_intersects_window(
    event,
    window_start,
    window_end,
):
    start_time = float(
        event["starting_time"]
    )

    finish_time = float(
        event["finish_time"]
    )

    if finish_time == start_time:
        return (
            window_start
            <= start_time
            < window_end
        )

    return (
        finish_time > window_start
        and start_time < window_end
    )


def timeline_for_window(
    timeline,
    window_start,
    window_end,
):
    return [
        event
        for event in timeline
        if event_intersects_window(
            event,
            window_start,
            window_end,
        )
    ]


def window_has_arrivals(
    timeline,
    window_start,
    window_end,
):
    for event in timeline:
        submission_time = event.get(
            "submission_time"
        )

        if submission_time is None:
            continue

        if (
            window_start
            <= float(submission_time)
            < window_end
        ):
            return True

    return False


# ============================================================
# Plot rendering
# ============================================================

def plot_job_intervals(
    ax,
    timeline,
    job_colors,
    window_start,
    window_end,
):
    rectangles = []
    rectangle_colors = []
    labels = []

    displayed_range = max(
        window_end - window_start,
        1.0,
    )

    thin_threshold = (
        displayed_range
        * THIN_JOB_LABEL_THRESHOLD
    )

    for event in timeline:
        original_start = float(
            event["starting_time"]
        )

        original_finish = float(
            event["finish_time"]
        )

        original_duration = (
            original_finish
            - original_start
        )

        job_id = event["job_id"]

        color = get_event_color(
            event,
            job_colors,
        )

        groups = contiguous_groups(
            event["allocated_resources"]
        )

        # ----------------------------------------------------
        # Positive-runtime event
        # ----------------------------------------------------

        if original_duration > 0:
            visible_start = max(
                original_start,
                window_start,
            )

            visible_finish = min(
                original_finish,
                window_end,
            )

            visible_duration = (
                visible_finish
                - visible_start
            )

            if visible_duration <= 0:
                continue

            for (
                node_start,
                node_count,
            ) in groups:
                rectangles.append(
                    Rectangle(
                        (
                            visible_start,
                            node_start,
                        ),
                        visible_duration,
                        node_count,
                    )
                )

                rectangle_colors.append(
                    color
                )

                if (
                    SHOW_JOB_LABELS
                    and job_id > 0
                ):
                    labels.append({
                        "job_id": job_id,
                        "x": (
                            visible_start
                            + visible_finish
                        ) / 2.0,
                        "y": (
                            node_start
                            + node_count / 2.0
                        ),
                        "color": color,
                        "thin": (
                            visible_duration
                            < thin_threshold
                        ),
                    })

            continue

        # ----------------------------------------------------
        # Zero-runtime event
        # ----------------------------------------------------

        if (
            not SHOW_ZERO_RUNTIME_JOBS
            or job_id <= 0
        ):
            continue

        if not (
            window_start
            <= original_start
            < window_end
        ):
            continue

        for (
            node_start,
            node_count,
        ) in groups:
            middle_node = (
                node_start
                + node_count / 2.0
            )

            ax.vlines(
                original_start,
                node_start,
                node_start + node_count,
                color=color,
                linewidth=ZERO_RUNTIME_LINEWIDTH,
                zorder=7,
            )

            if SHOW_JOB_LABELS:
                ax.annotate(
                    str(job_id),
                    xy=(
                        original_start,
                        middle_node,
                    ),
                    xytext=(2, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    color="black",
                    fontsize=ZERO_RUNTIME_LABEL_FONTSIZE,
                    fontweight="bold",
                    rotation=90,
                    rotation_mode="anchor",
                    clip_on=True,
                    zorder=8,
                )

    if rectangles:
        collection = PatchCollection(
            rectangles,
            facecolors=rectangle_colors,
            edgecolors="white",
            linewidths=0.12,
            rasterized=True,
        )

        ax.add_collection(collection)

    for label in labels:
        ax.text(
            label["x"],
            label["y"],
            str(label["job_id"]),
            ha="center",
            va="center",
            color=(
                "black"
                if label["thin"]
                else get_text_color(
                    label["color"]
                )
            ),
            fontsize=(
                THIN_JOB_LABEL_FONTSIZE
                if label["thin"]
                else JOB_LABEL_FONTSIZE
            ),
            fontweight="bold",
            rotation=(
                90
                if label["thin"]
                else 0
            ),
            rotation_mode="anchor",
            clip_on=True,
            zorder=8,
        )


def plot_arrivals(
    ax,
    full_timeline,
    num_nodes,
    window_start,
    window_end,
):
    if not SHOW_JOB_ARRIVALS:
        return

    arrival_times = {}

    for event in full_timeline:
        job_id = event["job_id"]

        if (
            job_id <= 0
            or "submission_time" not in event
        ):
            continue

        submission_time = float(
            event["submission_time"]
        )

        if not (
            window_start
            <= submission_time
            < window_end
        ):
            continue

        arrival_times.setdefault(
            job_id,
            submission_time,
        )

    if not arrival_times:
        return

    sorted_arrivals = sorted(
        arrival_times.items(),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    marker_y = num_nodes + 1.5

    ax.scatter(
        [
            submission_time
            for _, submission_time
            in sorted_arrivals
        ],
        [
            marker_y
        ] * len(sorted_arrivals),
        marker="v",
        s=ARRIVAL_MARKER_SIZE,
        color="red",
        edgecolors="darkred",
        linewidths=0.5,
        zorder=10,
        clip_on=False,
    )

    for index, (
        job_id,
        submission_time,
    ) in enumerate(sorted_arrivals):
        level = (
            index
            % ARRIVAL_LABEL_LEVELS
        )

        label_y = (
            marker_y
            + 3
            + level * 5
        )

        ax.text(
            submission_time,
            label_y,
            str(job_id),
            ha="center",
            va="bottom",
            color="red",
            fontsize=ARRIVAL_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
            clip_on=False,
            zorder=11,
        )

        ax.plot(
            [
                submission_time,
                submission_time,
            ],
            [
                marker_y + 0.5,
                label_y - 0.5,
            ],
            color="red",
            linewidth=0.35,
            alpha=0.6,
            zorder=9,
            clip_on=False,
        )


def plot_timeline(
    ax,
    timeline,
    full_timeline,
    title,
    num_nodes,
    window_start,
    window_end,
):
    job_colors = build_job_colors(
        full_timeline
    )

    top_limit = (
        num_nodes + ARRIVAL_BAND_HEIGHT
        if SHOW_JOB_ARRIVALS
        else num_nodes
    )

    ax.set_xlim(
        window_start,
        window_end,
    )

    ax.set_ylim(
        0,
        top_limit,
    )

    plot_job_intervals(
        ax=ax,
        timeline=timeline,
        job_colors=job_colors,
        window_start=window_start,
        window_end=window_end,
    )

    plot_arrivals(
        ax=ax,
        full_timeline=full_timeline,
        num_nodes=num_nodes,
        window_start=window_start,
        window_end=window_end,
    )

    if SHOW_JOB_ARRIVALS:
        ax.axhline(
            y=num_nodes,
            color="red",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=8,
        )

        ax.text(
            window_start,
            num_nodes + 1.5,
            "Job arrivals",
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=12,
        )

    ax.set_title(
        title,
        fontsize=16,
    )

    ax.set_xlabel(
        "Time",
        fontsize=12,
    )

    ax.set_ylabel(
        "Node ID",
        fontsize=12,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=MAX_X_TICKS
        )
    )

    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=9,
    )

    y_step = max(
        1,
        int(
            np.ceil(
                num_nodes
                / MAX_Y_TICKS
            )
        ),
    )

    y_ticks = list(
        range(
            0,
            num_nodes,
            y_step,
        )
    )

    if (
        not y_ticks
        or y_ticks[-1] != num_nodes - 1
    ):
        y_ticks.append(
            num_nodes - 1
        )

    ax.set_yticks(
        y_ticks
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linewidth=0.5,
    )

    state_labels = {
        -1: "Idle",
        -2: "Switching Off",
        -3: "Switching On",
        -4: "Sleeping",
    }

    legend_handles = [
        Patch(
            facecolor=STATE_COLORS[
                state_id
            ],
            edgecolor="white",
            label=state_label,
        )
        for (
            state_id,
            state_label,
        ) in state_labels.items()
    ]

    legend_handles.append(
        Patch(
            facecolor=TERMINATED_COLOR,
            edgecolor="white",
            label="Terminated",
        )
    )

    if SHOW_ZERO_RUNTIME_JOBS:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=ZERO_RUNTIME_LINEWIDTH,
                label="Zero-runtime job",
            )
        )

    if SHOW_JOB_ARRIVALS:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="v",
                color="red",
                markerfacecolor="red",
                markeredgecolor="darkred",
                markersize=7,
                linestyle="None",
                label="Job Arrival + Job ID",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.75,
    )


# ============================================================
# Plot generation
# ============================================================

def calculate_max_time(timeline):
    maximum = 0.0

    for event in timeline:
        maximum = max(
            maximum,
            float(
                event["starting_time"]
            ),
            float(
                event["finish_time"]
            ),
        )

        if "submission_time" in event:
            maximum = max(
                maximum,
                float(
                    event["submission_time"]
                ),
            )

    return max(
        maximum,
        1.0,
    )


def calculate_num_nodes(timeline):
    return (
        max(
            max(
                event["allocated_resources"]
            )
            for event in timeline
        )
        + 1
    )


def calculate_figure_height(num_nodes):
    return min(
        MAX_FIGURE_HEIGHT,
        max(
            MIN_FIGURE_HEIGHT,
            num_nodes * HEIGHT_PER_NODE,
        ),
    )


def save_plot(
    timeline,
    full_timeline,
    title,
    output_path,
    num_nodes,
    window_start,
    window_end,
):
    figure_height = calculate_figure_height(
        num_nodes
    )

    fig, ax = plt.subplots(
        figsize=(
            FIGURE_WIDTH,
            figure_height,
        )
    )

    plot_timeline(
        ax=ax,
        timeline=timeline,
        full_timeline=full_timeline,
        title=title,
        num_nodes=num_nodes,
        window_start=window_start,
        window_end=window_end,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)

    print("Saved:", output_path)


def plot_run(
    experiment_name,
    platform_name,
    workload_name,
    run_dir,
    node_log_path,
):
    timeline = read_timeline(
        node_log_path
    )

    if not timeline:
        print(
            "Skipping empty timeline:",
            node_log_path,
        )
        return 0

    num_nodes = calculate_num_nodes(
        timeline
    )

    max_time = calculate_max_time(
        timeline
    )

    output_dir = (
        run_dir
        / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_count = 0

    base_title = (
        f"{experiment_name} / "
        f"{platform_name} / "
        f"{workload_name} / "
        f"{run_dir.name}"
    )

    # --------------------------------------------------------
    # Full overview
    # --------------------------------------------------------

    if GENERATE_OVERVIEW:
        overview_path = (
            output_dir
            / f"{OUTPUT_NAME}.png"
        )

        save_plot(
            timeline=timeline,
            full_timeline=timeline,
            title=base_title,
            output_path=overview_path,
            num_nodes=num_nodes,
            window_start=0.0,
            window_end=max_time,
        )

        generated_count += 1

    # --------------------------------------------------------
    # 100,000-unit windows
    # --------------------------------------------------------

    if GENERATE_WINDOWS:
        time_digits = max(
            6,
            len(
                str(
                    int(
                        math.ceil(
                            max_time
                        )
                    )
                )
            ),
        )

        window_start = 0.0

        while window_start < max_time:
            window_end = min(
                window_start + WINDOW_SIZE,
                max_time,
            )

            window_timeline = timeline_for_window(
                timeline=timeline,
                window_start=window_start,
                window_end=window_end,
            )

            has_arrivals = window_has_arrivals(
                timeline=timeline,
                window_start=window_start,
                window_end=window_end,
            )

            # Do not create completely empty windows.
            if (
                window_timeline
                or has_arrivals
            ):
                start_text = str(
                    int(window_start)
                ).zfill(time_digits)

                end_text = str(
                    int(window_end)
                ).zfill(time_digits)

                output_path = (
                    output_dir
                    / (
                        f"{OUTPUT_NAME}_"
                        f"{start_text}-"
                        f"{end_text}.png"
                    )
                )

                title = (
                    f"{base_title} "
                    f"[{int(window_start)}"
                    f"–{int(window_end)}]"
                )

                save_plot(
                    timeline=window_timeline,
                    full_timeline=timeline,
                    title=title,
                    output_path=output_path,
                    num_nodes=num_nodes,
                    window_start=window_start,
                    window_end=window_end,
                )

                generated_count += 1

            window_start += WINDOW_SIZE

    return generated_count


# ============================================================
# Run discovery
# ============================================================

def discover_runs(workload_dir):
    records = []

    for run_dir in sorted(
        workload_dir.iterdir(),
        key=lambda path: natural_sort_key(
            path.name
        ),
    ):
        if not run_dir.is_dir():
            continue

        if (
            run_dir.name
            in IGNORED_DIRECTORY_NAMES
        ):
            continue

        node_log_path = find_run_node_log(
            run_dir
        )

        if node_log_path is None:
            continue

        records.append(
            (
                run_dir,
                node_log_path,
            )
        )

    return records


def generate_uid_gantt(uid_root):
    uid_root = resolve_uid_root(uid_root)

    generated_count = 0
    run_count = 0

    for experiment_dir in sorted(
        uid_root.iterdir(),
        key=lambda path: natural_sort_key(path.name),
    ):
        if not experiment_dir.is_dir():
            continue

        for platform_dir in sorted(
            experiment_dir.iterdir(),
            key=lambda path: natural_sort_key(path.name),
        ):
            if not platform_dir.is_dir():
                continue

            for workload_dir in sorted(
                platform_dir.iterdir(),
                key=lambda path: natural_sort_key(path.name),
            ):
                if not workload_dir.is_dir():
                    continue

                if workload_dir.name in IGNORED_DIRECTORY_NAMES:
                    continue

                runs = discover_runs(workload_dir)

                for run_dir, node_log_path in runs:
                    print(
                        "\nProcessing:",
                        run_dir,
                    )

                    generated_count += plot_run(
                        experiment_name=experiment_dir.name,
                        platform_name=platform_dir.name,
                        workload_name=workload_dir.name,
                        run_dir=run_dir,
                        node_log_path=node_log_path,
                    )

                    run_count += 1

    if run_count == 0:
        raise RuntimeError(
            "No run directories containing node_log.csv "
            f"were found in {uid_root}"
        )

    print(f"\nProcessed {run_count} runs.")
    print(f"Generated {generated_count} Gantt images.")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} results/<UID>"
        )

    generate_uid_gantt(
        sys.argv[1]
    )


if __name__ == "__main__":
    main()