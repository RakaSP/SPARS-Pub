"""Metadata-driven plots with no policy-name parsing or alias registry."""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# One colour per run, in this fixed order (checked with the dataviz palette
# validator; values are also printed on the marks, so colour is never alone).
RUN_COLORS = (
    "#2A78D6", "#EB6834", "#1BAF7A", "#EDA100",
    "#E87BA4", "#008300", "#6250D6", "#E34948",
)
# Time per node state, in the colours the Gantt chart uses for the same states.
STATE_SHARES = (
    ("Computing", "total_active_compute", "#D97A2B"),
    ("Idle", "total_active_idle", "#A9B7C6"),
    ("Switching on", "total_switching_on", "#27AE60"),
    ("Switching off", "total_switching_off", "#C0392B"),
    ("Sleeping", "total_sleeping", "#2C3E7A"),
)
NEUTRAL_COLOR = "#6B7C93"
INK = "#0B0B0B"
MUTED = "#52514E"
JOULES_PER_KWH = 3.6e6


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    states = {key: float(row.get(key) or 0.0) for _, key, _ in STATE_SHARES}
    return {
        "waiting_time": float(row["mean_waiting_time"]),
        "energy_waste": float(row["total_energy_waste"]),
        **states,
    }


def display_label(run: dict[str, Any]) -> str:
    label = str(run.get("display_name", run["name"]))
    timeout = run.get("timeout")
    suffix = f"{timeout:g} s" if timeout is not None else ""
    if suffix and f"{timeout:g}" in label:  # the name already says it
        suffix = ""
    variations = [
        f"{key.rsplit('.', 1)[-1]}={value}"
        for key, value in run.get("sweep", {}).items()
        if key
        not in {
            "paths.platform",
            "paths.workload",
            "run.power_management.config.timeout",
        }
    ]
    return "\n".join(part for part in (label, suffix, ", ".join(variations)) if part)


def _dominated(entries: Sequence[dict[str, Any]]) -> list[bool]:
    """True for a run that another run beats (or ties) on both waiting and waste."""
    def beats(other: dict[str, Any], run: dict[str, Any]) -> bool:
        return (
            other["waiting_time"] <= run["waiting_time"]
            and other["energy_waste"] <= run["energy_waste"]
            and (other["waiting_time"], other["energy_waste"])
            != (run["waiting_time"], run["energy_waste"])
        )

    return [any(beats(other, run) for other in entries if other is not run) for run in entries]


def _style(axis: "Axes", title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=9)


def plot_comparison(runs: Sequence[dict[str, Any]], output_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for run in runs:
        path = Path(run["output"]) / "metrics.csv"
        if path.is_file():
            entries.append({**run, **read_metrics(path)})
    if not entries:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = [display_label(run).replace("\n", " · ") for run in entries]
    # Colours are never reused: past 8 runs they all turn neutral and the
    # labels carry the identity.
    colors = [
        RUN_COLORS[i] if len(entries) <= len(RUN_COLORS) else NEUTRAL_COLOR
        for i in range(len(entries))
    ]
    waits = [run["waiting_time"] for run in entries]
    wastes = [run["energy_waste"] / JOULES_PER_KWH for run in entries]
    rows = range(len(entries))

    fig, ((wait_axis, point_axis), (waste_axis, state_axis)) = plt.subplots(
        2, 2, figsize=(14, max(7.0, 1.6 + 0.55 * len(entries) * 2))
    )

    for axis, values, title, unit in (
        (wait_axis, waits, "Mean waiting time", "s"),
        (waste_axis, wastes, "Wasted energy", "kWh"),
    ):
        axis.barh(rows, values, color=colors, height=0.6)
        axis.set_yticks(list(rows), labels)
        axis.invert_yaxis()
        top = max(values) or 1.0
        axis.set_xlim(0, top * 1.22)
        for row, value in zip(rows, values):
            axis.text(value + top * 0.015, row, f"{value:,.1f} {unit}" if unit == "kWh" else f"{value:,.0f} {unit}",
                      va="center", fontsize=9, color=INK)
        axis.set_xlabel(f"{unit}  ·  lower is better", color=MUTED, fontsize=9)
        _style(axis, title)
        axis.grid(axis="y", visible=False)

    # Trade-off: the point of a power-management policy.
    dominated = _dominated(entries)
    spots: dict[tuple[float, float], list[int]] = {}  # runs at the same point share a label
    for index, (x, y) in enumerate(zip(waits, wastes)):
        spots.setdefault((round(x, 3), round(y, 3)), []).append(index)
    for place, ((x, y), indices) in enumerate(sorted(spots.items())):
        first = indices[0]
        point_axis.scatter([x], [y], s=110, color=colors[first], edgecolors="white", linewidths=1.5, zorder=3)
        text = ", ".join(labels[i] for i in indices)
        if all(dominated[i] for i in indices):
            text += "  (worse on both)"
        point_axis.annotate(text, (x, y), xytext=(8, 6 if place % 2 == 0 else -14),
                            textcoords="offset points", fontsize=9, color=INK)
    span_x = (max(waits) - min(waits)) or max(waits) or 1.0
    span_y = (max(wastes) - min(wastes)) or max(wastes) or 1.0
    point_axis.set_xlim(min(waits) - span_x * 0.1, max(waits) + span_x * 0.45)
    point_axis.set_ylim(min(wastes) - span_y * 0.15, max(wastes) + span_y * 0.2)
    point_axis.set_xlabel("Mean waiting time (s)  ·  lower is better", color=MUTED, fontsize=9)
    point_axis.set_ylabel("Wasted energy (kWh)  ·  lower is better", color=MUTED, fontsize=9)
    _style(point_axis, "Trade-off: the best run is at the bottom left")

    # Where the node time went.
    totals = [sum(run[key] for _, key, _ in STATE_SHARES) or 1.0 for run in entries]
    left = [0.0] * len(entries)
    for name, key, color in STATE_SHARES:
        shares = [100.0 * run[key] / total for run, total in zip(entries, totals)]
        state_axis.barh(rows, shares, left=left, color=color, height=0.6, label=name,
                        edgecolor="white", linewidth=1.2)
        for row, (share, start) in enumerate(zip(shares, left)):
            if share >= 6:
                state_axis.text(start + share / 2, row, f"{share:.0f}%", ha="center", va="center",
                                fontsize=8, color="white" if name != "Idle" else INK)
        left = [start + share for start, share in zip(left, shares)]
    state_axis.set_yticks(list(rows), labels)
    state_axis.invert_yaxis()
    state_axis.set_xlim(0, 100)
    state_axis.set_xlabel("Share of node time (%)", color=MUTED, fontsize=9)
    _style(state_axis, "Where the node time went")
    state_axis.grid(axis="y", visible=False)
    state_axis.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    first = entries[0]
    phase = "evaluation" if first["phase"] in {"test", "run"} else "training"
    fig.suptitle(
        f"{Path(first['platform']).stem} / {Path(first['workload']).stem} / {phase}",
        x=0.01, ha="left", fontsize=14, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "metrics.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    (output / "experiments.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8"
    )
