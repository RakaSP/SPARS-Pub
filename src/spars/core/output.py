import ast
import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from spars.core.registry import get_run_timeout, resolve_run_components
from spars.core.logging import log_info

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator


# Shared data contracts and output schemas
_JobRecord = Mapping[str, Any]
_NodeHistory = Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]]
_EnergyLog = Mapping[int, Mapping[str, float]]
_StateDurations = Mapping[int, Mapping[str, Mapping[str, float]]]

_NODE_LOG_COLUMNS: tuple[str, ...] = (
    "dvfs_mode",
    "state",
    "submission_time",
    "start_time",
    "finish_time",
    "nodes",
    "job_id",
    "terminated",
)
_JOB_TIMING_COLUMNS: tuple[str, ...] = (
    "job_id",
    "subtime",
    "start_time",
    "finish_time",
)
_ENERGY_LOG_COLUMNS: tuple[str, ...] = (
    "id",
    "energy_consumption",
    "energy_effective",
    "energy_waste",
)
_STATE_DURATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("active_idle", "total_active_idle"),
    ("active_compute", "total_active_compute"),
    ("switching_off", "total_switching_off"),
    ("switching_on", "total_switching_on"),
    ("sleeping", "total_sleeping"),
)
_STATE_SWITCH_COLUMNS: tuple[str, ...] = (
    "time",
    "nb_sleeping",
    "nb_switching_on",
    "nb_switching_off",
    "nb_idle",
    "nb_computing",
)


# Node and job processing
def parse_nodes(nodes_value: Any) -> Any:
    if nodes_value is None or (
        isinstance(nodes_value, float) and pd.isna(nodes_value)
    ):
        return []

    if isinstance(nodes_value, (list, tuple)):
        return list(nodes_value)

    if isinstance(nodes_value, str):
        nodes_text = nodes_value.strip()
        if not nodes_text:
            return []

        try:
            parsed_nodes = ast.literal_eval(nodes_text)
            return (
                list(parsed_nodes)
                if isinstance(parsed_nodes, (list, tuple))
                else parsed_nodes
            )
        except (ValueError, SyntaxError):
            return []

    return nodes_value


def _build_node_intervals(
    nodes_data: _NodeHistory | None,
) -> pd.DataFrame:
    node_intervals = []

    for node_id, node in (nodes_data or {}).items():
        current_dvfs = None

        for interval in node.get("state_history", []):
            if "dvfs_mode" in interval:
                current_dvfs = interval["dvfs_mode"]

            if interval["start_time"] < interval["finish_time"]:
                node_intervals.append({
                    "node_id": node_id,
                    "state": interval["state"],
                    "dvfs_mode": current_dvfs,
                    "start_time": float(interval["start_time"]),
                    "finish_time": float(interval["finish_time"]),
                })

    return pd.DataFrame(
        node_intervals,
        columns=[
            "node_id",
            "state",
            "dvfs_mode",
            "start_time",
            "finish_time",
        ],
    )


def _prepare_jobs_for_node_merge(
    jobs: pd.DataFrame,
) -> pd.DataFrame:
    jobs_by_node_df = jobs.copy()
    jobs_by_node_df["nodes"] = jobs_by_node_df["nodes"].map(parse_nodes)
    jobs_by_node_df = jobs_by_node_df.explode("nodes").rename(
        columns={"nodes": "node_id"}
    )

    for column in ("start_time", "finish_time", "subtime"):
        if column in jobs_by_node_df.columns:
            jobs_by_node_df[column] = pd.to_numeric(
                jobs_by_node_df[column],
                errors="coerce",
            ).astype(float)

    if "terminated" not in jobs_by_node_df.columns:
        jobs_by_node_df["terminated"] = pd.NA

    if "job_id" not in jobs_by_node_df.columns:
        jobs_by_node_df["job_id"] = -1

    return jobs_by_node_df


def process_node_job_data(
    nodes_data: _NodeHistory | None,
    jobs: pd.DataFrame,
) -> pd.DataFrame:
    """Build node-state intervals and attach matching job submission times."""
    non_active_job_ids = {
        "switching_off": -2,
        "switching_on": -3,
        "sleeping": -4,
    }

    node_intervals_df = _build_node_intervals(nodes_data)

    if node_intervals_df.empty:
        return pd.DataFrame(columns=list(_NODE_LOG_COLUMNS))

    if jobs.empty:
        # No job ever ran (a stalled run): there is nothing to attach.
        jobs = pd.DataFrame(columns=[
            "nodes", "start_time", "finish_time", "subtime", "job_id", "terminated",
        ])
    jobs_by_node_df = _prepare_jobs_for_node_merge(jobs)

    active_intervals_df = node_intervals_df[
        node_intervals_df["state"] == "active"
    ].copy()
    active_with_jobs_df = pd.merge(
        active_intervals_df,
        jobs_by_node_df[[
            "node_id",
            "start_time",
            "finish_time",
            "subtime",
            "job_id",
            "terminated",
        ]],
        on=["node_id", "start_time", "finish_time"],
        how="left",
    )
    active_with_jobs_df["submission_time"] = active_with_jobs_df["subtime"]
    active_with_jobs_df = active_with_jobs_df.drop(columns=["subtime"])
    active_with_jobs_df["job_id"] = active_with_jobs_df["job_id"].fillna(-1)

    non_active_intervals_df = node_intervals_df[
        node_intervals_df["state"] != "active"
    ].copy()
    non_active_intervals_df["submission_time"] = float("nan")
    non_active_intervals_df["job_id"] = (
        non_active_intervals_df["state"]
        .map(non_active_job_ids)
        .fillna(-1)
    )
    non_active_intervals_df["terminated"] = pd.NA

    combined_intervals_df = pd.concat(
        [active_with_jobs_df, non_active_intervals_df],
        ignore_index=True,
    )

    grouped_intervals_df = combined_intervals_df.groupby(
        [
            "state",
            "dvfs_mode",
            "submission_time",
            "start_time",
            "finish_time",
            "job_id",
        ],
        dropna=False,
    ).agg(
        nodes=(
            "node_id",
            lambda node_ids: " ".join(
                map(
                    str,
                    sorted(
                        int(node_id)
                        for node_id in node_ids.dropna().tolist()
                    ),
                )
            ),
        ),
        terminated=(
            "terminated",
            lambda terminated_values: (
                bool(
                    pd.Series(
                        terminated_values,
                        dtype="boolean",
                    )
                    .fillna(False)
                    .any()
                )
                if terminated_values.notna().any()
                else pd.NA
            ),
        ),
    ).reset_index()

    grouped_intervals_df = grouped_intervals_df.sort_values(
        ["start_time", "finish_time"]
    )

    return grouped_intervals_df.loc[:, list(_NODE_LOG_COLUMNS)]


# Waiting-time reporting
def build_waiting_time_df(
    jobs_execution_log: Sequence[_JobRecord],
) -> pd.DataFrame:
    if not jobs_execution_log:
        # No job finished (a stalled run): an empty table, not an error.
        return pd.DataFrame(columns=[*_JOB_TIMING_COLUMNS, "waiting_time"])

    jobs_df = pd.DataFrame(jobs_execution_log)
    missing_columns = set(_JOB_TIMING_COLUMNS) - set(jobs_df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    submission_times = jobs_df["subtime"]
    start_times = jobs_df["start_time"]

    if not (
        pd.api.types.is_numeric_dtype(submission_times)
        and pd.api.types.is_numeric_dtype(start_times)
    ):
        submission_datetimes = pd.to_datetime(
            submission_times,
            errors="coerce",
        )
        start_datetimes = pd.to_datetime(
            start_times,
            errors="coerce",
        )
        waiting_times = (
            start_datetimes - submission_datetimes
        ).dt.total_seconds()
    else:
        waiting_times = start_times - submission_times

    waiting_time_df = jobs_df.loc[:, list(_JOB_TIMING_COLUMNS)].copy()
    waiting_time_df["waiting_time"] = waiting_times
    return waiting_time_df


def build_unfinished_waiting_df(
    unfinished_jobs: Sequence[_JobRecord],
    end_time: float,
) -> pd.DataFrame:
    """Jobs that were submitted but never started.

    They have waited until the run ended, so that is their waiting time (a lower
    bound: they would have waited longer). Leaving them out would make a run
    that starves jobs look faster than one that serves everybody.
    """
    columns = [*_JOB_TIMING_COLUMNS, "waiting_time"]
    if not unfinished_jobs:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(unfinished_jobs).loc[:, ["job_id", "subtime"]]
    # float NaN (not pd.NA): the columns stay float, so joining the finished jobs keeps their dtype
    df["start_time"] = float("nan")
    df["finish_time"] = float("nan")
    df["waiting_time"] = (end_time - df["subtime"]).clip(lower=0)
    return df.loc[:, columns]


def write_waiting_time_log(
    simulator: "Simulator",
    output_folder: str,
    filename: str = "waiting_time_log.csv",
) -> str:
    os.makedirs(output_folder, exist_ok=True)
    waiting_time_df = build_waiting_time_df(
        simulator.monitor.jobs_execution_log
    )
    unfinished_df = build_unfinished_waiting_df(
        simulator.jobs_manager.waiting_queue, simulator.current_time
    )
    if not unfinished_df.empty:
        waiting_time_df = pd.concat([waiting_time_df, unfinished_df], ignore_index=True)
    output_path = os.path.join(output_folder, filename)
    waiting_time_df.to_csv(output_path, index=False)
    return output_path


# Energy reporting
def build_energy_df(energy_log: _EnergyLog) -> pd.DataFrame:
    energy_df = pd.DataFrame.from_dict(energy_log, orient="index")
    energy_df.index.name = "id"
    energy_df = energy_df.reset_index()

    missing_columns = set(_ENERGY_LOG_COLUMNS) - set(energy_df.columns)
    if missing_columns:
        raise KeyError(
            "Missing required columns in energy log: "
            f"{sorted(missing_columns)}"
        )

    result_df = energy_df.loc[:, list(_ENERGY_LOG_COLUMNS)].copy()

    for column in _ENERGY_LOG_COLUMNS[1:]:
        result_df[column] = pd.to_numeric(
            result_df[column],
            errors="coerce",
        )

    return result_df


def write_energy_log(
    simulator: "Simulator",
    output_folder: str,
    filename: str = "energy_log.csv",
) -> str:
    os.makedirs(output_folder, exist_ok=True)
    energy_df = build_energy_df(simulator.monitor.energy)
    output_path = os.path.join(output_folder, filename)
    energy_df.to_csv(output_path, index=False)
    return output_path


# Metrics
def _sum_states_dur(states_dur: _StateDurations) -> dict[str, float]:
    totals = {
        "total_active_idle": 0.0,
        "total_active_compute": 0.0,
        "total_switching_off": 0.0,
        "total_switching_on": 0.0,
        "total_sleeping": 0.0,
    }

    if not states_dur:
        totals["total_time_all_states"] = 0.0
        return totals

    for state_durations in states_dur.values():
        for state_name, total_name in _STATE_DURATION_FIELDS:
            durations_by_dvfs = state_durations.get(state_name, {})

            if isinstance(durations_by_dvfs, dict):
                totals[total_name] += float(
                    pd.to_numeric(
                        pd.Series(durations_by_dvfs),
                        errors="coerce",
                    ).sum()
                )

    totals["total_time_all_states"] = sum(totals.values())
    return totals


def build_metrics_df(
    jobs_execution_log: Sequence[_JobRecord],
    energy_log: _EnergyLog,
    states_dur: _StateDurations | None = None,
    unfinished_waiting_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    waiting_time_df = (
        build_waiting_time_df(jobs_execution_log)
        if jobs_execution_log
        else pd.DataFrame(columns=["waiting_time"])
    )
    if unfinished_waiting_df is not None and not unfinished_waiting_df.empty:
        waiting_time_df = pd.concat(
            [waiting_time_df, unfinished_waiting_df], ignore_index=True
        )
    energy_df = (
        build_energy_df(energy_log)
        if energy_log
        else pd.DataFrame(columns=["energy_waste"])
    )

    waiting_time_series = pd.to_numeric(
        waiting_time_df.get("waiting_time", pd.Series(dtype=float)),
        errors="coerce",
    )
    total_waiting = waiting_time_series.sum(min_count=1)
    mean_waiting = waiting_time_series.mean()

    energy_waste_series = pd.to_numeric(
        energy_df.get("energy_waste", pd.Series(dtype=float)),
        errors="coerce",
    )
    total_waste = energy_waste_series.sum(min_count=1)

    energy_consumption_series = pd.to_numeric(
        energy_df.get("energy_consumption", pd.Series(dtype=float)),
        errors="coerce",
    )
    total_consumption = energy_consumption_series.sum(min_count=1)
    energy_effective = (
        total_consumption
        - (total_waste if pd.notna(total_waste) else 0.0)
    )

    if pd.isna(total_waiting):
        total_waiting = 0.0
    if pd.isna(total_waste):
        total_waste = 0.0
    if pd.isna(total_consumption):
        total_consumption = 0.0
    if pd.isna(energy_effective):
        energy_effective = 0.0

    state_totals = _sum_states_dur(states_dur or {})

    metrics = {
        "total_waiting_time": float(total_waiting),
        "mean_waiting_time": (
            float(mean_waiting)
            if pd.notna(mean_waiting)
            else 0.0
        ),
        "total_energy_waste": float(total_waste),
        "total_energy_consumption": float(total_consumption),
        "energy_effective": float(energy_effective),
        **state_totals,
    }

    return pd.DataFrame([metrics])


def write_metrics_log(
    simulator: "Simulator",
    output_folder: str,
    filename: str = "metrics.csv",
) -> str:
    os.makedirs(output_folder, exist_ok=True)

    metrics_df = build_metrics_df(
        simulator.monitor.jobs_execution_log,
        simulator.monitor.energy,
        simulator.monitor.states_dur,
        build_unfinished_waiting_df(
            simulator.jobs_manager.waiting_queue, simulator.current_time
        ),
    )

    output_path = os.path.join(output_folder, filename)
    metrics_df.to_csv(output_path, index=False)
    return output_path


# State-switch reporting
def write_state_switch_csv(
    simulator: "Simulator",
    output_folder: str,
    filename: str = "state_switch.csv",
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    state_switch_df = pd.DataFrame(simulator.monitor.state_switch)

    for column in _STATE_SWITCH_COLUMNS:
        if column not in state_switch_df.columns:
            state_switch_df[column] = pd.NA

    state_switch_df = state_switch_df.loc[
        :,
        list(_STATE_SWITCH_COLUMNS),
    ]

    for column in _STATE_SWITCH_COLUMNS[1:]:
        state_switch_df[column] = pd.to_numeric(
            state_switch_df[column],
            errors="coerce",
        )

    output_path = os.path.join(output_folder, filename)
    state_switch_df.to_csv(output_path, index=False)


# Simulation output orchestration
def log_output(
    simulator: "Simulator",
    output_folder: str,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    raw_job_log_df = pd.DataFrame(
        simulator.monitor.jobs_execution_log
    )
    raw_job_log_df.to_csv(
        f"{output_folder}/raw_job_log.csv",
        index=False,
    )

    if getattr(simulator, "stalled", None):
        with open(f"{output_folder}/stalled.json", "w", encoding="utf-8") as stream:
            json.dump(
                {"reason": simulator.stalled, **simulator.job_counts()},
                stream,
                indent=2,
            )

    unfinished_jobs_df = pd.DataFrame(
        simulator.jobs_manager.waiting_queue
    )
    unfinished_jobs_df.to_csv(
        f"{output_folder}/unfinished_jobs_log.csv",
        index=False,
    )

    write_waiting_time_log(simulator, output_folder)
    write_energy_log(simulator, output_folder)
    write_metrics_log(simulator, output_folder)
    write_state_switch_csv(simulator, output_folder)

    simulator.monitor.flush_state_hist_if_safe(force=True)

    complete_state_history = (
        simulator.monitor.build_complete_states_hist()
    )
    node_log_df = process_node_job_data(
        complete_state_history,
        raw_job_log_df,
    )
    node_log_df.to_csv(
        f"{output_folder}/node_log.csv",
        index=False,
    )

    del complete_state_history
    del node_log_df


# Configuration logging
def log_config_summary(cfg: Mapping[str, Any]) -> None:
    paths = cfg.get("paths", {})
    run_config = cfg.get("run", {})
    rl_config = cfg.get("rl", {})
    logging_config = cfg.get("logging", {})
    scheduler, _ = resolve_run_components(dict(run_config))

    log_info("=== Config Summary ===")

    log_info(
        "Paths: workload=%s, platform=%s, output=%s",
        paths.get("workload"),
        paths.get("platform"),
        paths.get("output"),
    )

    log_info(
        "Run: scheduler=%s, overrun_policy=%s, start_time=%s, timeout=%s",
        scheduler["name"],
        run_config.get("overrun_policy"),
        run_config.get("start_time"),
        get_run_timeout(dict(run_config)),
    )

    if rl_config.get("enabled"):
        log_info(
            "RL: ENABLED (type=%s, dt=%s, device=%s, epochs=%s)",
            rl_config.get("type"),
            rl_config.get("dt"),
            rl_config.get("device"),
            rl_config.get("epochs"),
        )

        log_info(
            "RL: agent=%s, checkpoint=%s",
            rl_config.get("assign"),
            rl_config.get("checkpoint"),
        )
    else:
        log_info("RL: disabled")

    log_info(
        "Logging: level=%s, file=%s",
        logging_config.get("level"),
        logging_config.get("file"),
    )
