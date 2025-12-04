
import warnings
import pandas as pd
import ast
import os

import os
import torch as T
import json
from typing import Any, Dict
from datetime import datetime
from SPARS.Logger import log_info
warnings.filterwarnings("ignore", category=FutureWarning)



def _to_int_series(s):
    # handle lists, strings like "3", floats like 3.0, and None
    return pd.to_numeric(s, errors="coerce").astype("Int64")  # nullable int


def _to_float_series(s):
    # unify time columns to float (seconds). If you use datetime, convert both sides to datetime64[ns] instead.
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_nodes(x):
    # handle NaN/None/empty strings
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple)) else v
        except Exception:
            return []
    return x  # as-is


def process_node_job_data(nodes_data, jobs):
    """Build intervals per node and attach job subtime as submission_time (floats kept)."""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4,
    }

    # --- node intervals ---
    node_intervals = []
    for node in (nodes_data or []):
        nid = node['id']
        current_dvfs = None
        for itv in node.get('state_history', []):
            if 'dvfs_mode' in itv:
                current_dvfs = itv['dvfs_mode']
            if itv['start_time'] < itv['finish_time']:
                node_intervals.append({
                    'node_id':    nid,
                    'state':      itv['state'],
                    'dvfs_mode':  current_dvfs,
                    'start_time': float(itv['start_time']),
                    'finish_time': float(itv['finish_time']),
                })

    node_intervals_df = pd.DataFrame(
        node_intervals,
        columns=['node_id', 'state', 'dvfs_mode', 'start_time', 'finish_time']
    )

    if node_intervals_df.empty:
        return pd.DataFrame(columns=[
            'dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id', 'terminated'
        ])

    # --- jobs exploded by node ---
    jobs_exploded = jobs.copy()

    # nodes "1 2 3" -> [1,2,3], then explode
    jobs_exploded['nodes'] = jobs_exploded['nodes'].map(parse_nodes)
    jobs_exploded = jobs_exploded.explode(
        'nodes').rename(columns={'nodes': 'node_id'})

    # keep times as float
    for c in ('start_time', 'finish_time', 'subtime'):
        if c in jobs_exploded.columns:
            jobs_exploded[c] = pd.to_numeric(
                jobs_exploded[c], errors='coerce').astype(float)

    # ensure essential cols exist minimally
    if 'terminated' not in jobs_exploded.columns:
        jobs_exploded['terminated'] = pd.NA
    if 'job_id' not in jobs_exploded.columns:
        jobs_exploded['job_id'] = -1

    # join ACTIVE intervals with jobs on (node_id, start_time, finish_time)
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    merged = pd.merge(
        active_df,
        jobs_exploded[['node_id', 'start_time', 'finish_time',
                       'subtime', 'job_id', 'terminated']],
        on=['node_id', 'start_time', 'finish_time'],
        how='left'
    )
    merged['submission_time'] = merged['subtime']  # carry from jobs
    merged.drop(columns=['subtime'], inplace=True)
    merged['job_id'] = merged['job_id'].fillna(-1)

    # non-active intervals: fill placeholders
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['submission_time'] = pd.NA
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['terminated'] = pd.NA

    combined = pd.concat([merged, non_active_df], ignore_index=True)

    # group nodes that share the same interval tuple
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id'],
        dropna=False
    ).agg(
        nodes=('node_id', lambda s: ' '.join(
            map(str, sorted(int(i) for i in s.dropna().tolist())))),
        terminated=('terminated', lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())
                    if s.notna().any() else pd.NA)
    ).reset_index()

    grouped = grouped.sort_values(['start_time', 'finish_time'])

    return grouped[['dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id', 'terminated']]


def build_waiting_time_df(jobs_execution_log: list) -> pd.DataFrame:
    """
    Convert jobs_execution_log (list of dict) into a DataFrame with:
    job_id, subtime, start_time, finish_time, waiting_time (start_time - subtime).

    Handles both numeric timestamps and datetime-like strings.
    """
    df = pd.DataFrame(jobs_execution_log)
    required = {'job_id', 'subtime', 'start_time', 'finish_time'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df['subtime']
    start = df['start_time']

    if not (pd.api.types.is_numeric_dtype(sub) and pd.api.types.is_numeric_dtype(start)):
        sub_dt = pd.to_datetime(sub, errors='coerce')
        start_dt = pd.to_datetime(start, errors='coerce')
        waiting = (start_dt - sub_dt).dt.total_seconds()
    else:
        waiting = start - sub

    out = df.loc[:, ['job_id', 'subtime', 'start_time', 'finish_time']].copy()
    out['waiting_time'] = waiting
    return out


def write_waiting_time_log(simulator, output_folder: str, filename: str = "waiting_time_log.csv") -> str:
    """
    Build waiting-time DataFrame from simulator.Monitor.jobs_execution_log
    and write it to <output_folder>/<filename>. Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    wt_df = build_waiting_time_df(simulator.Monitor.jobs_execution_log)
    path = os.path.join(output_folder, filename)
    wt_df.to_csv(path, index=False)
    return path


def build_energy_df(energy_log: list) -> pd.DataFrame:
    """
    Convert simulator.Monitor.energy (list[dict]) into a DataFrame with columns:
    id, energy_consumption, energy_effective, energy_waste.
    """
    df = pd.DataFrame(energy_log)
    required = {'id', 'energy_consumption', 'energy_effective', 'energy_waste'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in energy log: {sorted(missing)}")

    out = df.loc[:, ['id', 'energy_consumption',
                     'energy_effective', 'energy_waste']].copy()

    # (Optional) coerce to numeric in case inputs are strings
    for col in ['energy_consumption', 'energy_effective', 'energy_waste']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    return out


def write_energy_log(simulator, output_folder: str, filename: str = "energy_log.csv") -> str:
    """
    Build energy DataFrame from simulator.Monitor.energy and write it to CSV.
    Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    energy_df = build_energy_df(simulator.Monitor.energy)
    path = os.path.join(output_folder, filename)
    energy_df.to_csv(path, index=False)
    return path


def _sum_states_dur(states_dur: list) -> dict:
    """
    Sum durations across all nodes and all DVFS modes for each state bucket.
    Expected keys per node: active_idle, active_compute, switching_off, switching_on, sleeping.
    Returns a dict with totals for each bucket (float seconds).
    """
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

    for entry in states_dur:
        # each value is a dict of dvfs_mode -> duration
        for key, out_key in [
            ("active_idle", "total_active_idle"),
            ("active_compute", "total_active_compute"),
            ("switching_off", "total_switching_off"),
            ("switching_on", "total_switching_on"),
            ("sleeping", "total_sleeping"),
        ]:
            bucket = entry.get(key, {})
            if isinstance(bucket, dict):
                totals[out_key] += float(pd.to_numeric(pd.Series(bucket),
                                         errors="coerce").sum())

    totals["total_time_all_states"] = sum(totals.values())
    return totals


def build_metrics_df(jobs_execution_log: list, energy_log: list, states_dur: list | None = None) -> pd.DataFrame:
    """
    Return a 1-row DataFrame with:
      - total_waiting_time
      - mean_waiting_time
      - total_energy_waste
      - total_energy_consumption
      - energy_effective (= total_energy_consumption - total_energy_waste)
      - totals of node state durations aggregated over all nodes & dvfs:
        total_active_idle, total_active_compute, total_switching_off, total_switching_on,
        total_sleeping, total_time_all_states

    waiting_time is computed as start_time - subtime (seconds if datetimes).
    """
    # reuse existing builders
    wt_df = build_waiting_time_df(
        jobs_execution_log) if jobs_execution_log else pd.DataFrame(columns=["waiting_time"])
    en_df = build_energy_df(energy_log) if energy_log else pd.DataFrame(
        columns=["energy_waste"])

    # Waiting-time aggregates
    wt_series = pd.to_numeric(
        wt_df.get("waiting_time", pd.Series(dtype=float)), errors="coerce")
    total_waiting = wt_series.sum(min_count=1)
    mean_waiting = wt_series.mean() if not wt_series.empty else float("nan")

    # Energy aggregates
    waste_series = pd.to_numeric(
        en_df.get("energy_waste", pd.Series(dtype=float)), errors="coerce")
    total_waste = waste_series.sum(min_count=1)

    # Try several common column names for total consumption
    cons_col_candidates = ["energy_consumption",
                           "energy_total", "consumed_energy", "energy"]
    cons_series = None
    for col in cons_col_candidates:
        if col in en_df.columns:
            cons_series = pd.to_numeric(en_df[col], errors="coerce")
            break

    if cons_series is not None:
        total_consumption = cons_series.sum(min_count=1)
        energy_effective = total_consumption - \
            (total_waste if pd.notna(total_waste) else 0.0)
    else:
        total_consumption = float("nan")
        energy_effective = float("nan")

    # NaN-safe defaults to 0.0 for totals; keep mean as NaN if unavailable
    if pd.isna(total_waiting):
        total_waiting = 0.0
    if pd.isna(total_waste):
        total_waste = 0.0
    if pd.isna(total_consumption):
        total_consumption = 0.0
    if pd.isna(energy_effective):
        energy_effective = 0.0

    state_totals = _sum_states_dur(states_dur or [])

    row = {
        "total_waiting_time": float(total_waiting),
        "mean_waiting_time": float(mean_waiting) if pd.notna(mean_waiting) else 0.0,
        "total_energy_waste": float(total_waste),
        "total_energy_consumption": float(total_consumption),
        "energy_effective": float(energy_effective),
        **state_totals,
    }
    return pd.DataFrame([row])


def write_metrics_log(simulator, output_folder: str, filename: str = "metrics.csv") -> str:
    """
    Build metrics DataFrame and write it to <output_folder>/<filename>.
    """
    os.makedirs(output_folder, exist_ok=True)
    metrics_df = build_metrics_df(
        simulator.Monitor.jobs_execution_log,
        simulator.Monitor.energy,
        simulator.Monitor.states_dur,
    )
    path = os.path.join(output_folder, filename)
    metrics_df.to_csv(path, index=False)
    return path


def write_state_switch_csv(simulator, output_folder: str, filename: str = "state_switch.csv") -> str:
    """
    Save `state_switch` (list of dicts) to CSV with ordered columns:
    time, nb_sleeping, nb_switching_on, nb_switching_off, nb_idle, nb_computing.

    Returns the written filepath.
    """
    state_switch = simulator.Monitor.state_switch

    os.makedirs(output_folder, exist_ok=True)

    cols = ["time", "nb_sleeping", "nb_switching_on",
            "nb_switching_off", "nb_idle", "nb_computing"]
    df = pd.DataFrame(state_switch)

    # ensure all expected columns exist (missing -> NaN)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # order columns
    df = df[cols]

    # optional: coerce numeric columns (except 'time' if it's datetime-like strings)
    for c in cols:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    path = os.path.join(output_folder, filename)
    df.to_csv(path, index=False)


def log_output(simulator, output_folder):
    os.makedirs(f'{output_folder}', exist_ok=True)

    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv(f'{output_folder}/raw_node_log.csv', index=False)

    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv(f'{output_folder}/raw_job_log.csv', index=False)

    raw_terminated_log = pd.DataFrame(simulator.jobs_manager.terminated_jobs)
    raw_terminated_log.to_csv(f'{output_folder}/raw_terminated_log.csv', index=False)
    
    write_waiting_time_log(simulator, output_folder)
    write_energy_log(simulator, output_folder)
    write_metrics_log(simulator, output_folder)
    write_state_switch_csv(simulator, output_folder)

    node_log = process_node_job_data(
        simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv(f'{output_folder}/node_log.csv', index=False)

def _load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML or JSON config from a fixed path.
    If the file doesn't exist, raise (no silent fallback).
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml  # requires PyYAML
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def log_config_summary(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    run_cfg = cfg.get("run", {})
    rl_cfg = cfg.get("rl", {})
    log_cfg = cfg.get("logging", {})

    log_info("=== Config Summary ===")
    log_info(
        "Paths: workload=%s, platform=%s, output=%s",
        paths.get("workload"),
        paths.get("platform"),
        paths.get("output"),
    )
    log_info(
        "Run: algorithm=%s, overrun_policy=%s, start_time=%s, timeout=%s",
        run_cfg.get("algorithm"),
        run_cfg.get("overrun_policy"),
        run_cfg.get("start_time"),
        run_cfg.get("timeout"),
    )

    if rl_cfg.get("enabled"):
        log_info(
            "RL: ENABLED (type=%s, dt=%s, device=%s, epochs=%s, num_nodes=%s)",
            rl_cfg.get("type"),
            rl_cfg.get("dt"),
            rl_cfg.get("device"),
            rl_cfg.get("epochs"),
            rl_cfg.get("num_nodes"),
        )
        log_info(
            "RL: agent=%s, checkpoint=%s",
            rl_cfg.get("assign"),
            rl_cfg.get("checkpoint"),
        )
    else:
        log_info("RL: disabled")

    log_info(
        "Logging: level=%s, file=%s",
        log_cfg.get("level"),
        log_cfg.get("file"),
    )


# IMPORTANT: load Gym config BEFORE importing the env so monkey-patches apply


def _choose_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if T.cuda.is_available() else "cpu"
    return pref


def _parse_start_time(value) -> int:
    """
    Accepts:
      - int/float epoch
      - "now"
      - "YYYY-MM-DD HH:MM:SS"
    Returns epoch seconds (int).
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.lower() == "now":
            return int(datetime.now().timestamp())
        try:
            t = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return int(t.timestamp())
        except ValueError:
            raise ValueError(
                "run.start_time must be epoch int, 'now', or 'YYYY-MM-DD HH:MM:SS'"
            )
    raise TypeError("Unsupported start_time type")


# ---------------------------
# Helpers for flexible agent construction (ONLY addition)
# ---------------------------
def _load_object(spec: str):
    """Load 'pkg.mod:Obj' or 'pkg.mod.Obj' into a Python object."""
    import importlib
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        mod, _, name = spec.rpartition(".")
        if not mod:
            raise ValueError(f"Bad import path: {spec}")
    return getattr(importlib.import_module(mod), name)


def _instantiate_with_flexible_kwargs(cls, params: dict, *, positional_first: str | None = None):
    """
    Instantiate `cls` with kwargs in `params`. If the constructor needs a first positional
    argument (e.g., optimizer 'params'), set positional_first='params'.
    Filters unknown kwargs automatically when possible.
    """
    import inspect
    params = dict(params or {})

    def _call(p: dict):
        if positional_first and positional_first in p:
            pf = p.pop(positional_first)
            try:
                return cls(pf, **p)
            finally:
                p[positional_first] = pf
        return cls(**p)

    try:
        return _call(params)
    except TypeError:
        # Filter unknown kwargs unless ctor accepts **kwargs
        sig = None
        try:
            sig = inspect.signature(cls.__init__)
            has_varkw = any(
                a.kind == inspect.Parameter.VAR_KEYWORD for a in sig.parameters.values())
            if has_varkw:
                raise
            allowed = {k for k in sig.parameters if k != "self"}
            filtered = {k: v for k, v in params.items() if k in allowed}
            return _call(filtered)
        except Exception:
            raise


def _build_agent(rl_cfg: dict, device: str):
    """
    Build agent and optimizer ENTIRELY from cfg['rl']['agent'] with flexible params.
    - No hard-coded keys like obs_dim/act_dim are injected.
    - 'device' handling:
        * if agent.params.device == "auto" -> resolve with _choose_device
        * if agent.params.device missing   -> set to resolved device
        * if agent ctor doesn't accept 'device', it's filtered; if it's an nn.Module,
          we still move it to the device afterward.
    """
    agent_cfg = rl_cfg.get("agent") or {}

    # ----- Agent class -----
    AgentClass = _load_object(agent_cfg.get(
        "class", "RL_Agent.SPARS.agent:ActorCriticMLP"))
    params = dict(agent_cfg.get("params") or {})

    cfg_device = params.get("device", rl_cfg.get("device", "auto"))
    final_device = _choose_device(
        cfg_device if cfg_device is not None else "auto")

    if "device" not in params or str(params.get("device")).lower() == "auto":
        params["device"] = final_device

    model = _instantiate_with_flexible_kwargs(AgentClass, params)

    # Ensure nn.Module is moved even if ctor ignored 'device'
    import torch.nn as nn
    if isinstance(model, nn.Module):
        model.to(final_device)


    # ----- Optimizer -----
    opt_cfg = agent_cfg.get("optimizer") or {}
    OptClass = _load_object(opt_cfg.get("class", "torch.optim:Adam"))

    opt_params = dict(opt_cfg.get("params") or {})
    if "lr" not in opt_params and "learning_rate" in rl_cfg:
        opt_params["lr"] = float(rl_cfg["learning_rate"])

    optimizer = _instantiate_with_flexible_kwargs(
        OptClass,
        {"params": model.parameters() if hasattr(model, "parameters")
         else model, **opt_params},
        positional_first="params",
    )

    return model, optimizer
# ---------------------------


def get_action(model, obs):
    logits, V = model(obs)
    # print(logits)
    dist = T.distributions.Normal(logits, 0.02)
    # dist = T.distributions.Categorical(logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action, log_prob