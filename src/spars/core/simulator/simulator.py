import cProfile
import csv
import json
import math
import os
import pstats
import time
from bisect import bisect_left
from datetime import datetime
from numbers import Real
from typing import Any

from spars.core.experiments.defaults import PROJECT_ROOT
from spars.core.registry import resolve_run_components
from spars.core.logging import log_error, log_info, log_trace
from spars.core.planner import Planner
from spars.core.progress import emit_simulation_progress
from spars.core.output import log_output

from .jobs_manager import JobsManager
from .monitor import Monitor
from .platform_control import PlatformControl


Event = dict[str, Any]
EventGroup = dict[str, Any]

# Continuous RL decides on arrivals and job finishes. Power-transition
# completions deliberately do not call RL: they are consequences of the
# policy's own actions, and triggering on them lets an untrained policy cycle
# nodes off/on indefinitely.
EVENT_DRIVEN_RL_TYPES = frozenset({"continuous"})

# A run is stalled when jobs wait, nothing runs, and no job has started or
# finished for this many simulated seconds. ``run.stall_timeout`` changes it;
# None turns the check off.
DEFAULT_STALL_TIMEOUT = 100_000
_PROGRESS_EVENTS = frozenset({"execution_start", "execution_finished"})


_EVENT_PRIORITY = {
    "oracle": -1,
    "turn_on": 0,
    "turn_off": 1,
    "execution_finished": 2,
    "arrival": 3,
    "execution_start": 4,
    "switch_off": 5,
    "switch_on": 6,
}


def _event_priority(event: Event) -> float:
    return _EVENT_PRIORITY.get(event["type"], float("inf"))


# Input validation and path handling
def _resolve_data_path(
    path_value: str | os.PathLike[str],
    project_root: str,
) -> str:
    path = os.fspath(path_value)

    if os.path.isfile(path):
        return path

    normalized = path.replace("\\", "/")

    for prefix in ("workloads/", "platforms/"):
        marker_index = normalized.find(prefix)
        if marker_index == -1:
            continue

        candidate = os.path.join(
            project_root,
            "data",
            *normalized[marker_index:].split("/"),
        )
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Data file not found: {path_value}")


def _validated_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be an integer")

    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{field_name} must be an integer") from error

    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{field_name} must be an integer") from error

    if not math.isfinite(numeric_value) or numeric_value != integer_value:
        raise ValueError(f"{field_name} must be a finite integer")

    return integer_value


def _validated_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be numeric")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")

    return numeric_value


def parse_start_time(value: int | float | str) -> int:
    if isinstance(value, bool):
        raise TypeError("start_time must be a number, date string, or 'now'")

    if not isinstance(value, str):
        return _validated_integer(
            value,
            field_name="start_time",
        )

    if value.lower() == "now":
        return int(datetime.now().timestamp())

    return int(
        datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S",
        ).timestamp()
    )


def validate_workload(
    workload_info: Any,
    platform_node_count: int,
) -> None:
    """Validate workload structure before simulation state is created."""
    if not isinstance(workload_info, dict):
        raise TypeError("Workload file must contain a JSON object")

    jobs = workload_info.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Workload field 'jobs' must be a list")

    if "nb_res" not in workload_info:
        raise KeyError("Workload is missing required field 'nb_res'")

    max_requested_nodes = _validated_integer(
        workload_info["nb_res"],
        field_name="Workload nb_res",
    )
    if max_requested_nodes < 0:
        raise ValueError("Workload 'nb_res' cannot be negative")
    if max_requested_nodes > platform_node_count:
        raise RuntimeError(
            f"The workload needs {max_requested_nodes} nodes but the platform has {platform_node_count}: use a bigger platform."
        )

    required_job_fields = {
        "job_id",
        "res",
        "subtime",
        "reqtime",
        "runtime",
        "user_id",
    }
    seen_job_ids = set()

    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(f"Workload job at index {job_index} must be an object")

        missing_fields = required_job_fields - job.keys()
        if missing_fields:
            raise KeyError(
                f"Workload job at index {job_index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        job_id = _validated_integer(
            job["job_id"],
            field_name=f"Job at index {job_index} job_id",
        )
        _validated_integer(
            job["user_id"],
            field_name=f"Job {job_id} user_id",
        )

        if job_id in seen_job_ids:
            raise ValueError(f"Duplicate job_id in workload: {job_id}")
        seen_job_ids.add(job_id)

        requested_nodes = _validated_integer(
            job["res"],
            field_name=f"Job {job_id} res",
        )
        if requested_nodes <= 0:
            raise ValueError(f"Job {job_id} must request at least one node")
        if requested_nodes > platform_node_count:
            raise ValueError(
                f"Job {job_id} requests {requested_nodes} nodes, "
                f"but the platform has {platform_node_count}"
            )

        requested_time = _validated_number(
            job["reqtime"],
            field_name=f"Job {job_id} reqtime",
        )
        runtime = _validated_number(
            job["runtime"],
            field_name=f"Job {job_id} runtime",
        )
        _validated_number(
            job["subtime"],
            field_name=f"Job {job_id} subtime",
        )

        if requested_time <= 0:
            raise ValueError(
                f"Job {job_id} reqtime must be finite and greater than zero"
            )
        if runtime < 0:
            raise ValueError(f"Job {job_id} runtime must be finite and non-negative")


def _write_profile(
    profiler: cProfile.Profile,
    output_folder: str,
    top_n: int | None,
) -> None:
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    items = list(stats.stats.items())

    if top_n is not None:
        items = items[:top_n]

    csv_path = os.path.join(output_folder, "profile.csv")
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["function", "ncalls", "tottime_s", "cumtime_s"])

        for func, (_, ncalls, total_time, cumulative_time, _) in items:
            filename, line, function_name = func
            writer.writerow(
                [
                    f"{function_name} ({filename}:{line})",
                    ncalls,
                    f"{total_time:.6f}",
                    f"{cumulative_time:.6f}",
                ]
            )


class Simulator:
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        rl_kwargs: dict[str, Any] | None = None,
    ) -> "Simulator":
        paths = cfg["paths"]
        run = cfg["run"]
        rl = cfg["rl"]

        scheduler, power_management = resolve_run_components(run)

        project_root = str(PROJECT_ROOT)

        workload_path = _resolve_data_path(
            paths["workload"],
            project_root,
        )
        platform_path = _resolve_data_path(
            paths["platform"],
            project_root,
        )
        start_time = parse_start_time(run["start_time"])

        rl_overrides = rl_kwargs or {}
        rl_enabled = bool(rl["enabled"])
        learn = bool(rl_overrides.get("learn", rl["learn"]))
        rl_type = rl_overrides.get("rl_type", rl.get("type"))
        rl_dt = rl_overrides.get("rl_dt", rl.get("dt"))

        return cls(
            workload_path=workload_path,
            platform_path=platform_path,
            start_time=start_time,
            scheduler_name=scheduler["name"],
            scheduler_config=scheduler["config"],
            power_management_name=power_management["name"],
            power_management_config=power_management["config"],
            overrun_policy=run["overrun_policy"],
            rl=rl_enabled,
            learn=learn,
            rl_type=rl_type,
            rl_dt=rl_dt,
            force_wakeup=run["force_wakeup"],
            stall_timeout=run.get("stall_timeout", DEFAULT_STALL_TIMEOUT),
            output_dir=paths.get("output") or None,
        )

    def __init__(
        self,
        workload_path: str | os.PathLike[str],
        platform_path: str | os.PathLike[str],
        start_time: int | float,
        scheduler_name: str,
        scheduler_config: dict[str, Any],
        power_management_name: str,
        power_management_config: dict[str, Any],
        overrun_policy: str,
        rl: bool,
        learn: bool,
        rl_type: str | None,
        rl_dt: int | float | None,
        force_wakeup: bool,
        stall_timeout: int | float | None = DEFAULT_STALL_TIMEOUT,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        with open(workload_path, "r", encoding="utf-8") as file:
            self.workload_info = json.load(file)
        # Kept for components that rebuild a reference run of the same
        # workload (e.g. the oracle-referenced reward); not used here.
        self.workload_path = os.fspath(workload_path)
        self.platform_path = os.fspath(platform_path)
        self.overrun_policy = overrun_policy

        self.platform_control = PlatformControl(
            platform_path,
            overrun_policy,
            start_time,
        )
        machine = self.platform_control.machine

        validate_workload(self.workload_info, len(machine.nodes))

        self.monitor = Monitor(machine, start_time)
        self.current_time = start_time
        self.events = []
        self.is_running = False

        self.num_jobs = len(self.workload_info["jobs"])
        self.num_finished_jobs = 0
        self.jobs_manager = JobsManager()
        self.start_time = start_time

        self.planner = Planner(
            machines=machine,
            jobs_manager=self.jobs_manager,
            scheduler_name=scheduler_name,
            scheduler_config=scheduler_config,
            power_management_name=power_management_name,
            power_management_config=power_management_config,
            start_time=start_time,
            workload=self.workload_info,
            platform=machine.nodes,
            monitor=self.monitor,
            platform_control=self.platform_control,
            output_dir=output_dir,
        )

        self.rl = rl
        self.learn = learn
        self.rl_tick_scheduled = False

        if self.rl and rl_type is None:
            raise RuntimeError(
                "Select an RL_TYPE ('continuous' or 'discrete')"
            )

        if self.rl and rl_type not in {"continuous", "discrete"}:
            raise ValueError(
                "rl_type must be either 'continuous' or 'discrete'"
            )

        self.rl_type = rl_type

        normalized_rl_dt = rl_dt

        if self.rl_type == "discrete":
            if rl_dt is None:
                raise RuntimeError(
                    f"Discrete Time is required for RL_TYPE {self.rl_type}"
                )
            try:
                numeric_rl_dt = float(rl_dt)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError("rl_dt must be numeric") from error

            if not math.isfinite(numeric_rl_dt) or numeric_rl_dt <= 0:
                raise ValueError("rl_dt must be finite and greater than zero")

            if not isinstance(rl_dt, Real):
                normalized_rl_dt = numeric_rl_dt

        self.rl_dt = normalized_rl_dt
        self.force_wakeup = force_wakeup
        if stall_timeout is not None:
            stall_timeout = _validated_number(stall_timeout, field_name="stall_timeout")
            if stall_timeout <= 0:
                raise ValueError("stall_timeout must be greater than zero or None")
        self.stall_timeout = stall_timeout
        # Why the run was stopped early (None for a normal end), and since when
        # jobs have been waiting with nothing running (None: not waiting idle).
        self.stalled: str | None = None
        self._waiting_idle_since: float | None = None

        self.push_event(start_time, {"type": "simulation_start"})

        for workload_job in self.workload_info["jobs"]:
            job = dict(workload_job)
            job["type"] = "arrival"

            timestamp = job["subtime"] + start_time
            job["subtime"] = timestamp
            self.push_event(timestamp, job)

    def push_event(
        self,
        timestamp: int | float,
        event: Event,
    ) -> None:
        if not isinstance(event, dict):
            raise TypeError("event must be a dictionary")
        if "type" not in event:
            raise KeyError("event is missing required field 'type'")

        try:
            numeric_timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("event timestamp must be numeric") from error

        if not math.isfinite(numeric_timestamp):
            raise ValueError("event timestamp must be finite")
        if timestamp < self.current_time:
            raise ValueError("Cannot schedule events for past timestamps")

        events = self.events
        event_group = {
            "timestamp": timestamp,
            "events": [event],
        }

        if not events:
            events.append(event_group)
            return

        last_timestamp = events[-1]["timestamp"]
        if timestamp >= last_timestamp:
            if timestamp == last_timestamp:
                events[-1]["events"].append(event)
            else:
                events.append(event_group)
            return

        index = bisect_left(
            events,
            timestamp,
            key=lambda group: group["timestamp"],
        )

        if index < len(events) and events[index]["timestamp"] == timestamp:
            events[index]["events"].append(event)
        else:
            events.insert(index, event_group)

    def _schedule_rl_tick(self) -> None:
        if self.rl and self.rl_type == "discrete" and not self.rl_tick_scheduled:
            next_tick = self.current_time + self.rl_dt
            self.push_event(next_tick, {"type": "CALL_RL"})
            self.rl_tick_scheduled = True

    def _only_rl_calls_remain(self) -> bool:
        return bool(self.events) and all(
            event.get("type") == "CALL_RL"
            for event_group in self.events
            for event in event_group["events"]
        )

    def _should_force_wakeup(self) -> bool:
        if not self.force_wakeup or self.num_finished_jobs >= self.num_jobs:
            return False

        if self.rl_type == "continuous":
            return not self.events

        if self.rl_type != "discrete":
            return False

        current_event_is_rl_call = getattr(self, "event", {}).get("type") == "CALL_RL"
        return self._only_rl_calls_remain() and not current_event_is_rl_call

    def _force_wakeup_if_needed(self) -> None:
        nodes = self.platform_control.machine.nodes
        sleeping_nodes = []
        idle_count = 0

        for node_id, node in nodes.items():
            if node.get("state") == "sleeping":
                sleeping_nodes.append(node_id)
            elif node.get("state") == "active" and node.get("job_id") is None:
                idle_count += 1

        target_active_idle = min(
            len(nodes),
            sum(job["res"] for job in self.jobs_manager.waiting_queue),
        )
        nodes_needed = target_active_idle - idle_count

        if nodes_needed > 0:
            self.push_event(
                self.current_time,
                {
                    "type": "switch_on",
                    "nodes": sleeping_nodes[:nodes_needed],
                },
            )
            self.planner.invalidate("force_wakeup")

    def _log_event(self, event: Event) -> None:
        row = [f"[Time={self.current_time:.2f}]"]

        if "job_id" in event:
            row.append(f"job_id={event['job_id']}")

        if "type" in event:
            row.append(f"type={event['type']}")

        for key, value in event.items():
            if key in ("job_id", "type"):
                continue

            if key in ("start_time", "subtime") and isinstance(value, (float, int)):
                value = round(value, 2)

            row.append(f"{key}={value}")

        log_trace(" ".join(row))

    def _drain_current_time_events(
        self,
        events: list[Event],
    ) -> bool:
        added_events = False

        while self.events and self.events[0]["timestamp"] == self.current_time:
            event_group = self.events.pop(0)
            events.extend(event_group["events"])
            added_events = True

        return added_events

    def _push_event_buckets(
        self,
        event_buckets: list[EventGroup],
    ) -> None:
        for event_bucket in event_buckets:
            timestamp = event_bucket["timestamp"]
            for event in event_bucket["events"]:
                self.push_event(timestamp, event)

    def start_simulator(self) -> None:
        self.is_running = True
        emit_simulation_progress(0, self.num_jobs)
        self._schedule_rl_tick()

    def job_counts(self) -> dict[str, int]:
        """How many jobs finished, run, wait, or were not submitted yet."""
        waiting = len(self.jobs_manager.waiting_queue)
        running = len(self.jobs_manager.active_jobs)
        finished = self.num_finished_jobs
        return {
            "jobs": self.num_jobs,
            "finished_jobs": finished,
            "unfinished_jobs": self.num_jobs - finished,
            "running_jobs": running,
            "waiting_jobs": waiting,
            "not_submitted_jobs": self.num_jobs - finished - running - waiting,
        }

    def _stall(self, reason: str) -> None:
        counts = self.job_counts()
        self.stalled = (
            f"{reason} (simulated time {self.current_time:g} s). "
            f"{counts['unfinished_jobs']} of {counts['jobs']} jobs did not finish: "
            f"{counts['waiting_jobs']} waiting, {counts['running_jobs']} running, "
            f"{counts['not_submitted_jobs']} not submitted yet."
        )
        log_error("STALLED: %s The run was stopped early.", self.stalled)

    def _is_stalled(self, processed_events: list[Event]) -> bool:
        """Jobs wait, none runs, and none has started or finished for too long."""
        working = (
            not self.jobs_manager.waiting_queue
            or self.jobs_manager.active_jobs
            or any(event["type"] in _PROGRESS_EVENTS for event in processed_events)
        )
        if working:
            self._waiting_idle_since = None
            return False
        if self._waiting_idle_since is None:
            self._waiting_idle_since = self.current_time
        idle_for = self.current_time - self._waiting_idle_since
        if self.stall_timeout is None or idle_for < self.stall_timeout:
            return False
        self._stall(
            f"{len(self.jobs_manager.waiting_queue)} jobs wait and none has started "
            f"or finished for {idle_for:g} s"
        )
        return True

    def on_finish(self) -> EventGroup:
        self.is_running = False
        log_info(f"Simulation finished at time {self.current_time}.")
        self.monitor.on_finish()

        return {
            "now": self.current_time,
            "event_list": [
                {
                    "timestamp": self.current_time,
                    "events": [{"type": "simulation_finished"}],
                }
            ],
        }

    def _push_platform_result_events(
        self,
        result_events: list[dict[str, Any]],
    ) -> None:
        for result_event in result_events:
            self.push_event(
                result_event["timestamp"],
                result_event["event"],
            )

    def _start_job(
        self,
        event: Event,
        submitted_jobs: list[Event],
    ) -> None:
        if any(
            job["job_id"] == event["job_id"] for job in self.jobs_manager.active_jobs
        ):
            raise RuntimeError(f"Job {event['job_id']} is already executed")

        result = self.platform_control.compute(
            event["nodes"],
            event,
            self.current_time,
        )

        if result is None:
            raise RuntimeError(f"Job {event['job_id']} failed to execute")

        event["start_time"] = self.current_time
        submitted_jobs.append(event)

        finish_time, result_event = result
        self.jobs_manager.add_to_active_jobs(result_event)
        self.jobs_manager.remove_from_waiting_queue(event)
        self.push_event(finish_time, result_event)

    def _finish_job(
        self,
        event: Event,
        finished_jobs: list[Event],
    ) -> None:
        terminated = self.platform_control.release(
            event,
            self.current_time,
        )
        self.num_finished_jobs += 1
        emit_simulation_progress(
            self.num_finished_jobs,
            self.num_jobs,
        )
        event["terminated"] = terminated
        event["finish_time"] = self.current_time
        self.jobs_manager.remove_from_active_jobs(event)
        finished_jobs.append(event)

    def _process_event(
        self,
        event: Event,
        arrived_jobs: list[Event],
        submitted_jobs: list[Event],
        finished_jobs: list[Event],
    ) -> bool:
        """Process one event and report whether continuous RL must run."""
        self.event = event
        event_type = event["type"]

        if event_type == "switch_off":
            result_events = self.platform_control.switch_off(
                event["nodes"],
                self.current_time,
                oracle_durations=event.get("oracle_durations"),
            )
            self._push_platform_result_events(result_events)

        elif event_type == "turn_off":
            self.platform_control.turn_off(
                event["nodes"],
                self.current_time,
            )

        elif event_type == "switch_on":
            result_events = self.platform_control.switch_on(
                event["nodes"],
                self.current_time,
                oracle_durations=event.get("oracle_durations"),
            )
            self._push_platform_result_events(result_events)

        elif event_type == "turn_on":
            self.platform_control.turn_on(
                event["nodes"],
                self.current_time,
            )

        elif event_type == "arrival":
            arrived_jobs.append(event)
            self.jobs_manager.add_to_waiting_queue(event)
            return self.rl and self.rl_type in EVENT_DRIVEN_RL_TYPES

        elif event_type == "execution_start":
            self._start_job(event, submitted_jobs)

        elif event_type == "execution_finished":
            self._finish_job(event, finished_jobs)
            return self.rl and self.rl_type in EVENT_DRIVEN_RL_TYPES

        elif event_type == "change_dvfs_mode":
            self.platform_control.change_dvfs_mode(
                event["node"],
                event["mode"],
            )

        elif event_type == "CALL_RL":
            self.rl_tick_scheduled = False
            self._schedule_rl_tick()

        elif event_type == "oracle":
            for planned_event in event["plan"]:
                self.push_event(
                    float(planned_event["timestamp"]),
                    planned_event["event"],
                )

        return False

    def proceed(self) -> EventGroup:
        if self.num_finished_jobs == self.num_jobs:
            return self.on_finish()

        log_trace(f"Job remaining {self.num_jobs - self.num_finished_jobs}")

        if self._should_force_wakeup():
            self._force_wakeup_if_needed()

        # Force-wakeup may legitimately have nothing to enqueue (for example,
        # no sleeping nodes). Re-check before popping to avoid an IndexError.
        if not self.events and self.num_finished_jobs < self.num_jobs:
            self._stall(
                f"no events are left but {self.num_jobs - self.num_finished_jobs} "
                "jobs are unfinished"
            )
            return self.on_finish()

        event_group = self.events.pop(0)
        log_trace(
            "========== EXECUTE QUEUED EVENTS: time=%s ==========",
            event_group["timestamp"],
        )
        self.current_time = event_group["timestamp"]
        events: list[Event] = event_group["events"]

        self.monitor.record(
            mode="before",
            machine=self.platform_control.machine,
            current_time=self.current_time,
        )

        events.sort(key=_event_priority)

        processed_events: list[Event] = []
        arrived_jobs: list[Event] = []
        submitted_jobs: list[Event] = []
        finished_jobs: list[Event] = []
        need_rl = False

        while events:
            event = events.pop(0)
            processed_events.append(event)
            self._log_event(event)

            need_rl = (
                self._process_event(
                    event,
                    arrived_jobs,
                    submitted_jobs,
                    finished_jobs,
                )
                or need_rl
            )

            if self._drain_current_time_events(events):
                events.sort(key=_event_priority)

        self.monitor.record(
            mode="after",
            current_time=self.current_time,
            machine=self.platform_control.machine,
            record_job_arrival=arrived_jobs,
            record_job_submission=submitted_jobs,
            record_job_execution=finished_jobs,
        )
        self.monitor.note_processed_events(len(processed_events))

        if self.num_finished_jobs == self.num_jobs:
            return self.on_finish()

        if self._is_stalled(processed_events):
            return self.on_finish()

        if need_rl:
            self.push_event(
                self.current_time,
                {"type": "CALL_RL"},
            )

        return {
            "now": self.current_time,
            "event_list": [
                {
                    "timestamp": self.current_time,
                    "events": processed_events,
                }
            ],
        }

    def advance(self) -> None:
        previous_time = self.current_time

        while self.current_time == previous_time and self.is_running:
            group = self.proceed()

            if not self.is_running:
                break

            self.planner.update(
                current_time=self.current_time,
                push_event_buckets=self._push_event_buckets,
                processed_events=group["event_list"][0]["events"],
            )


def run_simulation(
    simulator: Simulator,
    output_folder: str,
    top_n: int | None = None,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    simulator.monitor.configure_state_hist_spill(
        output_folder=output_folder,
        flush_every=1000,
    )

    start_wall_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    simulator.start_simulator()
    while simulator.is_running:
        simulator.advance()
        simulator.monitor.flush_state_hist_if_safe()

    simulator.monitor.flush_state_hist_if_safe(force=True)
    profiler.disable()

    runtime_seconds = time.time() - start_wall_time

    log_output(simulator, output_folder)

    runtime_path = os.path.join(output_folder, "runtime_seconds.txt")
    with open(runtime_path, "w", encoding="utf-8") as file:
        file.write(f"{runtime_seconds}\n")

    _write_profile(
        profiler=profiler,
        output_folder=output_folder,
        top_n=top_n,
    )

    log_info(f"Simulation completed. Logs saved to: {output_folder}")
