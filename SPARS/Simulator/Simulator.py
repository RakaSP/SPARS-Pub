# SPARS/Simulator/Simulator.py
import cProfile, csv, json, os, pstats, time
from SPARS.Logger import log_info, log_trace
from bisect import bisect_left

from SPARS.Simulator.JobsManager import JobsManager
from SPARS.Simulator.Scheduler import Scheduler
from SPARS.Simulator.Monitor import Monitor
from SPARS.Simulator.PlatformControl import PlatformControl
from SPARS.Utils import log_output


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


class Simulator:
    @classmethod
    def from_config(cls, cfg: dict, rl_kwargs: dict | None = None):
        paths = cfg["paths"]
        run = cfg["run"]
        rl = cfg["rl"]
        start_time = run["start_time"]
        from datetime import datetime
        if isinstance(start_time, str):
            if start_time.lower() == "now":
                start_time = int(datetime.now().timestamp())
            else:
                start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            start_time = int(start_time)

        rl_enabled = bool(rl["enabled"])
        learn = bool((rl_kwargs or {}).get("learn", rl["learn"]))
        rl_type = (rl_kwargs or {}).get("rl_type", rl.get("type"))
        rl_dt = (rl_kwargs or {}).get("rl_dt", rl.get("dt"))

        return cls(
            workload_path=paths["workload"],
            platform_path=paths["platform"],
            start_time=start_time,
            algorithm=run["algorithm"],
            overrun_policy=run["overrun_policy"],
            rl=rl_enabled,
            learn=learn,
            rl_type=rl_type,
            rl_dt=rl_dt,
            algo_config=run["algo_config"],
            force_wakeup=run["force_wakeup"],
        )

    def __init__(self, workload_path, platform_path, start_time, algorithm,
                overrun_policy, rl, learn, rl_type, rl_dt, algo_config, force_wakeup):
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
        
        self.platform_control = PlatformControl(
            platform_path, overrun_policy, start_time)
        
        if self.workload_info['nb_res'] > len(self.platform_control.machine.nodes):
            raise RuntimeError(
                "Workload max requested node exceed number of nodes in platform.")
        self.monitor = Monitor(self.platform_control.machine, start_time)

        self.current_time = start_time
        self.events = []
        self.is_running = False

        self.num_jobs = len(self.workload_info['jobs'])
        self.num_finished_jobs = 0
        self.jobs_manager = JobsManager()
        self.start_time = start_time
        self.scheduler = Scheduler(
            machines=self.platform_control.machine,
            jobs_manager=self.jobs_manager,
            algorithm=algorithm,
            start_time=start_time,
            algo_config=algo_config,
            workload=self.workload_info,
            platform=self.platform_control.machine.nodes,
            monitor=self.monitor,
            platform_control=self.platform_control,
        )
        
        # RL
        self.rl = rl  # <- This means RL Enabled: True or False
        self.learn = learn
        
        self.rl_tick_scheduled = False
        if self.rl and rl_type is None:
            raise RuntimeError(
                "Select an RL_TYPE ('continuous' or 'discrete')")
        self.rl_type = rl_type
        if self.rl_type == 'discrete' and rl_dt is None:
            raise RuntimeError(
                "Discrete Time is required for RL_TYPE Discrete")
        self.rl_dt = rl_dt
        self.force_wakeup = force_wakeup

        # seed events
        self.push_event(start_time, {'type': 'simulation_start'})
        for job in self.workload_info['jobs']:
            job = dict(job)
            job['type'] = 'arrival'
            timestamp = job['subtime'] + start_time
            job['subtime'] = job['subtime'] + start_time
            self.push_event(timestamp, job)

    def push_event(self, timestamp, event):
        if timestamp < self.current_time:
            raise ValueError("Cannot schedule events for past timestamps")
        evs = self.events

        # Fast path: append or merge at end if in order
        if not evs:
            evs.append({'timestamp': timestamp, 'events': [event]})
            return
        last_ts = evs[-1]['timestamp']
        if timestamp >= last_ts:
            if timestamp == last_ts:
                evs[-1]['events'].append(event)
            else:
                evs.append({'timestamp': timestamp, 'events': [event]})
            return

        # Out-of-order insert: binary search without building a new ts list
        try:
            # Python 3.10+: bisect supports 'key'
            i = bisect_left(evs, timestamp, key=lambda e: e['timestamp'])
        except TypeError:
            # Python < 3.10: use a lightweight key-view
            class _TsView:
                __slots__ = ('seq',)
                def __init__(self, seq): self.seq = seq
                def __len__(self): return len(self.seq)
                def __getitem__(self, idx): return self.seq[idx]['timestamp']
            i = bisect_left(_TsView(evs), timestamp)

        if i < len(evs) and evs[i]['timestamp'] == timestamp:
            evs[i]['events'].append(event)
        else:
            evs.insert(i, {'timestamp': timestamp, 'events': [event]})

    def _schedule_first_rl_tick(self):
        if self.rl and self.rl_type == 'discrete' and not self.rl_tick_scheduled:
            next_tick = self.current_time + self.rl_dt
            self.push_event(next_tick, {'type': 'CALL_RL'})
            self.rl_tick_scheduled = True

    def _schedule_next_rl_tick(self):
        if self.rl and self.rl_type == 'discrete' and not self.rl_tick_scheduled:
            next_tick = self.current_time + self.rl_dt
            self.push_event(next_tick, {'type': 'CALL_RL'})
            self.rl_tick_scheduled = True

    def start_simulator(self):
        self.is_running = True
        self._schedule_first_rl_tick()
        
    def on_finish(self):
        self.is_running = False
        log_info(f"Simulation finished at time {self.current_time}.")
        self.monitor.on_finish()
        message = {'now': self.current_time, 'event_list': [
            {'timestamp': self.current_time, 'events': [{'type': 'simulation_finished'}]}]}
        return message

    def proceed(self):
        if self.num_finished_jobs == self.num_jobs:
            return self.on_finish()
        
        log_trace(f"Job remaining {self.num_jobs - self.num_finished_jobs}")
        
        only_rl_call_left = (
            len(self.events) > 0
            and all(
                event.get('type') == 'CALL_RL'
                for event_group in self.events
                for event in event_group['events']
            )
        )
        

        current_event_is_rl_call = (
            getattr(self, 'event', {}).get('type') == 'CALL_RL'
        )

        should_force_wakeup = (
            self.force_wakeup
            and self.num_finished_jobs < self.num_jobs
            and (
                (
                    self.rl_type == 'continuous'
                    and len(self.events) == 0
                )
                or (
                    self.rl_type == 'discrete'
                    and only_rl_call_left
                    and not current_event_is_rl_call
                )
            )
        )

        if should_force_wakeup:
            nodes = self.platform_control.machine.nodes
            sleeping_nodes = [nid for nid, node in nodes.items() if node.get("state") == "sleeping"]
            idle_count = sum(1 for node in nodes.values() if node.get("state") == "active" and node.get("job_id") is None)
            target_active_idle = min(len(nodes), sum(job["res"] for job in self.jobs_manager.waiting_queue))
            need = target_active_idle - idle_count

            if need > 0:
                self.push_event(self.current_time, {
                    "type": "switch_on",
                    "nodes": sleeping_nodes[:need],
                })
        elif len(self.events) == 0 and self.num_finished_jobs < self.num_jobs:
            return self.on_finish()
        
        # pop earliest events
        self.current_time, events = self.events.pop(0).values()

        self.monitor.record(mode='before', 
            machine=self.platform_control.machine,     
            current_time=self.current_time)

        # event ordering
        events = sorted(
            events,
            key=lambda e: _EVENT_PRIORITY.get(
                e["type"],
                float("inf"),
            ),
        )

        processed_events = []

        record_job_arrival = []
        record_job_submission = []
        record_job_execution = []

        need_rl = False

        while events:
            event = events.pop(0)
            processed_events.append(event)

            row = [f"[Time={self.current_time:.2f}]"]

            if "job_id" in event:
                row.append(f"job_id={event['job_id']}")

            if "type" in event:
                row.append(f"type={event['type']}")

            for key, value in event.items():
                if key in ("job_id", "type"):
                    continue

                if (
                    key in ("start_time", "subtime")
                    and isinstance(value, (float, int))
                ):
                    value = round(value, 2)

                row.append(f"{key}={value}")

            log_trace(" ".join(row))

            self.event = event
            etype = event["type"]

            if etype == 'switch_off':
                result_events = self.platform_control.switch_off(
                    event["nodes"],
                    self.current_time,
                    oracle_durations=event.get("oracle_durations"),
                )
                for ev in result_events:
                    self.push_event(ev['timestamp'], ev['event'])

            elif etype == 'turn_off':
                self.platform_control.turn_off(event['nodes'], self.current_time)
                # if self.rl and self.rl_type == 'continuous':
                #     need_rl = True

            elif etype == 'switch_on':
                result_events = self.platform_control.switch_on(
                    event["nodes"],
                    self.current_time,
                    oracle_durations=event.get("oracle_durations"),
                )
                for ev in result_events:
                    self.push_event(ev['timestamp'], ev['event'])

            elif etype == 'turn_on':
                self.platform_control.turn_on(event['nodes'], self.current_time)
                # if self.rl and self.rl_type == 'continuous':
                #     need_rl = True

            elif etype == 'arrival':
                record_job_arrival.append(event)
                self.jobs_manager.add_to_waiting_queue(event)
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'execution_start':
                if any(j['job_id'] == event['job_id'] for j in self.jobs_manager.active_jobs):
                    raise RuntimeError(
                        f"Job {event['job_id']} is already executed"
                    )

                result = self.platform_control.compute(
                    event['nodes'], event, self.current_time)
                if result is not None:
                    event['start_time'] = self.current_time
                    record_job_submission.append(event)
                    finish_time, _event = result
                    self.jobs_manager.add_to_active_jobs(_event)
                    self.jobs_manager.remove_from_waiting_queue(event)
                    self.push_event(finish_time, _event)
                else:
                    raise RuntimeError(
                        f"Job {event['job_id']} failed to execute"
                    )

            elif etype == 'execution_finished':
                terminated = self.platform_control.release(
                    event, self.current_time)
                self.num_finished_jobs += 1
                event['terminated'] = terminated
                event['finish_time'] = self.current_time
                self.jobs_manager.remove_from_active_jobs(event)
                record_job_execution.append(event)
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'change_dvfs_mode':
                _ = self.platform_control.change_dvfs_mode(
                    event['node'], event['mode'])

            elif etype == 'CALL_RL':
                self.rl_tick_scheduled = False
                self._schedule_next_rl_tick()
                
            elif etype == "oracle":
                for planned in event["plan"]:
                    self.push_event(
                        float(planned["timestamp"]),
                        planned["event"],
                    )
                    
            while (
                self.events
                and self.events[0]["timestamp"] == self.current_time
            ):
                same_time_group = self.events.pop(0)
                events.extend(same_time_group["events"])

            events.sort(
                key=lambda e: _EVENT_PRIORITY.get(
                    e["type"],
                    float("inf"),
                )
            )

        self.monitor.record(
            mode='after',
            current_time=self.current_time,
            machine=self.platform_control.machine,     
            record_job_arrival=record_job_arrival,
            record_job_submission=record_job_submission,
            record_job_execution=record_job_execution,
        )
        self.monitor.note_processed_events(
            len(processed_events)
        )

        if self.num_finished_jobs == self.num_jobs:
            message = self.on_finish()
            return message

        if need_rl:
            self.push_event(self.current_time, {'type': 'CALL_RL'})

        message = {
            "timestamp": self.current_time,
            "events": processed_events,
        }
        return {'now': self.current_time, 'event_list': [message]}

    def advance(self):
        now = self.current_time
        while self.current_time == now and self.is_running:
            self.proceed()

            scheduler_message = self.scheduler.schedule(
                self.current_time)

            log_trace(
                f"[{self.current_time}] Scheduler Message: {scheduler_message}")

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                for event in _data['events']:
                    self.push_event(timestamp, event)


def run_simulation(simulator, output_folder, top_n=None):
    os.makedirs(output_folder, exist_ok=True)
    simulator.monitor.configure_state_hist_spill(
        output_folder=output_folder,
        flush_every=1000,
    )

    # Wall-clock runtime for the whole simulation
    t0 = time.time()

    prof = cProfile.Profile()
    prof.enable()

    simulator.start_simulator()
    while simulator.is_running:
        simulator.advance()
        simulator.monitor.flush_state_hist_if_safe()
        
    simulator.monitor.flush_state_hist_if_safe(
        force=True,
    )

    prof.disable()

    t1 = time.time()
    runtime_s = t1 - t0

    log_output(simulator, output_folder)

    # Save runtime_seconds.txt
    runtime_path = os.path.join(output_folder, "runtime_seconds.txt")
    with open(runtime_path, "w") as f:
        f.write(f"{runtime_s}\n")

    # Save profiling CSV
    stats = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)

    csv_path = os.path.join(output_folder, "profile.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "ncalls", "tottime_s", "cumtime_s"])

        items = list(stats.stats.items())
        if top_n is not None:
            items = items[:top_n]

        for func, (cc, nc, tt, ct, callers) in items:
            filename, line, fn = func
            w.writerow([f"{fn} ({filename}:{line})", nc, f"{tt:.6f}", f"{ct:.6f}"])

    log_info(f"Simulation completed. Logs saved to: {output_folder}")