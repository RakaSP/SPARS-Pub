# SPARS/Simulator/Simulator.py
import json
from SPARS.Logger import log_info, log_trace
from math import ceil
from bisect import bisect_left

from SPARS.Simulator.JobsManager import JobsManager
from SPARS.Simulator.Scheduler import Scheduler
from SPARS.Simulator.Monitor import Monitor
from SPARS.Simulator.PlatformControl import PlatformControl
from SPARS.Utils import log_output


_EVENT_PRIORITY = {
    'turn_on': 0,
    'turn_off': 1,
    'execution_finished': 2,
    'execution_start': 3,
    'arrival': 4,
    'switch_off': 5,
    'switch_on': 6,
}


class Simulator:
    @classmethod
    def from_config(cls, cfg: dict,  rl_kwargs: dict | None = None):
        paths = cfg["paths"]
        run = cfg["run"]
        rl = cfg.get("rl", {})
        start_time = run.get("start_time", 0)
        from datetime import datetime
        if isinstance(start_time, str):
            if start_time.lower() == "now":
                start_time = int(datetime.now().timestamp())
            else:
                start_time = int(datetime.strptime(
                    start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            start_time = int(start_time)

        rl_enabled = bool(rl.get("enabled", False))
        rl_type = (rl_kwargs or {}).get(
            "rl_type", rl.get("type")) if rl_enabled else None
        rl_dt = (rl_kwargs or {}).get("rl_dt", rl.get(
            "dt")) if rl_type == "discrete" else None

        return cls(
            workload_path=paths["workload"],
            platform_path=paths["platform"],
            start_time=start_time,
            algorithm=run["algorithm"],
            overrun_policy=run.get("overrun_policy", "continue"),
            rl=rl_enabled,
            rl_type=rl_type,
            rl_dt=rl_dt,
            timeout=run.get("timeout", None),
        )

    def __init__(self, workload_path, platform_path, start_time, algorithm,
                 overrun_policy, rl=False, rl_type=None, rl_dt=None, timeout=None):
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)

        self.PlatformControl = PlatformControl(
            self.platform_info, overrun_policy, start_time)
        self.Monitor = Monitor(self.platform_info, start_time)

        self.current_time = start_time
        self.events = []
        self.is_running = False

        self.num_jobs = len(self.workload_info['jobs'])
        self.num_finished_jobs = 0
        self.jobs_manager = JobsManager()
        self.start_time = start_time
        self.scheduler = Scheduler(
            self.PlatformControl.machines,
            self.jobs_manager,
            algorithm,
            start_time,
            timeout
        )

        # RL
        self.rl = rl  # <- This means RL Enabled: True or False
        self.rl_tick_scheduled = False
        if self.rl and rl_type is None:
            raise RuntimeError(
                "Select an RL_TYPE ('continuous' or 'discrete')")
        self.rl_type = rl_type
        if self.rl_type == 'discrete' and rl_dt is None:
            raise RuntimeError(
                "Discrete Time is required for RL_TYPE Discrete")
        self.rl_dt = rl_dt

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

    # ---- RL tick helpers (discrete) ----
    def _schedule_first_rl_tick(self):
        if self.rl and self.rl_type == 'discrete':
            # next multiple of rl_dt at or after current_time
            next_tick = ((self.current_time + self.rl_dt - 1) //
                         self.rl_dt) * self.rl_dt
            self.push_event(next_tick, {'type': 'CALL_RL'})

    def _schedule_next_rl_tick(self):
        if self.rl and self.rl_type == 'discrete' and self.rl_tick_scheduled == False:
            # strictly after current_time
            next_tick = ((self.current_time // self.rl_dt) + 1) * self.rl_dt
            self.push_event(next_tick, {'type': 'CALL_RL'})
            self.rl_tick_scheduled = True

    def start_simulator(self):
        self.is_running = True
        # self._schedule_first_rl_tick()

    def on_finish(self):
        self.is_running = False
        log_info(f"Simulation finished at time {self.current_time}.")
        self.jobs_manager.on_finish()
        self.Monitor.on_finish()
        message = {'now': self.current_time, 'event_list': [
            {'timestamp': self.current_time, 'events': [{'type': 'simulation_finished'}]}]}
        return message

    def proceed(self):
        if len(self.events) == 0:
            message = self.on_finish()
            return message

        # pop earliest events
        self.current_time, events = self.events.pop(0).values()

        # schedule next RL discrete tick (aligned to grid)
        self._schedule_next_rl_tick()

        # logging (compact)
        for e in events:
            row = [f"[Time={self.current_time:.2f}]"]
            if "job_id" in e:
                row.append(f"job_id={e['job_id']}")
            if "type" in e:
                row.append(f"type={e['type']}")
            for k, v in e.items():
                if k in ("job_id", "type"):
                    continue
                if k in ("start_time", "subtime") and isinstance(v, (float, int)):
                    v = round(v, 2)
                row.append(f"{k}={v}")
            log_trace(" ".join(row))

        self.Monitor.record(mode='before', current_time=self.current_time)

        # deterministic event ordering
        events = sorted(events, key=lambda e: _EVENT_PRIORITY.get(
            e['type'], float('inf')))

        record_job_arrival = []
        record_job_submission = []
        record_job_execution = []

        need_rl = False
        for event in events:
            self.event = event
            etype = event['type']

            if etype == 'switch_off':
                result_events = self.PlatformControl.switch_off(
                    event['nodes'], self.current_time)
                for ev in result_events:
                    self.push_event(ev['timestamp'], ev['event'])

            elif etype == 'turn_off':
                self.PlatformControl.turn_off(event['nodes'])
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'switch_on':
                result_events = self.PlatformControl.switch_on(
                    event['nodes'], self.current_time)
                for ev in result_events:
                    self.push_event(ev['timestamp'], ev['event'])

            elif etype == 'turn_on':
                self.PlatformControl.turn_on(event['nodes'])
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'arrival':
                record_job_arrival.append(event)
                self.jobs_manager.add_job_to_waiting_queue(event)
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'execution_start':
                if event['job_id'] in self.jobs_manager.active_jobs_id:
                    raise RuntimeError(
                        f"Job {event['job_id']} is already executed"
                    )

                result = self.PlatformControl.compute(
                    event['nodes'], event, self.current_time)
                if result is not None:
                    event['start_time'] = self.current_time
                    record_job_submission.append(event)
                    finish_time, ev = result
                    self.jobs_manager.remove_job_from_waiting_queue(
                        event['job_id'], type='execution_start')
                    self.push_event(finish_time, ev)
                else:
                    raise RuntimeError(
                        f"Job {event['job_id']} failed to execute"
                    )

            elif etype == 'execution_finished':
                terminated = self.PlatformControl.release(
                    event, self.current_time)
                self.num_finished_jobs += 1
                event['terminated'] = terminated
                event['finish_time'] = self.current_time
                record_job_execution.append(event)
                if self.rl and self.rl_type == 'continuous':
                    need_rl = True

            elif etype == 'change_dvfs_mode':
                _ = self.PlatformControl.change_dvfs_mode(
                    event['node'], event['mode'])

            elif etype == 'CALL_RL':
                self.rl_tick_scheduled = False

        self.Monitor.record(
            mode='after',
            machines=self.PlatformControl.machines,
            current_time=self.current_time,
            record_job_arrival=record_job_arrival,
            record_job_submission=record_job_submission,
            record_job_execution=record_job_execution,
        )

        # self._schedule_next_rl_tick()
        
        if self.num_finished_jobs == self.num_jobs:
            message = self.on_finish()
            return message

        if need_rl:
            self.push_event(self.current_time, {'type': 'CALL_RL'})

        message = {'timestamp': self.current_time, 'events': events}
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


def run_simulation(simulator, output_folder):
    simulator.start_simulator()
    while simulator.is_running:
        simulator.advance()
    log_output(simulator, output_folder)
    log_info(f"Simulation completed. Logs saved to: {output_folder}")
