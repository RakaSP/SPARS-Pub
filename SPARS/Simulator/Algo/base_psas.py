from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
import math
import re

import numpy as np

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    job_id: int
    state: str
    row_idx: int


class BasePSAS:
    def __init__(self, machines, jobs_manager, start_time, timeout):
        self.machines = machines
        self.jobs_manager = jobs_manager

        self.state = machines.nodes
        self.machines_transitions = machines.transition_map
        self.waiting_queue = jobs_manager.waiting_queue
        self.scheduled_queue = []
        self.events = []
        self.current_time = float(start_time)
        self.timeout = timeout
        self.call_me_laters_tl = []

        self.computing = []
        self.idle = []
        self.sleeping = []
        self.switching_on = []
        self.switching_off = []
        self.selected_list = []

        self.timeout_list: dict = {}
        self.to_be_switched_off = []
        self.next_timeout_at = None

        self.next_releases = {
            nid: {'queue': [], 'release_time': self.current_time}
            for nid in self.state
        }


    def push_event(self, timestamp, event):
        bucket = next((x for x in self.events if x['timestamp'] == timestamp), None)
        if bucket:
            bucket['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
            self.events.sort(key=itemgetter('timestamp'))

    def set_time(self, current_time):
        self.current_time = float(current_time)


    @staticmethod
    def _sum_queue_abs(q):
        return float(q[-1]['finish_time'])

    def _remaining_time(self, total, started_at, now):
        return max(0.0, float(total) - max(0.0, now - float(started_at)))

    def _recalculate_release_at(self, entry):
        if not entry['queue']:
            entry['release_time'] = self.current_time
            return

        last_phase = entry['queue'][-1]
        finish_time = float(last_phase['finish_time'])

        if math.isinf(finish_time):
            entry['release_time'] = float('inf')
        else:
            entry['release_time'] = finish_time

    def _append_phase_abs(self, entry, phase, start_time, duration):
        st = float(start_time)
        ft = st + float(duration)
        entry['queue'].append({'phase': phase, 'start_time': st, 'finish_time': ft})
        entry['release_time'] = ft

    def _cursor_from_queue(self, entry):
        q = entry['queue']
        if q:
            return float(q[-1]['finish_time'])
        return float(entry['release_time'])


    def _transition_time(
        self,
        node_id: int,
        from_state: str,
        to_state: str,
    ) -> float:
        node_name = self.state[node_id]["node_name"]

        transition = self.machines_transitions[node_name][
            (from_state, to_state)
        ]

        return float(transition["transition_time"])

    def _wake_lead_time(self, node_id: int) -> float:
        t_sleep_to_on = self._transition_time(node_id, "sleeping", "switching_on")
        t_on_to_active = self._transition_time(node_id, "switching_on", "active")
        return float(t_sleep_to_on + t_on_to_active)


    def _prune_finished(self, entry):
        now = self.current_time
        if entry['queue']:
            entry['queue'] = [seg for seg in entry['queue'] if float(seg['finish_time']) > now]

    def _ensure_head(self, entry, phase_name, start_at, duration):
        q = entry['queue']
        if q and q[0]['phase'] == phase_name:
            if start_at is not None:
                q[0]['start_time'] = float(start_at)
                q[0]['finish_time'] = float(start_at) + float(duration)
        else:
            st = float(start_at)
            ft = st + float(duration)
            q.insert(0, {'phase': phase_name, 'start_time': st, 'finish_time': ft})

    def _rebuild_next_releases_global(self):
        now = self.current_time

        for nid, node in self.state.items():
            entry = self.next_releases.get(nid)
            if entry is None:
                entry = {'queue': [], 'release_time': now}
                self.next_releases[nid] = entry

            new_queue = []
            for seg in entry['queue']:
                seg_finish_time = float(seg['finish_time'])

                if _COMPUTE_RE.fullmatch(str(seg['phase'])):
                    phase_job_id = int(str(seg['phase'])[12:-1])
                    if (now >= seg_finish_time and
                            node['state'] == 'active' and
                            node.get('job_id') == phase_job_id):
                        seg['finish_time'] = float('inf')
                        new_queue.append(seg)
                        continue

                if seg_finish_time > now:
                    new_queue.append(seg)

            entry['queue'] = new_queue

            state = node['state']
            job_id = node.get('job_id')

            current_queue_valid = False
            if entry['queue']:
                first_phase = entry['queue'][0]
                if float(first_phase['start_time']) <= now < float(first_phase['finish_time']):
                    if ((first_phase['phase'] == 'switching_off' and state == 'switching_off') or
                            (first_phase['phase'] == 'switching_on' and state == 'switching_on') or
                            (_COMPUTE_RE.fullmatch(str(first_phase['phase'])) and state == 'active' and job_id is not None)):
                        current_queue_valid = True

            if not current_queue_valid:
                entry['queue'] = []

            q = entry['queue']

            t_off_sleep = self._transition_time(nid, 'switching_off', 'sleeping')
            t_sleep_on = self._transition_time(nid, 'sleeping', 'switching_on')
            t_on_active = self._transition_time(nid, 'switching_on', 'active')

            if state == 'switching_off':
                if not any(seg['phase'] == 'switching_off' for seg in q):
                    q.insert(0, {
                        'phase': 'switching_off',
                        'start_time': now,
                        'finish_time': now + t_off_sleep
                    })
                switching_off_phase = next((seg for seg in q if seg['phase'] == 'switching_off'), None)
                if switching_off_phase:
                    cursor = float(switching_off_phase['finish_time'])
                    if not any(seg['phase'] == 'switching_on' for seg in q):
                        start_on = cursor + float(t_sleep_on)
                        self._append_phase_abs(entry, 'switching_on', start_on, t_on_active)

            elif state == 'sleeping':
                start_on = now + float(t_sleep_on)
                self._append_phase_abs(entry, 'switching_on', start_on, t_on_active)

            elif state == 'switching_on':
                if not any(seg['phase'] == 'switching_on' for seg in q):
                    q.insert(0, {
                        'phase': 'switching_on',
                        'start_time': now,
                        'finish_time': now + t_on_active
                    })

            elif state == 'active':
                if job_id is None:
                    entry['queue'] = [seg for seg in q if seg['phase'] not in ('switching_off', 'switching_on')]

            self._recalculate_release_at(entry)

    def build_callbacks(self):
        execution_finish_lists = []

        for entry in self.next_releases.values():
            for q in entry['queue']:
                if _COMPUTE_RE.fullmatch(str(q['phase'])) and q['finish_time'] not in execution_finish_lists:
                    execution_finish_lists.append(q['finish_time'])

        for nid in self.sleeping:
            lead = self._wake_lead_time(nid)
            for ef in execution_finish_lists:
                if math.isinf(ef):
                    continue
                call_me_later_time = float(ef) - float(lead)
                if call_me_later_time < self.current_time:
                    continue
                if call_me_later_time not in self.call_me_laters_tl:
                    self.push_event(call_me_later_time, {'type': 'CALL_ME_LATER'})
                    self.call_me_laters_tl.append(call_me_later_time)

    def allocate(self, job, allocated_nodes):
        if not allocated_nodes:
            return

        for nid in allocated_nodes:
            if nid not in self.idle:
                raise RuntimeError('Non-Idle node is allocated')

        self.idle = [n for n in self.idle if n not in allocated_nodes]
        self.sleeping = [n for n in self.sleeping if n not in allocated_nodes]
        self.switching_on = [n for n in self.switching_on if n not in allocated_nodes]
        self.switching_off = [n for n in self.switching_off if n not in allocated_nodes]
        self.computing.extend(allocated_nodes)

        compute_speed = min(float(self.state[nid]['compute_speed']) for nid in allocated_nodes)
        if compute_speed <= 0.0:
            raise RuntimeError("compute_speed must be > 0")
        walltime = float(job['reqtime']) / compute_speed

        for nid in allocated_nodes:
            entry = self.next_releases.get(nid)
            if entry is None:
                raise RuntimeError(f"next_releases entry missing for node {nid}")
            cursor = float(entry['release_time'])
            self._append_phase_abs(entry, f'compute(job={job["job_id"]})', cursor, walltime)

        self.push_event(self.current_time, {
            **job,
            'type': 'execution_start',
            'nodes': allocated_nodes
        })


    def remove_from_timeout_list(self, node_ids):
        for node_id in node_ids:
            self.timeout_list.pop(node_id, None)

    def _mark_timed_out_nodes(self):
        if self.timeout is None:
            return

        now = self.current_time

        for nid in self.idle:
            if nid in self.to_be_switched_off:
                continue

            entry = self.next_releases.get(nid)
            if not entry or not entry['queue']:
                idle_start = now
            else:
                compute_phases = [seg for seg in entry['queue'] if str(seg['phase']).startswith('compute(job=')]
                if compute_phases:
                    idle_start = compute_phases[-1]['finish_time']
                else:
                    idle_start = entry['release_time'] if entry['release_time'] > 0 else now

            if (now - float(idle_start)) > float(self.timeout):
                self.to_be_switched_off.append(nid)

    def _rebuild_timeout_list(self):
        if self.timeout is None:
            self.timeout_list = {}
            self.next_timeout_at = None
            return

        now = self.current_time
        expire_at = now + float(self.timeout)

        self.timeout_list = {nid: t for nid, t in self.timeout_list.items() if nid in self.idle}

        for nid in self.idle:
            if nid not in self.timeout_list:
                self.timeout_list[nid] = expire_at

    def timeout_policy(self):
        if self.timeout is None:
            return

        now = self.current_time
        self._rebuild_timeout_list()


        keep, switch_off, next_earliest = {}, [], None
        for nid, time in self.timeout_list.items():
            node = self.state.get(nid)
            if node is None:
                continue

            if nid not in self.idle:
                continue

            if now >= time:
                node_in_selected = False
                should_keep = False

                for _job, selected, start_time, finish_time in self.selected_list:
                    if nid in selected:
                        node_in_selected = True
                        switch_off_duration = self._transition_time(nid, 'switching_off', 'sleeping')
                        lead = self._wake_lead_time(nid)
                        transition_time = float(switch_off_duration + lead)

                        if now + transition_time <= float(start_time):
                            switch_off.append(nid)
                        else:
                            should_keep = True
                        break

                if not node_in_selected:
                    switch_off.append(nid)
                elif should_keep:
                    keep[nid] = time
                    if time > now:
                        next_earliest = time if next_earliest is None else min(next_earliest, time)
            else:
                keep[nid] = time
                if time > now:
                    next_earliest = time if next_earliest is None else min(next_earliest, time)

        self.timeout_list = keep

        if switch_off:
            self.push_event(now, {'type': 'switch_off', 'nodes': switch_off})

        if next_earliest is not None and self.next_timeout_at != next_earliest:
            self.push_event(next_earliest, {'type': 'call_me_later_to'})
            self.next_timeout_at = next_earliest

    def _build_partitions(self):
        self.computing = []
        self.idle, self.sleeping = [], []
        self.switching_on, self.switching_off = [], []
        self.selected_list = []

        for nid, node in self.state.items():
            state = node.get('state')
            job_id = node.get('job_id')

            if job_id is not None and state == 'active':
                self.computing.append(nid)
            elif state == 'active' and job_id is None:
                self.idle.append(nid)
            elif state == 'sleeping':
                self.sleeping.append(nid)
            elif state == 'switching_on':
                self.switching_on.append(nid)
            elif state == 'switching_off':
                self.switching_off.append(nid)

    def prep_schedule(self):
        self.events = []
        self.to_be_switched_off = []

        self._rebuild_next_releases_global()
        self._build_partitions()
        self._rebuild_timeout_list()
        self._mark_timed_out_nodes()

    def _node_ready_at(self, node_id):
        entry = self.next_releases.get(node_id)
        if entry:
            return entry['release_time']
        return self.current_time