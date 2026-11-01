from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from spars.core.registry import validate_component_config

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager
    from spars.core.power_manager import PowerManager


class BasePSUS:
    def __init__(self, machines: "Machine", jobs_manager: "JobsManager", start_time, config):
        validate_component_config(config, type(self).__name__, set())
        self.state = machines.nodes
        self.jobs_manager = jobs_manager
        self.waiting_queue = jobs_manager.waiting_queue
        self.scheduled_queue = []
        self.events = []
        self.current_time = start_time

        self.available = []
        self.allocated = []

        self.power_state: PowerManager | None = None

        self.compute_agenda = [
            {
                'node_id': node_id,
                'started': False,
                'start_time': None,
                'finish_time': None,
                'release_time': 0.0,
            }
            for node_id in self.state.keys()
        ]

    def set_power_state(self, power_state: PowerManager) -> None:
        self.power_state = power_state

    def _compute_finish_times(self) -> tuple[float, ...]:
        return tuple(sorted({
            float(entry["finish_time"])
            for entry in self.compute_agenda
            if entry.get("started") and entry.get("finish_time") is not None
        }))

    def _agenda_by_id(self):
        return {e['node_id']: e for e in self.compute_agenda}

    def add_job_to_scheduled_queue(self, job, nodes):
        self.scheduled_queue.append({**job, 'nodes': nodes})

    def allocate(self, job, allocated_nodes):
        """Register allocation: remove from available, add to allocated, set pre-start release_time."""
        if not allocated_nodes:
            return

        alloc_ids = set(allocated_nodes)
        self.available = [n for n in self.available if n not in alloc_ids]
        already = set(self.allocated)
        self.allocated.extend(nid for nid in allocated_nodes if nid not in already)

        self.add_job_to_scheduled_queue(job, allocated_nodes)
        self.waiting_queue = [
            _job for _job in self.waiting_queue if job['job_id'] != _job['job_id']]
        compute_speed = min(self.state[nid]['compute_speed'] for nid in allocated_nodes)
        walltime = job['reqtime'] / compute_speed
        self._set_prestart_release_time(allocated_nodes, walltime)
    def _set_prestart_release_time(self, node_ids, walltime):
        """Before execution, just promise the walltime; no start/finish yet."""
        ag = self._agenda_by_id()
        for nid in node_ids:
            ca = ag[nid]
            if not ca['started']:
                ca['release_time'] = float(walltime)

    # ---------------- Agendas ----------------
    def update_custom_resources_agenda_partial(self, node_ids):
        """
        Recompute release_time only for the given node_ids.
        - If started & finish_time set: release_time = max(0, finish - now)
        - If not started: keep whatever pre-start release_time was set by allocation
        """
        now = self.current_time
        ag = self._agenda_by_id()
        for nid in node_ids:
            ca = ag.get(nid)
            if not ca:
                continue
            if ca['started'] and ca['finish_time'] is not None:
                ca['release_time'] = max(0.0, float(ca['finish_time'] - now))
            # else: keep pre-start release_time as-is

    def update_custom_resources_agenda_global(self):
        now = self.current_time
        allocated_ids = set(self.allocated)

        for ca in self.compute_agenda:
            nid = ca['node_id']
            node = self.state.get(nid)  # <-- direct lookup

            idle = (
                node is not None
                and node.get('state') == 'active'
                and node.get('job_id') is None
                and nid not in allocated_ids
            )
            if idle:
                ca['started'] = False
                ca['start_time'] = None
                ca['finish_time'] = None
                ca['release_time'] = 0.0
                continue

            if ca['started'] and ca['finish_time'] is not None:
                ca['release_time'] = max(0.0, float(ca['finish_time'] - now))

    # ---------------- Events builder ----------------

    def events_builder(self):
        """Build execution_start & power events; stamp start/finish when job becomes executable."""
        agenda = self._agenda_by_id()
        all_allocated_node_ids = []
        jobs_to_start = []
        for job in self.scheduled_queue:
            allocated_node_ids = job['nodes']
            all_allocated_node_ids.extend(allocated_node_ids)

            # Executable only if every allocated node is active & idle
            executable = True
            for nid in allocated_node_ids:
                node = self.state.get(nid)
                if not node or node['state'] != 'active' or node.get('job_id') is not None:
                    executable = False
                    break

            if executable:
                # Stamp start/finish & mark started
                comp_speeds = [self.state[nid]['compute_speed']
                               for nid in allocated_node_ids]
                compute_speed = min(comp_speeds)
                walltime = job['reqtime'] / compute_speed

                start = self.current_time
                finish = start + walltime
                for nid in allocated_node_ids:
                    ca = agenda[nid]
                    ca['started'] = True
                    ca['start_time'] = start
                    ca['finish_time'] = finish

                self.update_custom_resources_agenda_partial(allocated_node_ids)

                self.push_event(self.current_time, {
                    **job,
                    'type': 'execution_start',
                    'nodes': allocated_node_ids,
                })
                jobs_to_start.append(job)

        self.scheduled_queue = list(
            filter(lambda job: job not in jobs_to_start, self.scheduled_queue))

        # Power: wake sleeping nodes that are allocated
        switch_on = [
            nid for nid in all_allocated_node_ids if self.state[nid]['state'] == 'sleeping']

        if switch_on:
            self.push_event(self.current_time, {
                            'type': 'switch_on', 'nodes': switch_on})

    # ---------------- Events & time ----------------

    def push_event(self, timestamp, event):
        bucket = next(
            (x for x in self.events if x['timestamp'] == timestamp), None)
        if bucket:
            bucket['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})
            self.events.sort(key=itemgetter('timestamp'))

    def set_time(self, current_time):
        self.current_time = current_time

    # ---------------- Scheduler prep ----------------

    def _refused_wakes(self):
        """Sleeping nodes the power manager refuses to wake.

        A job planned on such a node is unplanned: it goes back to the queue
        and the next plan picks other nodes (or waits).
        """
        allow_wake = getattr(self.power_state, "allow_wake", None)
        asleep = [nid for nid, node in self.state.items() if node.get('state') == 'sleeping']
        if allow_wake is None or not asleep:
            return set()
        refused = set(asleep) - set(allow_wake(asleep, self.current_time))
        if not refused:
            return refused
        agenda = self._agenda_by_id()
        for job in list(self.scheduled_queue):
            if refused & set(job['nodes']):
                self.scheduled_queue.remove(job)
                for nid in job['nodes']:
                    if not agenda[nid]['started']:
                        agenda[nid]['release_time'] = 0.0
        return refused

    def prep_schedule(self):
        self.events = []
        self.refused_wake = self._refused_wakes()
        waiting_queue = self.jobs_manager.waiting_queue
        scheduled_ids = [job['job_id'] for job in self.scheduled_queue]
        self.waiting_queue = [
            job for job in waiting_queue if job['job_id'] not in scheduled_ids
        ]
        self.available, self.allocated = [], []
        allocated_ids = []
        for job in self.scheduled_queue:
            allocated_ids.extend(job['nodes'])
        for node_id, node in self.state.items():
            if node.get('job_id') is not None or node_id in allocated_ids:
                self.allocated.append(node_id)
            elif node_id not in self.refused_wake:
                self.available.append(node_id)
        self.update_custom_resources_agenda_global()
