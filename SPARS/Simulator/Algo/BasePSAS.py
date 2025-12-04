from operator import itemgetter
import re
import math
_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")


class BasePSAS:
    """
    Per-node Next Releases: Store queue of event to get the earliest node's next idle state
      next_releases = [
        {
          'node_id': <int>,
          'queue': [
            {'phase': <str>, 'start_time': <float>, 'finish_time': <float>},
            ...
          ],
          'release_time': <float>,  # absolute sim time when node becomes ACTIVE & IDLE
        },
        ...
      ]


    States/Phases:
      Machine state (from self.state[*]['state']): 'active', 'sleeping', 'switching_on', 'switching_off'
      Head phases we track: 'switching_off', 'switching_on', 'sleep_to_active', 'compute(job=...)'

    Partitions we expose each scheduling tick (mutually exclusive):
      - self.computing    : state=='active' and job_id is not None
      - self.idle         : state=='active' and job_id is None
      - self.sleeping     : state=='sleeping'
      - self.switching_on : state=='switching_on'
      - self.switching_off: state=='switching_off'
    """

    # ---------------- Init ----------------
    def __init__(self, machines, jobs_manager, start_time, timeout=None):
        self.machines = machines
        self.jobs_manager = jobs_manager

        self.state = machines.nodes
        self.machines_transitions = machines.machines_transition
        self.waiting_queue = jobs_manager.waiting_queue
        self.scheduled_queue = []
        self.events = []
        self.current_time = float(start_time)
        self.timeout = timeout
        self.call_me_laters_tl = []

        # New partitions
        self.computing = []
        self.idle = []
        self.sleeping = []
        self.switching_on = []
        self.switching_off = []
        self.selected_list = []

        self.timeout_list = []
        self.to_be_switched_off_ids = []
        self.next_timeout_at = None

        # resource agenda (rebuilt in prep_schedule)
        self.next_releases = [
            {'node_id': n['id'], 'queue': [],
                'release_time': self.current_time}
            for n in self.state
        ]

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
        self.current_time = float(current_time)

    # ---------------- Helpers ----------------
    def _releases_by_id(self):
        return {e['node_id']: e for e in self.next_releases}

    @staticmethod
    def _sum_queue_abs(q):
        """Return the absolute finish time of last phase or 0 if empty (caller sets to now)."""
        return float(q[-1]['finish_time'])

    def _remaining_time(self, total, started_at, now):
        """Remaining time in current phase; conservative if timestamps unknown."""

        return max(0.0, float(total) - max(0.0, now - float(started_at)))

    def _recalculate_release_at(self, entry):
        """Calculate release time, handling unknown (inf) finish times."""
        if not entry['queue']:
            entry['release_time'] = self.current_time
            return

        last_phase = entry['queue'][-1]
        finish_time = float(last_phase['finish_time'])

        # If any phase has unknown finish time (inf), the entire release time is unknown
        if math.isinf(finish_time):
            entry['release_time'] = float('inf')
        else:
            entry['release_time'] = finish_time

    def _append_phase_abs(self, entry, phase, start_time, duration):
        st = float(start_time)
        ft = st + float(duration)
        entry['queue'].append(
            {'phase': phase, 'start_time': st, 'finish_time': ft})
        entry['release_time'] = ft

    def _cursor_from_queue(self, entry):
        """Where the next phase would start: end of last phase, else entry['release_time'] (can be 0.0)."""
        q = entry['queue']
        if q:
            return float(q[-1]['finish_time'])
        # queue empty -> use recorded release_time (0.0 is allowed by your policy)
        return float(entry['release_time'])

    # ---------------- Transitions lookup (from machines_transitions) ----------------

    def _ensure_transition_index(self):
        """
        Build once: { node_id: { (from_state, to_state): transition_time, ... }, ... }
        Expected external attribute: self.machines_transitions = [
            {"node_id": 1, "transitions": [
                {"from": "sleeping", "to": "active", "transition_time": 12.3}, ...]},
            ...
        ]
        """
        if hasattr(self, "_trans_index_built") and self._trans_index_built:
            return

        self._trans_index = {}
        source = getattr(self, "machines_transitions")
        for row in source:
            nid = row.get("node_id")
            tlist = row.get("transitions") or []
            by_pair = {}
            for t in tlist:
                frm = str(t.get("from"))
                to = str(t.get("to"))
                tt = float(t.get("transition_time"))
                by_pair[(frm, to)] = tt
            if nid is not None:
                self._trans_index[int(nid)] = by_pair

        self._trans_index_built = True

    def _transition_time(self, node_id: int, from_state: str, to_state: str) -> float:
        """Return transition_time for (from_state -> to_state) for node_id; default if not found."""
        self._ensure_transition_index()
        by_pair = self._trans_index.get(int(node_id))
        return float(by_pair.get((from_state, to_state)))

    # ---------------- Resource agenda builders ----------------
    def _prune_finished(self, entry):
        """Drop phases that ended at or before now."""
        now = self.current_time
        if entry['queue']:
            entry['queue'] = [seg for seg in entry['queue']
                              if float(seg['finish_time']) > now]

    def _ensure_head(self, entry, phase_name, start_at, duration):
        """
        Ensure the queue head matches the current physical phase.
        If start_at is None and a matching head exists, keep its timing.
        Otherwise, insert/replace with (now or start_at) + duration.
        """
        q = entry['queue']

        if q and q[0]['phase'] == phase_name:
            if start_at is not None:
                q[0]['start_time'] = float(start_at)
                q[0]['finish_time'] = float(start_at) + float(duration)
        else:
            st = float(start_at)
            ft = st + float(duration)
            q.insert(0, {'phase': phase_name,
                     'start_time': st, 'finish_time': ft})

    # Rebuild to “earliest idle”
    def _rebuild_next_releases_global(self):
        by_id = self._releases_by_id()
        now = self.current_time

        for node in self.state:
            nid = node['id']
            entry = by_id.get(nid)
            if entry is None:
                entry = {'node_id': nid, 'queue': [], 'release_time': now}
                self.next_releases.append(entry)

            # Drop phases already finished, but handle compute phases that exceeded finish_time
            new_queue = []
            for seg in entry['queue']:
                seg_finish_time = float(seg['finish_time'])

                # Check if this is a compute phase that exceeded finish_time but job is still running
                if _COMPUTE_RE.fullmatch(str(seg['phase'])):
                    # Extract job ID from compute phase string - SIMPLE
                    phase_str = str(seg['phase'])
                    # 'compute(job=123)' -> extract '123'
                    # Remove 'compute(job=' and ')'
                    job_id_str = phase_str[12:-1]
                    phase_job_id = int(job_id_str)

                    # If current time is past finish_time AND node is still active with same job
                    if (now >= seg_finish_time and
                        node['state'] == 'active' and
                            node.get('job_id') == phase_job_id):
                        # Job is still running beyond its requested time - mark as unknown
                        seg['finish_time'] = float('inf')
                        new_queue.append(seg)
                        continue

                # For non-compute phases or finished compute phases, keep if not finished
                if seg_finish_time > now:
                    new_queue.append(seg)

            entry['queue'] = new_queue

            state = node['state']
            job_id = node.get('job_id')

            # Check if the current queue is valid for the node's state
            current_queue_valid = False
            if entry['queue']:
                first_phase = entry['queue'][0]
                # Queue is valid if current time is within the phase AND node state matches phase
                if (float(first_phase['start_time']) <= now < float(first_phase['finish_time'])):
                    if ((first_phase['phase'] == 'switching_off' and state == 'switching_off') or
                        (first_phase['phase'] == 'switching_on' and state == 'switching_on') or
                            (_COMPUTE_RE.fullmatch(str(first_phase['phase'])) and state == 'active' and job_id is not None)):
                        current_queue_valid = True

            # If queue is not valid for current state, clear and rebuild
            if not current_queue_valid:
                entry['queue'] = []  # Clear the queue

            # Now rebuild based on current state
            q = entry['queue']

            # Transition durations
            t_off_sleep = self._transition_time(
                nid, 'switching_off', 'sleeping')
            t_sleep_on = self._transition_time(nid, 'sleeping', 'switching_on')
            t_on_active = self._transition_time(nid, 'switching_on', 'active')

            if state == 'switching_off':
                # Add switching_off phase
                if not any(seg['phase'] == 'switching_off' for seg in q):
                    q.insert(0, {
                        'phase': 'switching_off',
                        'start_time': now,
                        'finish_time': now + t_off_sleep
                    })

                # After switching_off, add switching_on for earliest availability calculation
                switching_off_phase = next(
                    (seg for seg in q if seg['phase'] == 'switching_off'), None)
                if switching_off_phase:
                    cursor = float(switching_off_phase['finish_time'])
                    if not any(seg['phase'] == 'switching_on' for seg in q):
                        start_on = cursor + float(t_sleep_on)
                        self._append_phase_abs(
                            entry, 'switching_on', start_on, t_on_active)

            elif state == 'sleeping':
                # For sleeping nodes, calculate when they COULD be available if we started switching_on now
                start_on = now + float(t_sleep_on)
                self._append_phase_abs(
                    entry, 'switching_on', start_on, t_on_active)

            elif state == 'switching_on':
                # For switching_on nodes, calculate remaining switching_on time
                if not any(seg['phase'] == 'switching_on' for seg in q):
                    q.insert(0, {
                        'phase': 'switching_on',
                        'start_time': now,
                        'finish_time': now + t_on_active
                    })

            elif state == 'active':
                if job_id is None:
                    # active & idle: remove any transition phases since we're already available
                    entry['queue'] = [seg for seg in q if seg['phase']
                                      not in ('switching_off', 'switching_on')]
                # if computing, the allocator should have added compute phases

            # release_time = end of last phase, or now if none
            self._recalculate_release_at(entry)

    # ---------------- Allocation and Call me Laters----------------
    def build_callbacks(self):
        execution_finish_lists = []

        if self.current_time == 977.1563697140423:
            print('here')

        for node in self.next_releases:
            queues = node['queue']
            for q in queues:
                if _COMPUTE_RE.fullmatch(str(q['phase'])) and q['finish_time'] not in execution_finish_lists:
                    execution_finish_lists.append(q['finish_time'])

        for node in self.sleeping:
            switch_on_durations = self._transition_time(
                node['id'], 'switching_on', 'active')

            for ef in execution_finish_lists:
                if math.isinf(ef):
                    continue
                call_me_later_time = ef-switch_on_durations
                if call_me_later_time < self.current_time:
                    continue
                if call_me_later_time not in self.call_me_laters_tl:
                    self.push_event(call_me_later_time, {
                                    'type': 'CALL_ME_LATER'})
                    self.call_me_laters_tl.append(call_me_later_time)

    def allocate(self, job, allocated_nodes):
        """
        Reserve nodes and append ONLY the compute phase into next_releases.
        Wake/transition phases are already captured by next_releases' release_time.
        """
        if not allocated_nodes:
            return

        node_ids = [n['id'] for n in allocated_nodes]

        # 1) Check validity
        idle_ids = [n['id'] for n in self.idle]
        for nid in node_ids:
            if nid not in idle_ids:
                raise RuntimeError('Non-Idle node is allocated')

        # 2) Update partitions
        ids = set(node_ids)
        def _filter_out(lst): return [n for n in lst if n['id'] not in ids]
        self.idle = _filter_out(self.idle)
        self.sleeping = _filter_out(self.sleeping)
        self.switching_on = _filter_out(self.switching_on)
        self.switching_off = _filter_out(self.switching_off)
        self.computing.extend(allocated_nodes)

        # 3) Compute walltime via slowest node
        compute_speed = min(float(n['compute_speed']) for n in allocated_nodes)
        assert compute_speed > 0.0
        walltime = float(job['reqtime']) / compute_speed

        # 4) Append ONLY compute at each node's earliest-ready time (release_time)
        by_id = self._releases_by_id()
        for n in allocated_nodes:
            entry = by_id.get(n['id'])
            # next_releases should already exist from prep_schedule()
            assert entry is not None, f"next_releases entry missing for node {n['id']}"
            # earliest time node is ACTIVE & IDLE
            cursor = float(entry['release_time'])
            self._append_phase_abs(
                entry, f'compute(job={job["job_id"]})', cursor, walltime)

        self.push_event(self.current_time, {
            'type': 'execution_start',
            'job_id': job['job_id'],
            'subtime': job['subtime'],
            'reqtime': job['reqtime'],
            'runtime': job['runtime'],
            'res': job['res'],
            'nodes': node_ids
        })

    # ---------------- Timeout handling ----------------

    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [
            ti for ti in self.timeout_list if ti.get('node_id') not in ids]

    def _mark_timed_out_nodes(self):
        """
        For all idle nodes with idle duration exceed timeout,
        add them to to_be_switched_off_ids.
        """
        if self.timeout is None:
            return

        now = self.current_time

        for node in self.idle:
            node_id = node['id']

            # Check if already in switch off list
            if node_id in self.to_be_switched_off_ids:
                continue

            # Find when node became idle
            entry = self._releases_by_id().get(node_id)
            if not entry or not entry['queue']:
                idle_start = now
            else:
                compute_phases = [seg for seg in entry['queue']
                                  if seg['phase'].startswith('compute(job=')]
                if compute_phases:
                    idle_start = compute_phases[-1]['finish_time']
                else:
                    idle_start = entry['release_time'] if entry['release_time'] > 0 else now

            # Check if idle duration exceeds timeout
            if (now - idle_start) > self.timeout:
                self.to_be_switched_off_ids.append(node_id)

    def _rebuild_timeout_list(self):
        """
        Recompute timeout_list from CURRENT state/partitions.
        Policy:
        - If timeout is None: no timeouts -> clear list & marker.
        - Else: every ACTIVE & IDLE, NON-RESERVED node must have a deadline.
                Non-idle nodes must not have a deadline.
        """
        if self.timeout is None:
            self.timeout_list = []
            self.next_timeout_at = None
            return

        now = self.current_time
        expire_at = now + self.timeout

        # Build fast lookups
        idle_ids = {
            n['id'] for n in self.state
            if (n.get('state') == 'active') and (n.get('job_id') is None)
        }

        # Keep only entries for currently idle nodes
        keep_map = {}
        for t in self.timeout_list:
            nid = t['node_id']
            # keep only valid entries; strict access to 'time'
            if (nid in idle_ids):
                keep_map[nid] = float(t['time'])

        # Ensure every eligible node has a deadline; assign new ones to now+timeout
        for nid in (idle_ids):
            if nid not in keep_map:
                keep_map[nid] = expire_at

        # Write back as a list (unsorted is fine; timeout_policy will derive next_earliest)
        self.timeout_list = [{'node_id': nid, 'time': t}
                             for nid, t in keep_map.items()]

    def timeout_policy(self):
        if self.timeout is None:
            return

        now = self.current_time

        if self.current_time == 7822:
            print('here')

        # NEW: refresh (adds new idle nodes, removes non-idle)
        self._rebuild_timeout_list()

        state_by_id = {n['id']: n for n in self.state}
        idle_ids = [n['id'] for n in self.idle]

        keep, switch_off, next_earliest = [], [], None
        for t in self.timeout_list:
            nid = t['node_id']
            node = state_by_id.get(nid)
            if node is None:
                continue

            if nid not in idle_ids:
                continue

            if now >= t['time']:
                # Timeout reached - check if node is in selected_list
                node_in_selected = False
                should_keep = False

                for _job, selected, start_time, finish_time in self.selected_list:
                    selected_ids = [n['id'] for n in selected]
                    if nid in selected_ids:
                        node_in_selected = True
                        # Calculate total transition time: switch_off + switch_on
                        switch_off_duration = self._transition_time(
                            nid, 'switching_off', 'sleeping')
                        switch_on_duration = self._transition_time(
                            nid, 'switching_on', 'active')
                        transition_time = switch_off_duration + switch_on_duration

                        # Check if we can switch off and back on before the job starts
                        if now + transition_time <= start_time:
                            # We can switch off and still have time to switch back on before the job starts
                            switch_off.append(nid)
                        else:
                            # Not enough time - keep the node active
                            should_keep = True
                        break

                if not node_in_selected:
                    # Node not in selected_list, safe to switch off
                    switch_off.append(nid)
                elif should_keep:
                    # Node is in selected_list and we need to keep it
                    keep.append(t)
                    # Only set next_earliest if it's in the future
                    if t['time'] > now:
                        next_earliest = t['time'] if next_earliest is None else min(
                            next_earliest, t['time'])
            else:
                # Timeout not reached yet, keep tracking
                keep.append(t)
                # Only set next_earliest if it's in the future (though it should always be in this case)
                if t['time'] > now:
                    next_earliest = t['time'] if next_earliest is None else min(
                        next_earliest, t['time'])

        self.timeout_list = keep

        if switch_off:
            self.push_event(now, {'type': 'switch_off', 'nodes': switch_off})

        if next_earliest is not None and self.next_timeout_at != next_earliest:
            self.push_event(next_earliest, {'type': 'call_me_later_to'})
            self.next_timeout_at = next_earliest

    # ---------------- Partition & prep ----------------

    def _build_partitions(self):
        """Build mutually-exclusive node partitions."""
        self.computing = []
        self.idle, self.sleeping = [], []
        self.switching_on, self.switching_off = [], []
        self.selected_list = []

        for node in self.state:
            state = node.get('state')
            job_id = node.get('job_id')

            if job_id is not None and state == 'active':
                self.computing.append(node)
            elif state == 'active' and job_id is None:
                self.idle.append(node)
            elif state == 'sleeping':
                self.sleeping.append(node)
            elif state == 'switching_on':
                self.switching_on.append(node)
            elif state == 'switching_off':
                self.switching_off.append(node)

    def prep_schedule(self):
        """
        Rebuild partitions and resource_agenda from current state.
        """
        self.events = []

        if round(self.current_time, 2) == 6594.98:
            print('here')

        # Clear switch off list
        self.to_be_switched_off_ids = []

        # Reconcile queues
        self._rebuild_next_releases_global()

        # Rebuild partitions
        self._build_partitions()

        # Rebuild timeout_list
        self._rebuild_timeout_list()

        # NEW: Mark timed out nodes for switch off
        self._mark_timed_out_nodes()

    # ---------------- Readiness helpers ----------------
    def _node_ready_at(self, node):
        """
        Predict the absolute time when 'node' can start computing if selected now.
        This simply returns the 'release_time' from next_releases.
        """
        # Fetch the release time directly from next_releases (which holds the calculated next event time)
        node_id = node['id']
        entry = self._releases_by_id().get(node_id)

        if entry:
            return entry['release_time']

        # If no entry found, return current time as fallback (or handle as error)
        return self.current_time
