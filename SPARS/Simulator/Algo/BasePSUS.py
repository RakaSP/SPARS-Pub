from operator import itemgetter


class BasePSUS:
    def __init__(self, machines, jobs_manager, start_time, timeout=None):
        self.state = machines.nodes
        self.jobs_manager = jobs_manager
        self.waiting_queue = jobs_manager.waiting_queue
        self.scheduled_queue = []
        self.events = []
        self.current_time = start_time
        self.timeout = timeout

        self.available = []
        self.allocated = []

        self.timeout_list = []
        self.next_timeout_at = None

        # --- merged agenda: started & release_time included ---
        self.compute_agenda = [
            {
                'node_id': n['id'],
                'started': False,        # has the node started computing?
                'start_time': None,
                'finish_time': None,
                'release_time': 0.0,     # time-to-free; pre-start = job walltime
            }
            for n in self.state
        ]

    # ---------------- Core helpers ----------------
    def _agenda_by_id(self):
        return {e['node_id']: e for e in self.compute_agenda}

    def add_job_to_scheduled_queue(self, job, nodes):
        self.scheduled_queue.append({'job_id': job['job_id'],
                                     'subtime': job['subtime'],
                                     'reqtime': job['reqtime'],
                                     'runtime': job['runtime'],
                                     'res': job['res'],
                                     'nodes': nodes})
    # ---------------- Allocation ----------------

    def allocate(self, job, allocated_nodes):
        """Register allocation: remove from available, add to allocated, set pre-start release_time."""
        if not allocated_nodes:
            return

        alloc_ids = {n['id'] for n in allocated_nodes}
        # Remove from available
        self.available = [
            n for n in self.available if n['id'] not in alloc_ids]
        # Add to allocated (no dups)
        already = {n['id'] for n in self.allocated}
        self.allocated.extend(
            n for n in allocated_nodes if n['id'] not in already)

        node_ids = [n['id'] for n in allocated_nodes]
        self.add_job_to_scheduled_queue(job, node_ids)
        self.waiting_queue = [
            _job for _job in self.waiting_queue if job['job_id'] != _job['job_id']]
        # Pre-start: set release_time = walltime (slowest node)
        compute_speed = min(n['compute_speed'] for n in allocated_nodes)
        walltime = job['runtime'] / compute_speed
        self._set_prestart_release_time(node_ids, walltime)

    def _set_prestart_release_time(self, node_ids, walltime):
        """Before execution, just promise the walltime; no start/finish yet."""
        ag = self._agenda_by_id()
        for nid in node_ids:
            ca = ag[nid]
            # Only set if not started; allocation may happen earlier
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
        """
        Called ONLY from prep_schedule().
        1) Reset agenda entries for nodes that have FINISHED:
        idle := (state=='active' and job_id is None and node_id not in allocated_ids)
        -> started=False, start_time=None, finish_time=None, release_time=0.
        2) For nodes that are started: release_time = max(0, finish - now).
        (Pre-start entries keep their release_time from allocation.)
        """
        now = self.current_time
        by_id = {n['id']: n for n in self.state}
        allocated_ids = {n['id'] for n in self.allocated}

        for ca in self.compute_agenda:
            nid = ca['node_id']
            node = by_id.get(nid)

            # Finished/idle nodes (not currently allocated by scheduler)
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
                continue  # nothing more to compute for this node

            # Running nodes that already started: update remaining time
            if ca['started'] and ca['finish_time'] is not None:
                ca['release_time'] = max(0.0, float(ca['finish_time'] - now))
            # else (pre-start): keep the pre-start release_time set by allocation

    # ---------------- Events builder ----------------

    def events_builder(self):
        """Build execution_start & power events; stamp start/finish when job becomes executable."""
        node_by_id = {n['id']: n for n in self.state}
        agenda = self._agenda_by_id()
        all_allocated_node_ids = []
        jobs_to_start = []
        for job in self.scheduled_queue:
            allocated_node_ids = job['nodes']
            all_allocated_node_ids.extend(allocated_node_ids)

            # Executable only if every allocated node is active & idle
            executable = True
            for nid in allocated_node_ids:
                node = node_by_id.get(nid)
                if not node or node['state'] != 'active' or node.get('job_id') is not None:
                    executable = False
                    break

            if executable:

                # Stamp start/finish & mark started
                comp_speeds = [node_by_id[nid]['compute_speed']
                               for nid in allocated_node_ids]
                compute_speed = min(comp_speeds)
                walltime = job['runtime'] / compute_speed

                start = self.current_time
                finish = start + walltime
                for nid in allocated_node_ids:
                    ca = agenda[nid]
                    ca['started'] = True
                    ca['start_time'] = start
                    ca['finish_time'] = finish

                # Partial update ONLY these nodes
                self.update_custom_resources_agenda_partial(allocated_node_ids)

                # Emit execution_start event
                get = itemgetter('job_id', 'subtime',
                                 'runtime', 'reqtime', 'res')
                job_id, subtime, runtime, reqtime, res = get(job)
                self.push_event(self.current_time, {
                    'job_id': job_id,
                    'subtime': subtime,
                    'runtime': runtime,
                    'reqtime': reqtime,
                    'res': res,
                    'type': 'execution_start',
                    'nodes': allocated_node_ids,
                })

                jobs_to_start.append(job)

        self.scheduled_queue = list(
            filter(lambda job: job not in jobs_to_start, self.scheduled_queue))

        # Power: wake sleeping nodes that are allocated
        switch_on = [
            nid for nid in all_allocated_node_ids if node_by_id[nid]['state'] == 'sleeping']

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

    # ---------------- Timeout handling (unchanged) ----------------
    def remove_from_timeout_list(self, node_ids):
        ids = set(node_ids)
        self.timeout_list[:] = [
            ti for ti in self.timeout_list if ti.get('node_id') not in ids]

    def timeout_policy(self):
        if not self.timeout:
            return

        now = self.current_time

        expire_at = now + self.timeout

        state_by_id = {n['id']: n for n in self.state}
        allocated_ids = {n['id'] for n in self.allocated}

        # Current timeout entries (by id) for O(1) checks
        timeout_ids = {t['node_id'] for t in self.timeout_list}

        # 1) Remove any entries whose node just became allocated
        if timeout_ids & allocated_ids:
            self.timeout_list = [
                t for t in self.timeout_list if t['node_id'] not in allocated_ids]
            timeout_ids -= allocated_ids

        # 2) Add new timeouts for nodes that are idle, not allocated, and not already tracked
        for node in self.state:
            idle = (node['state'] == 'active' and node.get('job_id') is None)
            nid = node['id']
            if idle and nid not in allocated_ids and nid not in timeout_ids:
                self.timeout_list.append({'node_id': nid, 'time': expire_at})
                timeout_ids.add(nid)

        # 3) Walk the timeout list: apply your rules
        keep = []
        switch_off = []
        next_earliest = None

        for t in self.timeout_list:
            nid = t['node_id']

            node = state_by_id.get(nid)
            if node is None:
                # stale entry -> drop
                continue

            # If node is allocated -> must be removed (drop)
            if nid in allocated_ids:
                continue

            idle = (node['state'] == 'active' and node.get('job_id') is None)
            if not idle:
                # not idle -> no timeout tracking
                continue

            # idle & tracked
            if now < t['time']:
                keep.append(t)
                next_earliest = t['time'] if next_earliest is None else min(
                    next_earliest, t['time'])
            else:
                # now >= time
                # if not allocated -> switch off
                switch_off.append(nid)
                # (if it were allocated, we'd have dropped it above)

        # 4) Commit new timeout list
        self.timeout_list = keep

        # 5) Emit actions
        if switch_off:
            self.push_event(now, {'type': 'switch_off', 'nodes': switch_off})

        # 6) Schedule the next call_me_later exactly at the earliest pending timeout
        if next_earliest is not None and getattr(self, 'next_timeout_at', None) != next_earliest:
            self.push_event(next_earliest, {'type': 'call_me_later'})
            self.next_timeout_at = next_earliest

    # ---------------- Scheduler prep ----------------

    def prep_schedule(self):
        self.events = []
        waiting_queue = self.jobs_manager.waiting_queue
        scheduled_ids = [job['job_id'] for job in self.scheduled_queue]
        self.waiting_queue = [
            job for job in waiting_queue if job['job_id'] not in scheduled_ids]
        # Partition nodes from current state (single pass)
        self.available, self.allocated = [], []
        allocated_ids = []
        for job in self.scheduled_queue:
            allocated_ids.extend(job['nodes'])
        for node in self.state:
            if node.get('job_id') is not None or node['id'] in allocated_ids:
                self.allocated.append(node)
            else:
                self.available.append(node)

        # Only started nodes’ release_time is recomputed from finish_time
        self.update_custom_resources_agenda_global()
