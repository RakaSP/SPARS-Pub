from SPARS.Simulator.Machine import Machine


class PlatformControl:
    def __init__(self, platform_info, overrun_policy, start_time):
        """
        Manage platform resources and node availability for the simulator.
        Help adds machine events for simulator

        - `resources_agenda` tracks each node's next available time.
        - Overrun policy (when a job runs longer than the user-requested wall time):
          1) **continue**: `release_time` is initially the requested wall time. When that
             time is reached and the job is still running, the node stays allocated and
             the job’s `release_time` becomes unknown (the system cannot predict finish time).
          2) **terminate**: when the requested wall time is reached, the simulator
             stops the job and immediately frees the node.

        Args:
            platform_info: Platform configuration used to construct the node pool.
            policy: The overrun policy ('terminate' or 'continue')
            start_time: Simulation start timestamp.

        Attributes:
            machines: Machine inventory built from `platform_info`.
            resources_agenda: List of dicts, one per node, with keys:
                - `id`: node identifier
                - `release_time`: next time the node becomes free
        """
        self.machines = Machine(platform_info, start_time)

        self.overrun_policy = overrun_policy

    def get_state(self):
        return self.machines.nodes

    def validate_duplication(self, node_ids):
        is_duplicate = len(node_ids) != len(set(node_ids))

        if is_duplicate:
            # Find the duplicate values
            seen = set()
            duplicates = [x for x in node_ids if x in seen or seen.add(x)]
            raise RuntimeError(f"Duplicate node_ids found: {duplicates}")

    def compute(self, node_ids, job, current_time):
        self.validate_duplication(node_ids)

        if len(node_ids) != job['res']:
            raise RuntimeError(
                f"Resource allocation mismatch for job '{job['job_id']}': "
                f"Requested {job['res']} nodes, but allocated {len(node_ids)}."
            )
        success = self.machines.allocate(node_ids, job['job_id'])

        if not success:
            raise RuntimeError(
                f"Job {job['id']} failed to execute"
            )

        if self.overrun_policy == 'terminate':
            compute_power = min(node['compute_speed']
                                for node in self.machines.nodes if node['id'] in node_ids)

            actual_compute_demand = job['runtime']
            actual_finish_time = current_time + \
                (actual_compute_demand / compute_power)

            requested_compute_demand = job['reqtime']
            requested_finish_time = current_time + \
                (requested_compute_demand / compute_power)
            event = {'job_id': job['job_id'], 'type': 'execution_finished', 'res': job['res'], 'nodes': node_ids,
                     'start_time': current_time, 'subtime': job['subtime'], 'start_time': current_time, 'reqtime': job['reqtime'], 'req_finish_time': requested_finish_time, 'runtime': job['runtime'], 'actual_finish_time': actual_finish_time}

            finish_time = min(requested_finish_time, actual_finish_time)

            return finish_time, event

        elif self.overrun_policy == 'continue':
            compute_power = min(node['compute_speed']
                                for node in self.machines.nodes if node['id'] in node_ids)

            actual_compute_demand = job['runtime']
            actual_finish_time = current_time + \
                (actual_compute_demand / compute_power)

            requested_compute_demand = job['reqtime']
            requested_finish_time = current_time + \
                (requested_compute_demand / compute_power)
            event = {'job_id': job['job_id'], 'type': 'execution_finished', 'res': job['res'], 'nodes': node_ids,
                     'start_time': current_time, 'subtime': job['subtime'], 'start_time': current_time, 'reqtime': job['reqtime'], 'req_finish_time': requested_finish_time, 'runtime': job['runtime'], 'actual_finish_time': actual_finish_time}

            finish_time = max(requested_finish_time, actual_finish_time)

            return actual_finish_time, event

    def change_dvfs_mode(self, node_ids, mode):
        self.validate_duplication(node_ids)
        self.machines.change_dvfs_mode(node_ids, mode)
        return {'type': 'change_dvfs_mode', 'node': node_ids, 'mode': mode}

    def release(self, event, current_time):
        terminated = False  # under request
        if current_time < event['actual_finish_time']:
            terminated = True
        self.machines.release(event['nodes'])

        return terminated

    def reserve_node(self, node_ids):
        self.validate_duplication(node_ids)
        self.machines.reserve(node_ids)

    def turn_on(self, node_ids):
        self.validate_duplication(node_ids)
        self.machines.turn_on(node_ids)

    def turn_off(self, node_ids):
        self.validate_duplication(node_ids)
        self.machines.turn_off(node_ids)

    def switch_off(self, node_ids, current_time):
        self.validate_duplication(node_ids)
        # Trigger switch-off now
        self.machines.switch_off(node_ids)

        # Grouping maps
        # time when switching_off -> sleeping completes (for the returned event)
        turnoff_map = {}

        for node_id in node_ids:
            # Find transition spec for this node
            mt = next((mt for mt in self.machines.machines_transition
                       if mt.get('node_id') == node_id), None)

            if mt:
                for tr in mt.get('transitions'):
                    frm = tr.get('from')
                    to = tr.get('to')
                    tt = tr.get('transition_time')
                    if frm == 'switching_off' and to == 'sleeping':
                        t_off = tt  # switching_off -> sleeping
                        break

            # Absolute times
            turn_off_done_at = current_time + (t_off)

            # Group nodes by times
            turnoff_map.setdefault(turn_off_done_at, []).append(node_id)

        # Return the original 'turn_off' event at the time the nodes finish turning off
        result = []
        for ts, nodes in turnoff_map.items():
            result.append({
                'event': {'type': 'turn_off', 'nodes': nodes},
                'timestamp': ts
            })

        return result

    def switch_on(self, node_ids, current_time):
        self.validate_duplication(node_ids)
        # Trigger switch-on now
        self.machines.switch_on(node_ids)

        # Grouping maps
        # time when switching_on -> active completes (for the returned event)
        turnon_map = {}

        for node_id in node_ids:
            # Find transition spec for this node
            mt = next((mt for mt in self.machines.machines_transition
                       if mt.get('node_id') == node_id), None)

            if mt:
                for tr in mt.get('transitions'):
                    frm = tr.get('from')
                    to = tr.get('to')
                    tt = tr.get('transition_time', 0)
                    if frm == 'sleeping' and to == 'switching_on':
                        t_sleep_to_on = tt  # sleeping -> switching_on
                    elif frm == 'switching_on' and to == 'active':
                        t_on = tt  # switching_on -> active

            # Absolute time when node is ACTIVE again
            turn_on_done_at = current_time + (t_sleep_to_on) + (t_on)

            # Group nodes by activation time
            turnon_map.setdefault(turn_on_done_at, []).append(node_id)

        # Return 'turn_on' events at the time nodes finish turning on
        result = []
        for ts, nodes in turnon_map.items():
            result.append({
                'event': {'type': 'turn_on', 'nodes': nodes},
                'timestamp': ts
            })

        return result
