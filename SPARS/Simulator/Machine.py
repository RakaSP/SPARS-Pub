class Machine:
    def __init__(self, platform_info, start_time):

        self.platform_info = platform_info
        self.machines = self.platform_info['machines']
        self.current_time = start_time
        self.is_running = False
        self.nodes = []
        self.machines_transition = []

        for machine in self.machines:
            dvfs_mode = machine['dvfs_mode']
            active_state = machine['states']['active']
            dvfs_profile = machine['dvfs_profiles'][dvfs_mode]

            power = dvfs_profile['power'] if active_state['power'] == 'from_dvfs' else active_state['power']
            compute_speed = dvfs_profile['compute_speed'] if active_state[
                'compute_speed'] == 'from_dvfs' else active_state['compute_speed']

            node = {
                'id': machine['id'],
                'state': 'active',
                'dvfs_mode': dvfs_mode,
                'power': power,
                'compute_speed': compute_speed,
                'transitions': active_state['transitions'],
                'job_id': None,
            }

            self.nodes.append(node)

            node_transitions = []
            for from_state, data in machine["states"].items():
                for trans in data.get("transitions", []):
                    node_transitions.append({
                        "from": from_state,
                        "to": trans["state"],
                        "transition_time": trans["transition_time"]
                    })
            self.machines_transition.append({
                "node_id": machine["id"],
                "transitions": node_transitions
            })

    def change_dvfs_mode(self, nodes, mode):
        for node_id in nodes:
            if not any(n['id'] == node_id for n in self.nodes):
                raise ValueError(f"Node ID {node_id} not found in self.nodes")

        for node in self.nodes:
            if node['id'] in nodes:
                if node['state'] != 'active':
                    raise RuntimeError(
                        f"Node {node['id']} must be in 'active' state to change DVFS mode")
                if mode not in self.machines[node['id']]['dvfs_profiles']:
                    raise ValueError(
                        f"Invalid DVFS mode '{mode}' for node {node['id']}")

                node['dvfs_mode'] = mode
                profile = self.machines[node['id']]['dvfs_profiles'][mode]
                node['power'] = profile['power']
                node['compute_speed'] = profile['compute_speed']

    def _update_node_state(self, node, new_state):
        if new_state not in [t['state'] for t in node['transitions']]:
            raise RuntimeError(
                f"Invalid state transition from '{node['state']}' to '{new_state}' on node {node['id']}")
        node['state'] = new_state
        state_def = self.machines[node['id']]['states'][new_state]
        if state_def['power'] == 'from_dvfs':
            dvfs_profile = self.machines[node['id']
                                         ]['dvfs_profiles'][node['dvfs_mode']]
            node['power'] = dvfs_profile['power']
        else:
            node['power'] = state_def['power']

        if state_def['compute_speed'] == 'from_dvfs':
            dvfs_profile = self.machines[node['id']
                                         ]['dvfs_profiles'][node['dvfs_mode']]
            node['compute_speed'] = dvfs_profile['compute_speed']
        else:
            node['compute_speed'] = state_def['compute_speed']

        node['transitions'] = state_def['transitions']

    def switch_on(self, nodes):
        for node in self._get_nodes_by_ids(nodes):
            self._update_node_state(node, 'switching_on')
            node['job_id'] = None

    def turn_on(self, nodes):
        for node in self._get_nodes_by_ids(nodes):
            self._update_node_state(node, 'active')

    def switch_off(self, nodes):
        for node in self._get_nodes_by_ids(nodes):
            self._update_node_state(node, 'switching_off')
            if node['job_id'] is not None:
                raise RuntimeError(
                    f"Node {node['id']} cannot be switched off — node is currently allocated for {node['job_id']}")

    def turn_off(self, nodes):
        for node in self._get_nodes_by_ids(nodes):
            self._update_node_state(node, 'sleeping')
            node['job_id'] = None

    def reserve(self, node_ids):
        for node in self._get_nodes_by_ids(node_ids):
            node['reserved'] = True

    def allocate(self, nodes, job_id):
        for node in self._get_nodes_by_ids(nodes):
            if node['state'] != 'active':
                raise RuntimeError(
                    f"Node {node['id']} cannot be allocated — node is not active")
            if job_id is None:
                raise RuntimeError(
                    f"Node {node['id']} cannot be allocated — job_id is None")
            if node['job_id'] is not None:
                raise RuntimeError(
                    f"Node {node['id']} cannot be allocated for {job_id} — node is already allocated for {node['job_id']}")

            node['job_id'] = job_id
            node['reserved'] = False
        return True

    def release(self, nodes):
        for node in self._get_nodes_by_ids(nodes):
            if node['state'] != 'active':
                raise RuntimeError(
                    f"Node {node['id']} cannot be released — state is not 'active'")
            elif node['state'] == 'active' and node['job_id'] == None:
                raise RuntimeError(
                    f"Node {node['id']} cannot be released — state is 'active' but not computing")
            node['job_id'] = None
            node['reserved'] = False

    def _get_nodes_by_ids(self, node_ids):
        found = [n for n in self.nodes if n['id'] in node_ids]
        if len(found) != len(node_ids):
            existing_ids = {n['id'] for n in self.nodes}
            missing = [nid for nid in node_ids if nid not in existing_ids]
            raise ValueError(f"Node IDs not found: {missing}")
        return found
