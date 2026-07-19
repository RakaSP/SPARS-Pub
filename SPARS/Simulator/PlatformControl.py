from scipy.stats import norm
import numpy as np
from SPARS.Simulator.Machine import Machine


class PlatformControl:
    def __init__(self, platform_path, overrun_policy, start_time):
        self.machine = Machine(platform_path, start_time)
        self.overrun_policy = overrun_policy
        self.exact = False
        self.seed=42
        self.rng = np.random.default_rng(self.seed)

    def validate_duplication(self, node_ids):
        is_duplicate = len(node_ids) != len(set(node_ids))

        if is_duplicate:
            seen = set()
            duplicates = [x for x in node_ids if x in seen or seen.add(x)]
            raise RuntimeError(f"Duplicate node_ids found: {duplicates}")

    def sample_transition_time(self, transition):
        mean = transition["transition_time"]

        if self.exact:
            return mean

        std = transition["std"]

        if std == 0:
            return mean

        sampled = norm(
            loc=mean,
            scale=std,
        ).rvs(random_state=self.rng)

        return float(sampled)
    
    def compute(self, node_ids, job, current_time):
        self.validate_duplication(node_ids)

        if len(node_ids) != job['res']:
            raise RuntimeError(
                f"Resource allocation mismatch for job '{job['job_id']}': "
                f"Requested {job['res']} nodes, but allocated {len(node_ids)}."
            )
        success = self.machine.allocate(node_ids, job['job_id'])

        if not success:
            raise RuntimeError(
                f"Job {job['id']} failed to execute"
            )

        if self.overrun_policy == 'terminate':
            compute_power = min(self.machine.nodes[nid]['compute_speed'] for nid in node_ids)

            actual_compute_demand = job['runtime']
            actual_finish_time = current_time + \
                (actual_compute_demand / compute_power)

            requested_compute_demand = job['reqtime']
            requested_finish_time = current_time + \
                (requested_compute_demand / compute_power)
            event = {'job_id': job['job_id'], 'type': 'execution_finished', 'res': job['res'], 'nodes': node_ids,
                     'start_time': current_time, 'subtime': job['subtime'], 'start_time': current_time, 'reqtime': job['reqtime'], 'req_finish_time': requested_finish_time, 'runtime': job['runtime'], 'actual_finish_time': actual_finish_time, 'user_id': job['user_id']}

            finish_time = min(requested_finish_time, actual_finish_time)

            return finish_time, event

        elif self.overrun_policy == 'continue':
            compute_power = min(self.machine.nodes[nid]['compute_speed'] for nid in node_ids)

            actual_compute_demand = job['runtime']
            actual_finish_time = current_time + \
                (actual_compute_demand / compute_power)

            requested_compute_demand = job['reqtime']
            requested_finish_time = current_time + \
                (requested_compute_demand / compute_power)
            event = {'job_id': job['job_id'], 'type': 'execution_finished', 'res': job['res'], 'nodes': node_ids,
                     'start_time': current_time, 'subtime': job['subtime'], 'start_time': current_time, 'reqtime': job['reqtime'], 'req_finish_time': requested_finish_time, 'runtime': job['runtime'], 'actual_finish_time': actual_finish_time, 'user_id': job['user_id']}

            finish_time = max(requested_finish_time, actual_finish_time)

            return actual_finish_time, event

    def change_dvfs_mode(self, node_ids, mode):
        self.validate_duplication(node_ids)
        self.machine.change_dvfs_mode(node_ids, mode)
        return {'type': 'change_dvfs_mode', 'node': node_ids, 'mode': mode}

    def release(self, event, current_time):
        terminated = False
        if current_time < event['actual_finish_time']:
            terminated = True
        self.machine.release(event['nodes'])

        return terminated

    def reserve_node(self, node_ids):
        self.validate_duplication(node_ids)
        self.machine.reserve(node_ids)

    def turn_on(self, node_ids, current_time):
        self.validate_duplication(node_ids)
        self.machine.turn_on(node_ids, current_time)

    def turn_off(self, node_ids, current_time):
        self.validate_duplication(node_ids)
        self.machine.turn_off(node_ids, current_time)

    def switch_off(self, node_ids, current_time, oracle_durations=None):
        self.validate_duplication(node_ids)
        self.machine.switch_off(node_ids, current_time)
        turnoff_map = {}

        for node_id in node_ids:
            node_name = self.machine.nodes[node_id]["node_name"]
            transitions = self.machine.transition_map[node_name]

            active_to_switching_off = transitions[
                ("active", "switching_off")
            ]
            switching_off_to_sleeping = transitions[
                ("switching_off", "sleeping")
            ]

            if oracle_durations is not None and node_id in oracle_durations:
                total_duration = float(oracle_durations[node_id])
            else:
                total_duration = (
                    self.sample_transition_time(active_to_switching_off)
                    + self.sample_transition_time(switching_off_to_sleeping)
                )

            turn_off_done_at = current_time + total_duration

            turnoff_map.setdefault(
                turn_off_done_at,
                [],
            ).append(node_id)

        result = []

        for timestamp, nodes in turnoff_map.items():
            result.append(
                {
                    "event": {
                        "type": "turn_off",
                        "nodes": nodes,
                    },
                    "timestamp": timestamp,
                }
            )

        return result


    def switch_on(self, node_ids, current_time, oracle_durations=None):
        self.validate_duplication(node_ids)
        self.machine.switch_on(node_ids, current_time)
        turnon_map = {}

        for node_id in node_ids:
            node_name = self.machine.nodes[node_id]["node_name"]
            transitions = self.machine.transition_map[node_name]

            sleeping_to_switching_on = transitions[
                ("sleeping", "switching_on")
            ]
            switching_on_to_active = transitions[
                ("switching_on", "active")
            ]

            if oracle_durations is not None and node_id in oracle_durations:
                total_duration = float(oracle_durations[node_id])
            else:
                total_duration = (
                    self.sample_transition_time(sleeping_to_switching_on)
                    + self.sample_transition_time(switching_on_to_active)
                )

            turn_on_done_at = current_time + total_duration

            turnon_map.setdefault(
                turn_on_done_at,
                [],
            ).append(node_id)

        result = []

        for timestamp, nodes in turnon_map.items():
            result.append(
                {
                    "event": {
                        "type": "turn_on",
                        "nodes": nodes,
                    },
                    "timestamp": timestamp,
                }
            )

        return result
