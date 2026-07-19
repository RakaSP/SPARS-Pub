import torch as T


class Reward:
    def __init__(
        self,
        weight1: float = 0.5,
        weight2: float = 0.5,
        device: str = "cuda",
        require_grad: bool = True,
    ) -> None:
        self.weight1 = float(weight1)
        self.weight2 = float(weight2)
        self.device = T.device(device)
        self.require_grad = bool(require_grad)

    def _to_tensor(self, value: float) -> T.Tensor:
        return T.tensor(
            value,
            dtype=T.float32,
            device=self.device,
            requires_grad=self.require_grad,
        )

    @staticmethod
    def get_need_decision_idx(simulator):
        machine = simulator.platform_control.machine
        need_decision_idx = []

        for node_id, node in machine.nodes.items():
            state = node["state"]

            is_switching_off = state == "switching_off"
            is_switching_on = state == "switching_on"
            is_switching = (
                is_switching_off
                or is_switching_on
            )

            is_idle = state == "active"
            is_sleeping = state == "sleeping"
            is_allocated = node.get("job_id") is not None

            is_really_idle = (
                is_idle
                and not is_allocated
            )

            can_switch_off = (
                not is_switching
                and is_really_idle
            )

            can_switch_on = (
                not is_switching
                and is_sleeping
            )

            if can_switch_off or can_switch_on:
                need_decision_idx.append(node_id)

        return need_decision_idx

    @staticmethod
    def get_node_max_non_compute_ecr(
        machine,
        node_id,
    ):
        node = machine.nodes[node_id]
        node_spec = machine.node_specs[node["node_name"]]
        profile = node_spec["dvfs_profiles"][node["dvfs_mode"]]

        non_compute_powers = [
            float(profile["power_idle"])
        ]

        for state_name, state_def in node_spec["states"].items():
            if state_name == "active":
                continue

            power = state_def["power"]

            if power == "from_dvfs":
                power = profile["power_idle"]

            non_compute_powers.append(float(power))

        return max(non_compute_powers)

    def calculate_reward(
        self,
        obs,
        next_obs,
        simulator,
        next_simulator,
        actions,
    ):
        current_time = next_simulator.current_time

        Delta_T = (
            next_simulator.current_time
            - simulator.current_time
        )

        if Delta_T == 0:
            return self._to_tensor(0.0)

        need_decision_idx = self.get_need_decision_idx(
            simulator
        )

        machine = simulator.platform_control.machine

        wasted_energy = 0.0

        for node_id in need_decision_idx:
            host_wasted_energy = (
                next_simulator.monitor.energy[node_id].get(
                    "energy_waste"
                )
                - simulator.monitor.energy[node_id].get(
                    "energy_waste"
                )
            )

            max_non_compute_ecr = (
                self.get_node_max_non_compute_ecr(
                    machine,
                    node_id,
                )
            )

            host_wasted_energy /= (
                max_non_compute_ecr
                * Delta_T
            )

            wasted_energy += host_wasted_energy

        wasted_energy /= max(
            len(need_decision_idx),
            1,
        )

        if not next_simulator.is_running:
            reward = (
                -self.weight1
                * wasted_energy
            )

            return self._to_tensor(reward)

        start_times = {
            job["job_id"]: job["start_time"]
            for job
            in next_simulator.monitor.jobs_submission_log
        }

        waiting_time_since_last_dt = 0.0
        n_job_waitting = 0

        last_dt = (
            current_time
            - Delta_T
        )

        for job in next_simulator.monitor.jobs_arrival_log:
            start_time = start_times.get(
                job["job_id"]
            )

            if (
                start_time is not None
                and start_time < last_dt
            ):
                continue

            n_job_waitting += 1

            if start_time is not None:
                waittime = (
                    start_time
                    - last_dt
                )
            else:
                waittime = Delta_T

            waiting_time_since_last_dt += (
                waittime
                / Delta_T
            )

        waiting_time_since_last_dt /= max(
            n_job_waitting,
            1,
        )

        reward = (
            -self.weight1
            * wasted_energy
            -self.weight2
            * waiting_time_since_last_dt
        )

        return self._to_tensor(reward)