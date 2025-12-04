from SPARS.Logger import log_trace
from typing import Dict, Any
import torch as T


class Reward:
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.9,
        device: str = "cuda",
        require_grad: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.device = T.device(device)
        self.require_grad = bool(require_grad)

    # --------------------------
    # Helpers
    # --------------------------
    def _to_tensor(self, value: float) -> T.Tensor:
        return T.tensor(value, dtype=T.float32, device=self.device, requires_grad=self.require_grad)

    @staticmethod
    def _sum_wait(logs: list[Dict[str, Any]], time) -> float:
        # Robust to missing keys/None
        total = 0.0
        for log in logs:
            sub = log["subtime"]
            total += (time - sub)

        return total

    # --------------------------
    # Terms
    # --------------------------
    def wasted_energy_reward(self, monitor, next_monitor, tick_seconds) -> T.Tensor:
        """
        R1 = (next_total_waste - current_total_waste) normalized by total ECR * Δt
        Assumes each node is ACTIVE: uses its dvfs_mode to fetch ECR.
        """
        current_total_waste = sum(e.get('energy_waste')
                                  for e in monitor.energy)
        next_total_waste = sum(e.get('energy_waste')
                               for e in next_monitor.energy)
        R1 = next_total_waste - current_total_waste

        # Build index: node_id -> dvfs_profiles
        ecr_by_id: Dict[int, Dict[str, float]] = {
            e["id"]: e["dvfs_profiles"] for e in monitor.ecr}

        # Total ECR assuming nodes are active ⇒ use dvfs profile for each node's dvfs_mode
        # This will raise KeyError on unknown id/mode (prefer loud fail over silent 0).
        total_ecr = 0.0
        for n in monitor.nodes_state:
            total_ecr += float(ecr_by_id[n["id"]][n["dvfs_mode"]])

        denom = max(total_ecr * tick_seconds, 1e-9)  # avoid div/0
        normalized_R1 = (R1/64)
        # normalized_R1 = -self.alpha * (R1/32)
        log_trace(f'Wasted Energy: {normalized_R1}')
        return self._to_tensor(normalized_R1)

    def waiting_time_reward(self, next_monitor, current_time, next_time) -> T.Tensor:

        total_waiting_time = 0
        count_jobs = 0

        jobs_submission_log = next_monitor.jobs_submission_log
        jobs_submitted_ids = {job["job_id"] for job in jobs_submission_log}
        for job in jobs_submission_log:
            if current_time <= job["start_time"] <= next_time:
                total_waiting_time += job["start_time"] -job['subtime']
                count_jobs+= 1

        jobs_arrival_log = next_monitor.jobs_arrival_log

        for job in jobs_arrival_log:
            if job['job_id'] not in jobs_submitted_ids:
                total_waiting_time += (next_time -
                                       max(job['subtime'], current_time))
                count_jobs+= 1

        if count_jobs == 0:
            R2 = 0
        else:
            R2 = total_waiting_time / count_jobs

        wt = self._to_tensor(R2)
        # wt = self._to_tensor(-self.beta * R2)
        log_trace(f'Waiting Time: {wt}')

        return wt

    def calculate_reward(self, monitor, next_monitor, current_time, next_time) -> T.Tensor:
        tick_seconds = next_time-current_time
        wasted_energy = self.wasted_energy_reward(monitor, next_monitor, tick_seconds)
        waiting_time = self.waiting_time_reward(
                next_monitor, current_time, next_time) 
        reward = -(self.alpha * wasted_energy) - (self.beta * waiting_time)
        # return  reward, wasted_energy, waiting_time
        return  reward
