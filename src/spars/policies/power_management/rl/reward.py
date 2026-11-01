"""Shared incremental cost: total wasted energy and aggregate waiting time."""

import math
from types import SimpleNamespace
from typing import Any

import torch as T

from spars.core.logging import log_trace
from spars.core.runner.agent import resolve_device




class Reward:
    schema = "cumulative-energy-wait-v2"

    def __init__(
        self,
        weight1=0.5,
        weight2=0.5,
        device="gpu",
        require_grad=False,
        reference_seconds=3600.0,
        energy_reference_joules=None,
    ):
        self.weight1 = float(weight1)
        self.weight2 = float(weight2)
        self.reference_seconds = float(reference_seconds)
        self.energy_reference_joules = (
            None if energy_reference_joules is None else float(energy_reference_joules)
        )
        for name, value in (("weight1", self.weight1), ("weight2", self.weight2)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.reference_seconds) or self.reference_seconds <= 0:
            raise ValueError("reference_seconds must be finite and positive")
        if self.energy_reference_joules is not None and (
            not math.isfinite(self.energy_reference_joules)
            or self.energy_reference_joules <= 0
        ):
            raise ValueError("energy_reference_joules must be finite and positive")
        self.device = T.device(resolve_device(device))
        self.require_grad = bool(require_grad)

    def scales(self, machine):
        """Fixed denominators, independent of jobs, actions and interval length."""
        energy = self.energy_reference_joules
        if energy is None:
            reference_power = 0.0
            for node in machine.nodes.values():
                spec = machine.node_specs[node["node_name"]]
                powers = [
                    float(profile["power_idle"])
                    for profile in spec["dvfs_profiles"].values()
                ]
                powers.extend(
                    float(state["power"])
                    for state in spec["states"].values()
                    if state["power"] != "from_dvfs"
                )
                reference_power += max(powers)
            # Explicit unit scale also supports zero-power test platforms.
            energy = max(1.0, reference_power * self.reference_seconds)
        return {"energy_joules": energy, "waiting_seconds": self.reference_seconds}

    def snapshot(self, simulator):
        """Capture only the pre-action state ``components`` reads.

        Replaces a deep copy of the simulator. ``window`` lets the waiting sum
        skip jobs that started before the interval: their terms are exactly
        0.0, so the result is bit-identical to scanning every arrival.
        """
        monitor = simulator.monitor
        return SimpleNamespace(
            current_time=simulator.current_time,
            platform_control=simulator.platform_control,
            monitor=SimpleNamespace(
                energy={
                    node_id: {"energy_waste": entry["energy_waste"]}
                    for node_id, entry in monitor.energy.items()
                }
            ),
            window=SimpleNamespace(
                # Arrival order; jobs are never requeued after starting.
                waiting=list(simulator.jobs_manager.waiting_queue),
                arrivals=len(monitor.jobs_arrival_log),
                submissions=len(monitor.jobs_submission_log),
            ),
        )

    def components(self, simulator, next_simulator):
        begin = float(simulator.current_time)
        end = float(next_simulator.current_time)
        if end < begin:
            raise ValueError("Reward interval cannot go backwards")
        scales = self.scales(simulator.platform_control.machine)
        energy = sum(
            entry["energy_waste"] - simulator.monitor.energy[node_id]["energy_waste"]
            for node_id, entry in next_simulator.monitor.energy.items()
        )
        arrivals = next_simulator.monitor.jobs_arrival_log
        submissions = next_simulator.monitor.jobs_submission_log
        window = getattr(simulator, "window", None)
        if window is not None:
            arrivals = window.waiting + arrivals[window.arrivals :]
            submissions = submissions[window.submissions :]
        starts = {job["job_id"]: float(job["start_time"]) for job in submissions}
        waiting = sum(
            max(
                0.0,
                min(end, starts.get(job["job_id"], end))
                - max(begin, float(job["subtime"])),
            )
            for job in arrivals
        )
        if energy < -1e-6:
            raise ValueError("Cumulative wasted energy cannot decrease")
        return {
            "wasted_energy_joules": max(0.0, energy),
            "waiting_time_seconds": waiting,
            "energy_cost": max(0.0, energy) / scales["energy_joules"],
            "waiting_cost": waiting / scales["waiting_seconds"],
        }

    def cost(self, parts) -> float:
        """Weighted, scaled cost of one interval (the negated plain reward)."""
        return self.weight1 * parts["energy_cost"] + self.weight2 * parts["waiting_cost"]

    def calculate_reward(
        self, obs: Any, next_obs: Any, simulator: Any, next_simulator: Any, actions: Any
    ):
        parts = self.components(simulator, next_simulator)
        value = (
            -self.weight1 * parts["energy_cost"] - self.weight2 * parts["waiting_cost"]
        )
        log_trace(
            "Reward components: waste_J=%.6f waiting_s=%.6f energy_cost=%.6f waiting_cost=%.6f reward=%.6f",
            parts["wasted_energy_joules"],
            parts["waiting_time_seconds"],
            parts["energy_cost"],
            parts["waiting_cost"],
            value,
        )
        return T.tensor(
            value, dtype=T.float32, device=self.device, requires_grad=self.require_grad
        )
