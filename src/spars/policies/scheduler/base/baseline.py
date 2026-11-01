from __future__ import annotations

from typing import TYPE_CHECKING

from spars.core.decision import SchedulingDecision
from spars.core.registry import validate_component_config

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager


class BaselineScheduler:
    """Start whatever can start right now on idle nodes, lowest node ids first.

    A subclass only says in which order the waiting jobs are tried
    (``order``). Jobs are tried in that order until one does not fit on the
    idle nodes. There is no plan for the future and no power handling: nodes
    that are not on are never used, so pair it with the ``none`` power manager.
    """

    def __init__(self, machines: "Machine", jobs_manager: "JobsManager", start_time, config):
        validate_component_config(config, type(self).__name__, set())
        self.state = machines.nodes
        self.jobs_manager = jobs_manager
        self.current_time = float(start_time)

    def set_time(self, current_time: float) -> None:
        self.current_time = float(current_time)

    def order(self, jobs: list[dict]) -> list[dict]:
        raise NotImplementedError

    def schedule(self) -> SchedulingDecision:
        idle = sorted(
            node_id
            for node_id, node in self.state.items()
            if node["state"] == "active" and node.get("job_id") is None
        )
        starts = self.start_jobs(idle)
        if not starts:
            return SchedulingDecision()
        events = [
            {**job, "type": "execution_start", "nodes": nodes} for job, nodes in starts
        ]
        return SchedulingDecision(
            events=({"timestamp": self.current_time, "events": events},),
        )

    def start_jobs(self, idle: list[int]) -> list[tuple[dict, list[int]]]:
        """The jobs to start now with their nodes; takes nodes off ``idle``."""
        starts = []
        for job in self.order(list(self.jobs_manager.waiting_queue)):
            if job["res"] > len(idle):
                break
            starts.append((job, idle[: job["res"]]))
            del idle[: job["res"]]
        return starts
