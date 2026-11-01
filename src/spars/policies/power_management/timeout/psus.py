from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from spars.core.decision import SchedulingDecision

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.jobs_manager import JobsManager


class PSUSTimeoutPowerManager:
    """Switches a node off after it has been idle for the timeout; pair with a PSUS scheduler.

    PSUS reservations are supplied by the scheduler through
    ``SchedulingDecision.reserved_nodes``. Reserved nodes are kept powered on,
    matching the v1 behavior, while idle unreserved nodes are switched off
    after the configured timeout.
    """

    # timeout: idle seconds before a node is switched off; None never switches off.
    DEFAULTS = {"timeout": None}

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        start_time: float,
        config: dict[str, Any],
    ) -> None:
        unexpected_config = set(config) - {"timeout"}
        if unexpected_config:
            raise ValueError(
                "Invalid PSUS timeout power-manager config: "
                f"unexpected keys: {sorted(unexpected_config)}"
            )

        del jobs_manager
        self.state = machines.nodes
        self.current_time = start_time
        self.timeout = config.get("timeout")
        self.timeout_list: list[dict[str, float | int]] = []
        self.next_timeout_at: float | None = None

    def prepare(self, current_time: float) -> None:
        # Legacy PSUS updated timeout tracking after scheduling, not before it.
        self.current_time = current_time
        if not self.timeout:
            self.timeout_list = []
            self.next_timeout_at = None

    def remaining_idle_timeout(
        self,
        node_id: int,
        current_time: float,
    ) -> float:
        if not self.timeout:
            return math.inf

        for entry in self.timeout_list:
            if int(entry["node_id"]) == int(node_id):
                return float(entry["time"]) - float(current_time)

        return math.inf

    @staticmethod
    def _reserved_nodes(
        scheduling_decision: SchedulingDecision,
    ) -> set[int]:
        reserved = set(scheduling_decision.reserved_nodes)

        for event_bucket in scheduling_decision.events:
            for event in event_bucket["events"]:
                if event.get("type") == "execution_start":
                    reserved.update(event.get("nodes", []))

        return reserved

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(scheduling_decision, SchedulingDecision):
            raise TypeError(
                "scheduling_decision must be a SchedulingDecision"
            )

        self.current_time = current_time
        self_events: list[dict[str, Any]] = []

        if not self.timeout:
            return ()

        now = self.current_time
        expire_at = now + self.timeout
        allocated_ids = self._reserved_nodes(scheduling_decision)

        timeout_ids = {
            int(entry["node_id"])
            for entry in self.timeout_list
        }

        if timeout_ids & allocated_ids:
            self.timeout_list = [
                entry
                for entry in self.timeout_list
                if int(entry["node_id"]) not in allocated_ids
            ]
            timeout_ids -= allocated_ids

        for node_id, node in self.state.items():
            idle = (
                node["state"] == "active"
                and node.get("job_id") is None
            )
            if (
                idle
                and node_id not in allocated_ids
                and node_id not in timeout_ids
            ):
                self.timeout_list.append({
                    "node_id": node_id,
                    "time": expire_at,
                })
                timeout_ids.add(node_id)

        keep: list[dict[str, float | int]] = []
        switch_off: list[int] = []
        next_earliest: float | int | None = None

        for entry in self.timeout_list:
            node_id = int(entry["node_id"])
            node = self.state.get(node_id)

            if node is None or node_id in allocated_ids:
                continue

            idle = (
                node["state"] == "active"
                and node.get("job_id") is None
            )
            if not idle:
                continue

            deadline = entry["time"]
            if now < deadline:
                keep.append(entry)
                next_earliest = (
                    deadline
                    if next_earliest is None
                    else min(next_earliest, deadline)
                )
            else:
                switch_off.append(node_id)

        self.timeout_list = keep

        if switch_off:
            self_events.append({
                "timestamp": now,
                "events": [
                    {
                        "type": "switch_off",
                        "nodes": switch_off,
                    }
                ],
            })

        if (
            next_earliest is not None
            and self.next_timeout_at != next_earliest
        ):
            self_events.append({
                "timestamp": next_earliest,
                "events": [{"type": "call_me_later"}],
            })
            self.next_timeout_at = next_earliest

        return tuple(self_events)
