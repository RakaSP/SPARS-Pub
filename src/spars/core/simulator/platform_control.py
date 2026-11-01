import math
from typing import Any

import numpy as np

from .machine import Machine


_RANDOM_SEED = 42
VALID_OVERRUN_POLICIES = frozenset({"terminate", "continue"})


class PlatformControl:
    def __init__(
        self,
        platform_path: str,
        overrun_policy: str,
        start_time: float,
    ) -> None:
        if overrun_policy not in VALID_OVERRUN_POLICIES:
            raise ValueError(
                f"Unknown overrun policy: {overrun_policy!r}. "
                f"Expected one of {sorted(VALID_OVERRUN_POLICIES)}"
            )

        self.machine = Machine(platform_path, start_time)
        self.overrun_policy = overrun_policy
        self.exact = False
        self.seed = _RANDOM_SEED
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def validate_duplication(node_ids: list[int]) -> None:
        if len(node_ids) == len(set(node_ids)):
            return

        seen: set[int] = set()
        duplicates = []

        for node_id in node_ids:
            if node_id in seen:
                duplicates.append(node_id)
            else:
                seen.add(node_id)

        raise RuntimeError(f"Duplicate node_ids found: {duplicates}")

    def sample_transition_time(
        self,
        transition: dict[str, Any],
    ) -> float:
        try:
            mean = float(transition["transition_time"])
            std = float(transition["std"])
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Transition time and standard deviation must be numeric"
            ) from error

        if not math.isfinite(mean):
            raise ValueError("Transition time must be finite")
        if mean < 0:
            raise ValueError("Transition time cannot be negative")
        if not math.isfinite(std):
            raise ValueError("Transition standard deviation must be finite")
        if std < 0:
            raise ValueError("Transition standard deviation cannot be negative")

        if self.exact or std == 0:
            return mean

        # scipy.stats.norm(...).rvs(random_state=Generator) ultimately
        # delegates to Generator.normal(). Calling the generator directly
        # preserves the exact random stream while avoiding construction of a
        # frozen scipy distribution for every power transition.
        return float(self.rng.normal(loc=mean, scale=std))

    def _execution_finish_details(
        self,
        node_ids: list[int],
        job: dict[str, Any],
        current_time: float,
    ) -> tuple[float, float, dict[str, Any]]:
        compute_speed = min(
            self.machine.nodes[node_id]["compute_speed"] for node_id in node_ids
        )
        if compute_speed <= 0:
            raise RuntimeError("compute_speed must be greater than zero")

        actual_finish_time = current_time + (job["runtime"] / compute_speed)
        requested_finish_time = current_time + (job["reqtime"] / compute_speed)

        event = {
            "job_id": job["job_id"],
            "type": "execution_finished",
            "res": job["res"],
            "nodes": node_ids,
            "start_time": current_time,
            "subtime": job["subtime"],
            "reqtime": job["reqtime"],
            "req_finish_time": requested_finish_time,
            "runtime": job["runtime"],
            "actual_finish_time": actual_finish_time,
            "user_id": job["user_id"],
        }

        return requested_finish_time, actual_finish_time, event

    def compute(
        self,
        node_ids: list[int],
        job: dict[str, Any],
        current_time: float,
    ) -> tuple[float, dict[str, Any]]:
        self.validate_duplication(node_ids)

        if len(node_ids) != job["res"]:
            raise RuntimeError(
                f"Resource allocation mismatch for job {job['job_id']!r}: "
                f"Requested {job['res']} nodes, but allocated "
                f"{len(node_ids)}."
            )

        success = self.machine.allocate(
            node_ids,
            job["job_id"],
        )

        if not success:
            raise RuntimeError(f"Job {job['job_id']} failed to execute")

        (
            requested_finish_time,
            actual_finish_time,
            event,
        ) = self._execution_finish_details(
            node_ids,
            job,
            current_time,
        )

        if self.overrun_policy == "terminate":
            finish_time = min(
                requested_finish_time,
                actual_finish_time,
            )
            return finish_time, event

        # "continue" preserves the existing behavior: the job finishes at
        # its actual runtime even when that is earlier than the request.
        return actual_finish_time, event

    def change_dvfs_mode(
        self,
        node_ids: list[int],
        mode: str,
    ) -> dict[str, Any]:
        self.validate_duplication(node_ids)
        self.machine.change_dvfs_mode(node_ids, mode)

        return {
            "type": "change_dvfs_mode",
            "node": node_ids,
            "mode": mode,
        }

    def release(
        self,
        event: dict[str, Any],
        current_time: float,
    ) -> bool:
        terminated = current_time < event["actual_finish_time"]
        self.machine.release(event["nodes"], current_time=current_time)
        return terminated

    def reserve_node(self, node_ids: list[int]) -> None:
        self.validate_duplication(node_ids)
        self.machine.reserve(node_ids)

    def turn_on(
        self,
        node_ids: list[int],
        current_time: float,
    ) -> None:
        self.validate_duplication(node_ids)
        self.machine.turn_on(node_ids, current_time)

    def turn_off(
        self,
        node_ids: list[int],
        current_time: float,
    ) -> None:
        self.validate_duplication(node_ids)
        self.machine.turn_off(node_ids, current_time)

    def switch_off(
        self,
        node_ids: list[int],
        current_time: float,
        oracle_durations: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_duplication(node_ids)
        self.machine.switch_off(node_ids, current_time)

        turn_off_times: dict[float, list[int]] = {}

        for node_id in node_ids:
            node_name = self.machine.nodes[node_id]["node_name"]
            transitions = self.machine.transition_map[node_name]

            active_to_switching_off = transitions[("active", "switching_off")]
            switching_off_to_sleeping = transitions[("switching_off", "sleeping")]

            if oracle_durations is not None and node_id in oracle_durations:
                total_duration = float(oracle_durations[node_id])
            else:
                total_duration = self.sample_transition_time(
                    active_to_switching_off
                ) + self.sample_transition_time(switching_off_to_sleeping)

            completed_at = current_time + total_duration
            turn_off_times.setdefault(
                completed_at,
                [],
            ).append(node_id)

        return [
            {
                "event": {
                    "type": "turn_off",
                    "nodes": nodes,
                },
                "timestamp": timestamp,
            }
            for timestamp, nodes in turn_off_times.items()
        ]

    def switch_on(
        self,
        node_ids: list[int],
        current_time: float,
        oracle_durations: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_duplication(node_ids)
        self.machine.switch_on(node_ids, current_time)

        turn_on_times: dict[float, list[int]] = {}

        for node_id in node_ids:
            node_name = self.machine.nodes[node_id]["node_name"]
            transitions = self.machine.transition_map[node_name]

            sleeping_to_switching_on = transitions[("sleeping", "switching_on")]
            switching_on_to_active = transitions[("switching_on", "active")]

            if oracle_durations is not None and node_id in oracle_durations:
                total_duration = float(oracle_durations[node_id])
            else:
                total_duration = self.sample_transition_time(
                    sleeping_to_switching_on
                ) + self.sample_transition_time(switching_on_to_active)

            completed_at = current_time + total_duration
            turn_on_times.setdefault(
                completed_at,
                [],
            ).append(node_id)

        return [
            {
                "event": {
                    "type": "turn_on",
                    "nodes": nodes,
                },
                "timestamp": timestamp,
            }
            for timestamp, nodes in turn_on_times.items()
        ]
