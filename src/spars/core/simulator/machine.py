from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


NodeId = int
NodeIdSpec = int | str | list[int | str]
NodeRecord = dict[str, Any]
NodeSpec = dict[str, Any]
TransitionMap = dict[tuple[str, str], Mapping[str, Any]]


class Machine:
    """Runtime representation of the platform nodes and their state machine."""

    def __init__(
        self,
        platform_path: str | Path,
        start_time: int | float,
    ) -> None:
        with open(platform_path, "r", encoding="utf-8") as file:
            platform_data = json.load(file)

        if not isinstance(platform_data, dict):
            raise TypeError("Platform file must contain a JSON object")

        machines = platform_data.get("machines")
        if not isinstance(machines, list):
            raise TypeError("Platform field 'machines' must be a list")

        self.platform: list[NodeSpec] = machines
        self.is_running = False
        self.total_energy_rate = 0.0

        # Runtime node state is stored per ID, while immutable platform
        # specifications and transitions are shared by node type.
        self.nodes: dict[NodeId, NodeRecord] = {}
        self.node_specs: dict[str, NodeSpec] = {}
        self.transition_map: dict[str, TransitionMap] = {}

        self.build_nodes(start_time)

        if not self.nodes:
            raise ValueError("Platform must define at least one node")

    @staticmethod
    def _validate_compute_speed(
        compute_speed: Any,
        *,
        context: str,
    ) -> float:
        try:
            numeric_speed = float(compute_speed)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{context} compute_speed must be numeric") from error

        if not math.isfinite(numeric_speed) or numeric_speed <= 0:
            raise ValueError(
                f"{context} compute_speed must be finite and greater than zero"
            )

        return numeric_speed

    @staticmethod
    def _validated_power(
        power: Any,
        *,
        context: str,
    ) -> float:
        try:
            numeric_power = float(power)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{context} power must be numeric") from error

        if not math.isfinite(numeric_power) or numeric_power < 0:
            raise ValueError(f"{context} power must be finite and non-negative")

        return numeric_power

    @staticmethod
    def _validate_transition(
        transition: Mapping[str, Any],
        *,
        context: str,
    ) -> None:
        target_state = transition.get("state")
        if not isinstance(target_state, str) or not target_state:
            raise ValueError(f"{context} transition must define a target state")

        for field in ("transition_time", "std"):
            if field not in transition:
                raise KeyError(f"{context} transition is missing {field!r}")

            try:
                numeric_value = float(transition[field])
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"{context} transition {field} must be numeric"
                ) from error

            if not math.isfinite(numeric_value):
                raise ValueError(f"{context} transition {field} must be finite")
            if numeric_value < 0:
                raise ValueError(f"{context} transition {field} cannot be negative")

    @staticmethod
    def _expand_node_ids(id_spec: NodeIdSpec) -> list[NodeId]:
        """Expand integer IDs and inclusive ``"start-end"`` ranges."""
        if isinstance(id_spec, (int, str)):
            id_spec = [id_spec]

        if not isinstance(id_spec, list):
            raise TypeError(
                "Node 'id' must be an integer, string, or list of IDs/ranges"
            )

        expanded_ids: list[NodeId] = []

        for item in id_spec:
            if isinstance(item, int):
                expanded_ids.append(item)
                continue

            if not isinstance(item, str):
                raise TypeError(f"Invalid node ID entry: {item!r}")

            normalized_id = item.strip()

            if "-" not in normalized_id:
                try:
                    expanded_ids.append(int(normalized_id))
                except ValueError as error:
                    raise ValueError(f"Invalid node ID: {normalized_id!r}") from error

                continue

            range_parts = normalized_id.split("-")

            if len(range_parts) != 2:
                raise ValueError(f"Invalid node ID range: {normalized_id!r}")

            try:
                start_id = int(range_parts[0].strip())
                end_id = int(range_parts[1].strip())
            except ValueError as error:
                raise ValueError(f"Invalid node ID range: {normalized_id!r}") from error

            if start_id > end_id:
                raise ValueError(f"Invalid descending node ID range: {normalized_id!r}")

            expanded_ids.extend(range(start_id, end_id + 1))

        return expanded_ids

    @classmethod
    def _resolve_profile_power(
        cls,
        dvfs_profile: Mapping[str, Any],
        computing: bool,
    ) -> float:
        """Resolve idle/compute power while supporting legacy profiles."""
        power_key = "power_compute" if computing else "power_idle"

        if power_key in dvfs_profile:
            power = dvfs_profile[power_key]
        elif "power" in dvfs_profile:
            power = dvfs_profile["power"]
        else:
            raise KeyError(
                f"DVFS profile must define {power_key!r} or the legacy 'power' field"
            )

        return cls._validated_power(
            power,
            context=f"DVFS profile {power_key!r}",
        )

    def _resolve_state_power(
        self,
        node_spec: Mapping[str, Any],
        state_name: str,
        dvfs_mode: str,
        computing: bool = False,
    ) -> float:
        state_definition = node_spec["states"][state_name]

        if state_definition["power"] != "from_dvfs":
            return self._validated_power(
                state_definition["power"],
                context=f"State {state_name!r}",
            )

        dvfs_profile = node_spec["dvfs_profiles"][dvfs_mode]

        return self._resolve_profile_power(
            dvfs_profile,
            computing=computing,
        )

    def _refresh_node_power(self, node: NodeRecord) -> None:
        """Refresh power after utilization or DVFS changes."""
        node_spec = self.node_specs[node["node_name"]]
        computing = node["state"] == "active" and node["job_id"] is not None

        node["power"] = self._resolve_state_power(
            node_spec=node_spec,
            state_name=node["state"],
            dvfs_mode=node["dvfs_mode"],
            computing=computing,
        )

    def build_nodes(self, start_time: int | float) -> None:
        for node_spec in self.platform:
            node_name = node_spec["node_name"]
            node_ids = self._expand_node_ids(node_spec["id"])

            if not node_ids:
                raise ValueError(f"Node group {node_name!r} contains no node IDs")

            if node_name in self.node_specs:
                raise ValueError(f"Duplicate node_name: {node_name!r}")

            self.node_specs[node_name] = node_spec

            for profile_name, profile in node_spec["dvfs_profiles"].items():
                profile["compute_speed"] = self._validate_compute_speed(
                    profile["compute_speed"],
                    context=(
                        f"DVFS profile {profile_name!r} for node type {node_name!r}"
                    ),
                )
                self._resolve_profile_power(profile, computing=False)
                self._resolve_profile_power(profile, computing=True)

            transition_map: TransitionMap = {}
            for from_state, state_data in node_spec["states"].items():
                if state_data["power"] != "from_dvfs":
                    self._validated_power(
                        state_data["power"],
                        context=(f"State {from_state!r} for node type {node_name!r}"),
                    )

                for transition in state_data.get("transitions", []):
                    self._validate_transition(
                        transition,
                        context=(f"Node type {node_name!r}, state {from_state!r}"),
                    )
                    target_state = transition["state"]
                    if target_state not in node_spec["states"]:
                        raise ValueError(
                            f"Node type {node_name!r} transition from "
                            f"{from_state!r} targets unknown state "
                            f"{target_state!r}"
                        )
                    transition_key = (
                        from_state,
                        target_state,
                    )
                    if transition_key in transition_map:
                        raise ValueError(
                            "Duplicate platform transition "
                            f"{transition_key!r} for node type {node_name!r}"
                        )
                    transition_map[transition_key] = transition

            self.transition_map[node_name] = transition_map

            dvfs_mode = node_spec["dvfs_mode"]
            active_state = node_spec["states"]["active"]
            dvfs_profile = node_spec["dvfs_profiles"][dvfs_mode]

            # Nodes start active but idle because no job is allocated yet.
            power = self._resolve_state_power(
                node_spec=node_spec,
                state_name="active",
                dvfs_mode=dvfs_mode,
                computing=False,
            )

            if active_state["compute_speed"] == "from_dvfs":
                compute_speed = dvfs_profile["compute_speed"]
            else:
                compute_speed = active_state["compute_speed"]

            compute_speed = self._validate_compute_speed(
                compute_speed,
                context=f"Node type {node_name!r}",
            )

            for node_id in node_ids:
                if node_id in self.nodes:
                    raise ValueError(f"Duplicate node ID: {node_id}")

                self.nodes[node_id] = {
                    "node_name": node_name,
                    "state": "active",
                    "state_start_time": start_time,
                    "idle_since": start_time,
                    "dvfs_mode": dvfs_mode,
                    "power": power,
                    "compute_speed": compute_speed,
                    "transitions": active_state["transitions"],
                    "job_id": None,
                    "reserved": False,
                }

    def change_dvfs_mode(
        self,
        node_ids: Iterable[NodeId],
        mode: str,
    ) -> None:
        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(f"Node {node_id} must be active to change DVFS mode")

            node_spec = self.node_specs[node["node_name"]]

            if mode not in node_spec["dvfs_profiles"]:
                raise ValueError(f"Invalid DVFS mode {mode!r} for node {node_id}")

            profile = node_spec["dvfs_profiles"][mode]
            compute_speed = self._validate_compute_speed(
                profile["compute_speed"],
                context=f"DVFS mode {mode!r} on node {node_id}",
            )
            node["dvfs_mode"] = mode
            node["compute_speed"] = compute_speed

            # Preserve whether an active node is idle or computing.
            self._refresh_node_power(node)

    def _update_node_state(
        self,
        node_id: NodeId,
        node: NodeRecord,
        new_state: str,
    ) -> None:
        node_name = node["node_name"]
        old_state = node["state"]
        transition_key = (old_state, new_state)
        node_transitions = self.transition_map[node_name]

        if transition_key not in node_transitions:
            raise RuntimeError(
                f"Invalid state transition from "
                f"{old_state!r} to {new_state!r} "
                f"on node {node_id}"
            )

        node_spec = self.node_specs[node_name]
        state_definition = node_spec["states"][new_state]
        dvfs_profile = node_spec["dvfs_profiles"][node["dvfs_mode"]]

        power = self._resolve_state_power(
            node_spec=node_spec,
            state_name=new_state,
            dvfs_mode=node["dvfs_mode"],
            computing=(new_state == "active" and node["job_id"] is not None),
        )

        if state_definition["compute_speed"] == "from_dvfs":
            compute_speed = dvfs_profile["compute_speed"]
        else:
            compute_speed = state_definition["compute_speed"]

        if new_state == "active":
            compute_speed = self._validate_compute_speed(
                compute_speed,
                context=f"Active state on node {node_id}",
            )

        node["state"] = new_state
        node["power"] = power
        node["compute_speed"] = compute_speed
        node["transitions"] = state_definition["transitions"]

    def switch_on(
        self,
        node_ids: Iterable[NodeId],
        current_time: int | float,
    ) -> None:
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "switching_on",
            )
            node["idle_since"] = None
            node["state_start_time"] = current_time
            node["job_id"] = None

    def turn_on(
        self,
        node_ids: Iterable[NodeId],
        current_time: int | float,
    ) -> None:
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "active",
            )
            node["idle_since"] = current_time
            node["state_start_time"] = current_time

    def switch_off(
        self,
        node_ids: Iterable[NodeId],
        current_time: int | float,
    ) -> None:
        for node_id, node in self._get_nodes(node_ids):
            if node["job_id"] is not None:
                raise RuntimeError(
                    f"Node {node_id} cannot be switched off — "
                    f"allocated for {node['job_id']}"
                )

            self._update_node_state(
                node_id,
                node,
                "switching_off",
            )
            node["idle_since"] = None
            node["state_start_time"] = current_time

    def turn_off(
        self,
        node_ids: Iterable[NodeId],
        current_time: int | float,
    ) -> None:
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "sleeping",
            )
            node["idle_since"] = None
            node["state_start_time"] = current_time
            node["job_id"] = None

    def reserve(self, node_ids: Iterable[NodeId]) -> None:
        for _, node in self._get_nodes(node_ids):
            node["reserved"] = True

    def allocate(
        self,
        node_ids: Iterable[NodeId],
        job_id: int,
    ) -> bool:
        if job_id is None:
            raise RuntimeError("Cannot allocate nodes — job_id is None")

        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(
                    f"Node {node_id} cannot be allocated — node is not active"
                )

            if node["job_id"] is not None:
                raise RuntimeError(
                    f"Node {node_id} is already allocated for {node['job_id']}"
                )

            node["idle_since"] = None
            node["job_id"] = job_id
            node["reserved"] = False

            # Allocation changes an active node from idle to compute power.
            self._refresh_node_power(node)

        return True

    def release(self, node_ids: Iterable[NodeId], *, current_time: float) -> None:
        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(
                    f"Node {node_id} cannot be released — state is not active"
                )

            if node["job_id"] is None:
                raise RuntimeError(
                    f"Node {node_id} cannot be released — node is not computing"
                )

            node["idle_since"] = current_time
            node["job_id"] = None
            node["reserved"] = False

            # Release changes an active node from compute to idle power.
            self._refresh_node_power(node)

    def _get_nodes(
        self,
        node_ids: Iterable[NodeId],
    ) -> list[tuple[NodeId, NodeRecord]]:
        # Materialize once so generators are validated and consumed consistently.
        requested_node_ids = list(node_ids)

        missing_node_ids = [
            node_id for node_id in requested_node_ids if node_id not in self.nodes
        ]

        if missing_node_ids:
            raise ValueError(f"Node IDs not found: {missing_node_ids}")

        return [(node_id, self.nodes[node_id]) for node_id in requested_node_ids]
