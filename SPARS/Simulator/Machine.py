import json


class Machine:
    def __init__(self, platform_path, start_time):
        with open(platform_path, "r", encoding="utf-8") as file:
            self.platform = json.load(file)["machines"]

        self.is_running = False
        self.total_energy_rate = 0.0

        # Individual runtime nodes:
        # node_id -> node runtime data
        self.nodes = {}

        # Shared specifications:
        # node_name -> node specification
        self.node_specs = {}

        # Shared transition definitions:
        # node_name -> {(from_state, to_state): transition_detail}
        self.transition_map = {}

        self.build_nodes(start_time)

    @staticmethod
    def _expand_node_ids(id_spec):
        """
        Expand compressed node IDs.

        Examples:
            ["0-10", "13-14", 15, "18-20"]
            -> [0, 1, 2, ..., 10, 13, 14, 15, 18, 19, 20]

            "0-15"
            -> [0, 1, ..., 15]

            5
            -> [5]
        """
        if isinstance(id_spec, (int, str)):
            id_spec = [id_spec]

        if not isinstance(id_spec, list):
            raise TypeError(
                "Node 'id' must be an integer, string, or list of IDs/ranges"
            )

        expanded_ids = []

        for item in id_spec:
            if isinstance(item, int):
                expanded_ids.append(item)
                continue

            if not isinstance(item, str):
                raise TypeError(
                    f"Invalid node ID entry: {item!r}"
                )

            item = item.strip()

            if "-" not in item:
                try:
                    expanded_ids.append(int(item))
                except ValueError as error:
                    raise ValueError(
                        f"Invalid node ID: {item!r}"
                    ) from error

                continue

            parts = item.split("-")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid node ID range: {item!r}"
                )

            try:
                start_id = int(parts[0].strip())
                end_id = int(parts[1].strip())
            except ValueError as error:
                raise ValueError(
                    f"Invalid node ID range: {item!r}"
                ) from error

            if start_id > end_id:
                raise ValueError(
                    f"Invalid descending node ID range: {item!r}"
                )

            expanded_ids.extend(
                range(start_id, end_id + 1)
            )

        return expanded_ids

    @staticmethod
    def _resolve_profile_power(dvfs_profile, computing):
        """
        Resolve power from a DVFS profile.

        New format:
            power_idle
            power_compute

        Legacy format:
            power

        The legacy fallback keeps older platform files working.
        """
        power_key = "power_compute" if computing else "power_idle"

        if power_key in dvfs_profile:
            return dvfs_profile[power_key]

        if "power" in dvfs_profile:
            return dvfs_profile["power"]

        raise KeyError(
            f"DVFS profile must define {power_key!r} "
            "or the legacy 'power' field"
        )

    def _resolve_state_power(
        self,
        node_spec,
        state_name,
        dvfs_mode,
        computing=False,
    ):
        state_def = node_spec["states"][state_name]

        if state_def["power"] != "from_dvfs":
            return state_def["power"]

        dvfs_profile = node_spec["dvfs_profiles"][dvfs_mode]

        return self._resolve_profile_power(
            dvfs_profile,
            computing=computing,
        )

    def _refresh_node_power(self, node):
        """
        Refresh a node's current power after allocation, release,
        DVFS changes, or another change that affects utilization.
        """
        node_spec = self.node_specs[node["node_name"]]

        computing = (
            node["state"] == "active"
            and node["job_id"] is not None
        )

        node["power"] = self._resolve_state_power(
            node_spec=node_spec,
            state_name=node["state"],
            dvfs_mode=node["dvfs_mode"],
            computing=computing,
        )

    def build_nodes(self, start_time):
        for node_spec in self.platform:
            node_name = node_spec["node_name"]
            node_ids = self._expand_node_ids(node_spec["id"])

            if not node_ids:
                raise ValueError(
                    f"Node group {node_name!r} contains no node IDs"
                )

            if node_name in self.node_specs:
                raise ValueError(
                    f"Duplicate node_name: {node_name!r}"
                )

            # Store one shared specification for this node type.
            self.node_specs[node_name] = node_spec

            # Store one shared transition map for this node type.
            self.transition_map[node_name] = {
                (from_state, transition["state"]): transition
                for from_state, state_data
                in node_spec["states"].items()
                for transition
                in state_data.get("transitions", [])
            }

            dvfs_mode = node_spec["dvfs_mode"]
            active_state = node_spec["states"]["active"]
            dvfs_profile = node_spec["dvfs_profiles"][dvfs_mode]

            # Nodes start active but idle because job_id is None.
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

            # Build an individual runtime node for every expanded ID.
            for node_id in node_ids:
                if node_id in self.nodes:
                    raise ValueError(
                        f"Duplicate node ID: {node_id}"
                    )

                self.nodes[node_id] = {
                    "node_name": node_name,
                    "state": "active",
                    "state_start_time": start_time,
                    "dvfs_mode": dvfs_mode,
                    "power": power,
                    "compute_speed": compute_speed,
                    "transitions": active_state["transitions"],
                    "job_id": None,
                    "reserved": False,
                }

    def change_dvfs_mode(self, node_ids, mode):
        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(
                    f"Node {node_id} must be active "
                    "to change DVFS mode"
                )

            node_name = node["node_name"]
            node_spec = self.node_specs[node_name]

            if mode not in node_spec["dvfs_profiles"]:
                raise ValueError(
                    f"Invalid DVFS mode {mode!r} "
                    f"for node {node_id}"
                )

            profile = node_spec["dvfs_profiles"][mode]

            node["dvfs_mode"] = mode
            node["compute_speed"] = profile["compute_speed"]

            # Keep the node on idle or compute power after changing DVFS.
            self._refresh_node_power(node)

    def _update_node_state(
        self,
        node_id,
        node,
        new_state,
    ):
        node_name = node["node_name"]
        old_state = node["state"]

        transition_key = (
            old_state,
            new_state,
        )

        node_transitions = self.transition_map[node_name]

        if transition_key not in node_transitions:
            raise RuntimeError(
                f"Invalid state transition from "
                f"{old_state!r} to {new_state!r} "
                f"on node {node_id}"
            )

        node_spec = self.node_specs[node_name]
        state_def = node_spec["states"][new_state]

        dvfs_profile = node_spec["dvfs_profiles"][
            node["dvfs_mode"]
        ]

        power = self._resolve_state_power(
            node_spec=node_spec,
            state_name=new_state,
            dvfs_mode=node["dvfs_mode"],
            computing=(
                new_state == "active"
                and node["job_id"] is not None
            ),
        )

        if state_def["compute_speed"] == "from_dvfs":
            compute_speed = dvfs_profile["compute_speed"]
        else:
            compute_speed = state_def["compute_speed"]

        node["state"] = new_state
        node["power"] = power
        node["compute_speed"] = compute_speed
        node["transitions"] = state_def["transitions"]

    def switch_on(self, node_ids, current_time):
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "switching_on",
            )

            node["state_start_time"] = current_time
            node["job_id"] = None

    def turn_on(self, node_ids, current_time):
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "active",
            )

            node["state_start_time"] = current_time

    def switch_off(self, node_ids, current_time):
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

            node["state_start_time"] = current_time

    def turn_off(self, node_ids, current_time):
        for node_id, node in self._get_nodes(node_ids):
            self._update_node_state(
                node_id,
                node,
                "sleeping",
            )

            node["state_start_time"] = current_time
            node["job_id"] = None

    def reserve(self, node_ids):
        for _, node in self._get_nodes(node_ids):
            node["reserved"] = True

    def allocate(self, node_ids, job_id):
        if job_id is None:
            raise RuntimeError(
                "Cannot allocate nodes — job_id is None"
            )

        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(
                    f"Node {node_id} cannot be allocated — "
                    "node is not active"
                )

            if node["job_id"] is not None:
                raise RuntimeError(
                    f"Node {node_id} is already allocated "
                    f"for {node['job_id']}"
                )

            node["job_id"] = job_id
            node["reserved"] = False

            # Active node is now computing.
            self._refresh_node_power(node)

        return True

    def release(self, node_ids):
        for node_id, node in self._get_nodes(node_ids):
            if node["state"] != "active":
                raise RuntimeError(
                    f"Node {node_id} cannot be released — "
                    "state is not active"
                )

            if node["job_id"] is None:
                raise RuntimeError(
                    f"Node {node_id} cannot be released — "
                    "node is not computing"
                )

            node["job_id"] = None
            node["reserved"] = False

            # Active node is now idle.
            self._refresh_node_power(node)

    def _get_nodes(self, node_ids):
        missing = [
            node_id
            for node_id in node_ids
            if node_id not in self.nodes
        ]

        if missing:
            raise ValueError(
                f"Node IDs not found: {missing}"
            )

        return [
            (node_id, self.nodes[node_id])
            for node_id in node_ids
        ]