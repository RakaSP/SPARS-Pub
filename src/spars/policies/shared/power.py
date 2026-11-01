"""Node power lookups that schedulers and power managers both need."""

from typing import Any


def resolve_dvfs_power(machine: "dict[str, Any]", node: "dict[str, Any]", power_type):
    """Idle or compute power of the node's current DVFS profile."""
    if power_type == "idle":
        power_key = "power_idle"
    elif power_type == "compute":
        power_key = "power_compute"
    else:
        raise ValueError("power_type must be either 'idle' or 'compute'")

    dvfs_mode = node["dvfs_mode"]
    profile = machine["dvfs_profiles"][dvfs_mode]
    if power_key in profile:
        return float(profile[power_key])
    # Older platform files have one "power" value for both.
    if "power" in profile:
        return float(profile["power"])
    raise KeyError(
        f"DVFS profile {dvfs_mode!r} must define {power_key!r} or 'power'"
    )


class NodePower:
    """Node power lookups shared by schedulers and power managers.

    Needs self.machines (nodes, node_specs) and self.state.
    """

    _resolve_dvfs_power = staticmethod(resolve_dvfs_power)

    def _machine_for(self, node_id: int):
        node_name = self.machines.nodes[node_id]["node_name"]
        return self.machines.node_specs[node_name]

    def _state_power(self, node_id: int, state_name: str) -> float:
        node = self.state[node_id]
        machine = self._machine_for(node_id)
        value = machine["states"][state_name]["power"]
        if value == "from_dvfs":
            # Non-compute phases count as waste and use the idle DVFS power.
            return self._resolve_dvfs_power(machine, node, "idle")
        return float(value)

    def _idle_power(self, node_id: int) -> float:
        return self._state_power(node_id, "active")
