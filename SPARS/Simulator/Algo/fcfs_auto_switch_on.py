import math
from .BaseAlgorithm import BaseAlgorithm


class FCFS(BaseAlgorithm):
    """
    First-Come-First-Served using only IDLE nodes.

    Node selection is energy-aware:
      Minimize ( sum(power) / min(compute_speed) ).
    Tie-breaks:
      1) Shorter remaining idle-timeout first (closer to switch-off => pick sooner)
      2) Lower total power
      3) Lexicographically smaller node-id list
    Assumes each node has 'compute_speed' and 'power'.
    """

    # ---------- public ----------
    def schedule(self):
        super().prep_schedule()
        self.FCFSAuto()

        super().events_builder()
        if self.timeout is not None:
            super().timeout_policy()
        return self.events

    def FCFSAuto(self):
        for job in self.waiting_queue[:]:  # snapshot to avoid iterator issues
            required = int(job["res"])

            # 1) Prefer ACTIVE & IDLE
            candidates = list(self.idle)

            # 2) If not enough, allow SLEEPING nodes as fallback
            if len(candidates) < required:
                # these are non-reserved by construction
                candidates += list(self.sleeping)

            if len(candidates) < required:
                break  # FCFS: stop at the first unsatisfied job

            selected = self._select_nodes_energy_aware(required, candidates)
            if not selected:
                break

            super().allocate(job, selected)  # will add sleep_to_active for sleepers

    # ---------- internals ----------
    def _remaining_idle_timeout(self, node_id: int) -> float:
        """
        Remaining time until this idle node would be switched off by timeout_policy.
        If not tracked, return a large number so it sorts to the end.
        """
        if self.timeout is None:
            return math.inf

        for entry in self.timeout_list:
            if entry["node_id"] == node_id:
                remaining = float(entry["time"] - self.current_time)
                return remaining

        return math.inf

    def _select_nodes_energy_aware(self, required_nodes: int, _candidates):
        """
        Choose 'required_nodes' from candidates to minimize:
            score = (sum power) / min speed
        Implementation:
          - Gather unique speed thresholds among candidates nodes.
          - For each threshold s: keep candidates with speed >= s.
          - If at least 'required_nodes' exist: pick the 'required_nodes' lowest-power nodes.
          - Compute score = sum(power) / s and keep the best set.
        Tie-breaks:
          - Shorter sum of remaining candidate timeout (prefer nodes closer to switch-off)
          - Lower total power
          - Smaller node id list
        """
        if len(_candidates) < required_nodes:
            return None

        # Normalize records
        normalized = []

        for node in _candidates:
            speed = float(node.get("compute_speed"))
            power = float(node.get("power"))
            rem_timeout = self._remaining_idle_timeout(node["id"])
            normalized.append({
                "node": node,
                "speed": speed,
                "power": power,
                "remaining_timeout": rem_timeout,
            })

        # Unique speeds (thresholds), high to low
        speed_levels = sorted({item["speed"]
                              for item in normalized}, reverse=True)

        best_key = None
        best_pick = None

        for threshold_speed in speed_levels:
            candidates = [
                item for item in normalized if item["speed"] >= threshold_speed]
            if len(candidates) < required_nodes:
                continue

            # Pick lowest-power nodes first; if same power, pick those closer to timeout
            candidates.sort(key=lambda it: (
                it["power"], it["remaining_timeout"], it["node"]["id"]))
            picked = candidates[:required_nodes]

            total_power = sum(it["power"] for it in picked)
            # runtime cancels for comparisons; use threshold as the bottleneck speed

            energy_score = total_power / threshold_speed

            total_remaining_timeout = sum(
                it["remaining_timeout"] for it in picked)
            id_list = sorted(it["node"]["id"] for it in picked)

            # Compare by (energy_score, total_remaining_timeout, total_power, ids)
            candidate_key = (
                energy_score, total_remaining_timeout, total_power, id_list)

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_pick = [it["node"] for it in picked]

        return best_pick
