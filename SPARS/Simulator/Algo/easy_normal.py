from .fcfs_normal import FCFSNormal
from bisect import bisect_left


class EASYNormal(FCFSNormal):
    def schedule(self):
        super().prep_schedule()
        super().FCFSNormal()

        self.backfill()
        super().events_builder()
        if self.timeout is not None:
            super().timeout_policy()

        return self.events

    def backfill(self):

        if len(self.waiting_queue) > 2:
            p_job = self.waiting_queue[0]

            backfill_queue = self.waiting_queue[1:]
            next_releases = sorted(self.next_releases,
                                   key=lambda a: a['release_time'])
            last_host = next_releases[p_job['res'] - 1]
            p_start_t = last_host['release_time']

            candidates = [r['node_id']
                          for r in next_releases if r['release_time'] <= p_start_t]
            head_job_reservation = candidates[-p_job['res']:]

            for job in backfill_queue:
                available = list(self.idle)
                not_reserved = [
                    node for node in available if node['id'] not in head_job_reservation]

                if job['res'] <= len(not_reserved):
                    selected_nodes = super()._select_nodes_energy_aware(
                        job['res'], not_reserved)
                    if not selected_nodes:
                        continue
                    super().allocate(job, selected_nodes)
                else:
                    selected_nodes = self.find_node_combination(
                        max_finish_time=9999999999, job=job, candidates=available)
                    if not selected_nodes:
                        continue
                    super().allocate(job, selected_nodes)

    def find_node_combination(self, max_finish_time: float, job: dict, candidates: list):
        """
        Pick a size `job['res']` subset from `candidates` that can finish before `max_finish_time`
        """
        required_nodes = int(job.get('res'))
        if required_nodes <= 0:
            return []
        if not candidates or len(candidates) < required_nodes:
            return None

        now = float(getattr(self, "current_time"))
        required_time = float(job.get('reqtime'))

        # Build node_id -> release_time from the resource agenda
        next_releases_by_id = self._releases_by_id()
        release_times_by_node_id = {
            int(node_id): float(entry.get('release_time'))
            for node_id, entry in next_releases_by_id.items()
        }

        # Normalize candidates and filter obviously infeasible ones
        normalized = []  # each: dict with node, node_id, speed, power, release_time, remaining_idle_timeout
        for node in candidates:
            node_id = int(node['id'])
            speed = float(node.get('compute_speed'))
            if speed <= 0.0:
                continue
            release_time = float(release_times_by_node_id.get(
                node_id, node.get('release_time', now)))
            if required_time > 0.0 and release_time >= float(max_finish_time):
                # If it can't even start before the deadline for a positive required_time, skip
                continue
            power = float(node.get('power'))
            remaining_idle_timeout = float(
                super()._remaining_idle_timeout(node_id))
            normalized.append({
                'node': node,
                'node_id': node_id,
                'speed': speed,
                'power': power,
                'release_time': release_time,
                'remaining_idle_timeout': remaining_idle_timeout
            })

        if len(normalized) < required_nodes:
            return None

        # If no work is required, just ensure start_time <= deadline and pick by tie-breaks
        if required_time <= 0.0:
            pool = [x for x in normalized if x['release_time']
                    <= float(max_finish_time)]
            if len(pool) < required_nodes:
                return None
            pool.sort(key=lambda x: (
                x['power'], x['remaining_idle_timeout'], x['node_id']))
            return [x['node'] for x in pool[:required_nodes]]

        # Sweep unique cutoff release times (ascending) up to the deadline
        normalized.sort(key=lambda x: x['release_time'])
        cutoff_release_times = sorted({
            x['release_time'] for x in normalized
            if x['release_time'] <= float(max_finish_time)
        })

        # Maintain eligible nodes (release_time <= cutoff) sorted by speed ascending for bisect
        # tuples: (speed, power, node_id, release_time, remaining_idle_timeout, node)
        eligible_nodes = []
        eligible_speeds = []      # parallel list of speeds for bisect
        scan_index = 0

        best_key = None
        best_selection = None

        for cutoff_release_time in cutoff_release_times:
            # Extend eligible set with all nodes that have release_time <= cutoff
            while scan_index < len(normalized) and normalized[scan_index]['release_time'] <= cutoff_release_time:
                item = normalized[scan_index]
                insert_pos = bisect_left(eligible_speeds, item['speed'])
                eligible_speeds.insert(insert_pos, item['speed'])
                eligible_nodes.insert(insert_pos, (
                    item['speed'],
                    item['power'],
                    item['node_id'],
                    item['release_time'],
                    item['remaining_idle_timeout'],
                    item['node']
                ))
                scan_index += 1

            time_budget = float(max_finish_time) - float(cutoff_release_time)
            if time_budget <= 0.0:
                continue

            min_speed_required = required_time / time_budget
            threshold_pos = bisect_left(eligible_speeds, min_speed_required)
            available = len(eligible_speeds) - threshold_pos
            if available < required_nodes:
                continue

            # Choose the required number of slowest nodes that still meet the speed threshold
            chosen = eligible_nodes[threshold_pos: threshold_pos + required_nodes]

            # Verify timing (should be feasible by construction, but keep strict)
            speeds = [t[0] for t in chosen]
            powers = [t[1] for t in chosen]
            node_ids = [t[2] for t in chosen]
            release_times = [t[3] for t in chosen]
            remaining_idle_timeouts = [t[4] for t in chosen]

            min_speed = min(speeds)
            if min_speed <= 0.0:
                continue
            start_time = max(release_times)
            finish_time = start_time + (required_time / min_speed)
            if finish_time > float(max_finish_time):
                continue

            total_power = sum(powers)
            energy_score = total_power / min_speed
            total_remaining_idle_timeout = sum(remaining_idle_timeouts)
            node_ids_sorted = sorted(node_ids)

            candidate_key = (energy_score, total_remaining_idle_timeout,
                             total_power, node_ids_sorted)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_selection = [t[5] for t in chosen]

        return best_selection
