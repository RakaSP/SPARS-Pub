import math
import re
from itertools import islice
from bisect import bisect_left, bisect_right, insort
from .fcfs_psas import FCFSPSAS

_COMPUTE_RE = re.compile(r"^compute\(job=\d+\)$")
EPS = 1e-9

class _GapQueueIndex:
    __slots__ = ("queue", "starts", "prefix_max_finish")

    def __init__(self, queue):
        self.queue = sorted(
            queue,
            key=lambda seg: float(seg["start_time"]),
        )
        self.starts = [
            float(seg["start_time"])
            for seg in self.queue
        ]
        self.prefix_max_finish = []
        max_finish = -math.inf
        for seg in self.queue:
            max_finish = max(
                max_finish,
                float(seg["finish_time"]),
            )
            self.prefix_max_finish.append(max_finish)

    def insert(self, seg):
        start = float(seg["start_time"])
        index = bisect_right(self.starts, start)
        self.queue.insert(index, seg)
        self.starts.insert(index, start)
        self.prefix_max_finish.insert(index, -math.inf)

        max_finish = (
            self.prefix_max_finish[index - 1]
            if index > 0
            else -math.inf
        )
        for pos in range(index, len(self.queue)):
            max_finish = max(
                max_finish,
                float(self.queue[pos]["finish_time"]),
            )
            self.prefix_max_finish[pos] = max_finish

    def gap_bounds(self, timestamp):
        timestamp = float(timestamp)
        prefix_len = bisect_right(self.starts, timestamp)

        if prefix_len > 0:
            if self.prefix_max_finish[prefix_len - 1] > timestamp:
                return None
            idle_since = float(
                self.queue[prefix_len - 1]["finish_time"]
            )
        else:
            idle_since = -math.inf

        if prefix_len < len(self.queue):
            free_until = self.starts[prefix_len]
        else:
            free_until = math.inf

        return idle_since, free_until


def _iter_gap_candidate_times(
    timeline,
    min_start_time,
    max_start_time=None,
):
    min_start_time = float(min_start_time)
    index = bisect_left(timeline, min_start_time - EPS)
    emitted_minimum = False

    if max_start_time is None:
        end_index = len(timeline)
    else:
        max_start_time = float(max_start_time)
        end_index = bisect_left(
            timeline,
            max_start_time - EPS,
            lo=index,
        )

    if (
        max_start_time is None
        or min_start_time + EPS < max_start_time
    ):
        while index < end_index:
            timestamp = float(timeline[index])
            index += 1

            if not emitted_minimum and min_start_time < timestamp:
                yield min_start_time
                emitted_minimum = True
            if timestamp == min_start_time:
                emitted_minimum = True
            yield timestamp

        if not emitted_minimum:
            yield min_start_time


class EASYPSAS(FCFSPSAS):
    def schedule(self):
        super().prep_schedule()
        now = float(self.current_time)

        started_now = self._current_fcfs_commit()

        remaining0 = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        if not remaining0:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        head_fcfs_entry = self._future_fcfs_head(
            remaining0,
            barrier=now,
        )
        if head_fcfs_entry is None:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        head_job, head_nodes, head_start, head_finish = head_fcfs_entry

        self._current_easy_commit(
            started_now=started_now,
            head_job_id=head_job["job_id"],
            head_start_time=float(head_start),
            head_reserved_ids=head_nodes,
        )

        remaining = [j for j in self.waiting_queue if j["job_id"] not in started_now]
        if not remaining:
            self.selected_list = []
            if self.timeout is not None:
                super().timeout_policy()
            super().build_callbacks()
            return self.events

        fcfs_seed_plan = self._future_fcfs_seed_head(
            head_job=head_job,
            head_nodes=head_nodes,
            head_start=float(head_start),
            jobs_after=[j for j in remaining if j["job_id"] != head_job["job_id"]],
        )

        easy_future_plan = self._future_easy_backfill_from_fcfs(fcfs_seed_plan)

        self.selected_list = list(easy_future_plan)

        self._emit_wake_triggers_from_plan(self.selected_list)

        if self.timeout is not None:
            super().timeout_policy()
        super().build_callbacks()
        return self.events

    def _future_fcfs_head(self, jobs, barrier):
        """Return only the first valid FCFS-plan entry."""
        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )
        node_selection_static = self._build_node_selection_static_data(
            candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in candidates
            if nid in node_selection_static
        }
        candidates = [
            nid for nid in candidates if nid in release_times
        ]

        for job in jobs:
            req = int(job["res"])
            if req <= 0:
                continue

            result = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=candidates,
                release_times=release_times,
                min_start_time=float(barrier),
                node_static_data=node_selection_static,
            )
            if result is None:
                return None

            nodes, start_time = result
            start_time = float(start_time)
            compute_speed = min(
                float(self.state[nid]["compute_speed"])
                for nid in nodes
            )
            finish_time = start_time + (
                float(job["reqtime"]) / compute_speed
            )
            return job, nodes, start_time, float(finish_time)

        return None

    def _current_easy_commit(
        self,
        started_now,
        head_job_id,
        head_start_time,
        head_reserved_ids,
    ):
        now = float(self.current_time)
        seen_head = False
        head_reserved_set = set(head_reserved_ids)

        idle_all = list(self.idle)
        idle_non_reserved = [
            nid for nid in idle_all if nid not in head_reserved_set
        ]
        node_selection_static = self._build_node_selection_static_data(
            idle_all,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in idle_all
            if nid in node_selection_static
        }
        idle_all = [nid for nid in idle_all if nid in release_times]
        idle_non_reserved = [
            nid for nid in idle_non_reserved if nid in release_times
        ]

        def _remove_selected(selected_nodes):
            nonlocal idle_all, idle_non_reserved
            selected_set = set(selected_nodes)
            idle_all = [
                nid for nid in idle_all if nid not in selected_set
            ]
            idle_non_reserved = [
                nid
                for nid in idle_non_reserved
                if nid not in selected_set
            ]

        for job in self.waiting_queue:
            jid = job["job_id"]

            if jid == head_job_id:
                seen_head = True
                continue
            if not seen_head or jid in started_now:
                continue

            req = int(job["res"])
            if req <= 0:
                continue

            if len(idle_non_reserved) >= req:
                result = self._select_nodes_energy_aware_prepared(
                    required_nodes=req,
                    candidates=idle_non_reserved,
                    release_times=release_times,
                    min_start_time=now,
                    node_static_data=node_selection_static,
                )
                if result is not None:
                    nodes, start_time = result
                    if float(start_time) <= now + EPS:
                        super().allocate(job, nodes)
                        started_now.add(jid)
                        _remove_selected(nodes)
                        continue

            if len(idle_all) < req:
                continue

            result = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=idle_all,
                release_times=release_times,
                min_start_time=now,
                node_static_data=node_selection_static,
            )
            if result is None:
                continue

            nodes, start_time = result
            if float(start_time) > now + EPS:
                continue

            if any(nid in head_reserved_set for nid in nodes):
                compute_speed = min(
                    float(self.state[nid]["compute_speed"])
                    for nid in nodes
                )
                finish_time = now + (
                    float(job["reqtime"]) / compute_speed
                )
                if finish_time > float(head_start_time) + EPS:
                    continue

            super().allocate(job, nodes)
            started_now.add(jid)
            _remove_selected(nodes)

    def _future_fcfs_seed_head(
        self,
        head_job,
        head_nodes,
        head_start,
        jobs_after,
    ):
        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        node_selection_static = self._build_node_selection_static_data(
            candidates,
            self.next_releases,
        )
        release_times = {
            nid: float(self.next_releases[nid]["release_time"])
            for nid in candidates
            if nid in node_selection_static
        }
        candidates = [
            nid for nid in candidates if nid in release_times
        ]
        state = self.state

        def _append_planned_compute(job, selected_nids, start_time):
            compute_speed = min(
                float(state[nid]["compute_speed"])
                for nid in selected_nids
            )
            finish_time = float(start_time) + (
                float(job["reqtime"]) / compute_speed
            )
            for nid in selected_nids:
                release_times[nid] = float(finish_time)
            return float(finish_time)

        head_start = float(head_start)
        head_finish = _append_planned_compute(
            head_job,
            head_nodes,
            head_start,
        )
        plan = [
            (
                head_job,
                head_nodes,
                head_start,
                float(head_finish),
            )
        ]

        barrier = head_start
        for job in jobs_after:
            req = int(job["res"])
            if req <= 0:
                continue

            result = self._select_nodes_energy_aware_prepared(
                required_nodes=req,
                candidates=candidates,
                release_times=release_times,
                min_start_time=barrier,
                node_static_data=node_selection_static,
            )
            if result is None:
                break

            nodes, start_time = result
            start_time = float(start_time)
            finish_time = _append_planned_compute(
                job,
                nodes,
                start_time,
            )
            plan.append((
                job,
                nodes,
                start_time,
                float(finish_time),
            ))
            barrier = start_time

        return plan

    def _make_scheduled_by_id(self):
        return {
            nid: {
                "queue": list(entry["queue"]),
                "release_time": float(entry["release_time"]),
            }
            for nid, entry in self.next_releases.items()
        }

    @staticmethod
    def _insert_segment(q, seg):
        st = float(seg["start_time"])
        i = 0
        while i < len(q) and float(q[i]["start_time"]) <= st:
            i += 1
        q.insert(i, seg)

    def _append_compute_fixed(
        self,
        job,
        nodes,
        st,
        scheduled_by_id,
        gap_indexes=None,
        finish_timeline=None,
        finish_time_set=None,
    ):
        sp = min(float(self.state[n]["compute_speed"]) for n in nodes)
        wall = float(job["reqtime"]) / sp
        ft = float(st) + wall
        phase = f'compute(job={job["job_id"]})'
        seg = {
            "phase": phase,
            "start_time": float(st),
            "finish_time": float(ft),
        }
        for n in nodes:
            e = scheduled_by_id[n]
            e["release_time"] = max(
                float(e["release_time"]),
                float(ft),
            )
            if gap_indexes is None:
                self._insert_segment(e["queue"], dict(seg))
            else:
                gap_indexes[n].insert(seg)

        if (
            finish_timeline is not None
            and finish_time_set is not None
            and ft not in finish_time_set
        ):
            insort(finish_timeline, float(ft))
            finish_time_set.add(float(ft))

        return float(ft)

    def _window_feasible(
        self,
        job,
        nodes,
        st,
        scheduled_by_id,
        gap_indexes=None,
    ):
        st = float(st)
        compute_speed = min(
            float(self.state[nid]["compute_speed"])
            for nid in nodes
        )
        if compute_speed <= 0:
            return False

        finish_time = st + (
            float(job["reqtime"]) / compute_speed
        )

        if gap_indexes is not None:
            for nid in nodes:
                bounds = gap_indexes[nid].gap_bounds(st)
                if bounds is None:
                    return False
                _idle_since, free_until = bounds
                if finish_time > float(free_until) + EPS:
                    return False
            return True

        for nid in nodes:
            queue = sorted(
                scheduled_by_id[nid]["queue"],
                key=lambda seg: float(seg["start_time"]),
            )
            for seg in queue:
                seg_start = float(seg["start_time"])
                if finish_time <= seg_start + EPS:
                    break

                seg_finish = float(seg["finish_time"])
                if st >= seg_finish - EPS:
                    continue

                return False

        return True

    def _future_easy_backfill_from_fcfs(self, fcfs_plan):
        now = float(self.current_time)
        if not fcfs_plan:
            return []

        scheduled_by_id = self._make_scheduled_by_id()

        candidates = (
            list(self.idle)
            + list(self.sleeping)
            + list(self.computing)
            + list(self.switching_on)
            + list(self.switching_off)
        )

        gap_node_selection_static = (
            self._build_gap_node_selection_static_data(
                candidates,
                scheduled_by_id,
            )
        )
        gap_indexes = {
            nid: _GapQueueIndex(scheduled_by_id[nid]["queue"])
            for nid in candidates
            if nid in scheduled_by_id
        }
        available_nodes = [
            nid
            for nid in candidates
            if nid in gap_node_selection_static and nid in gap_indexes
        ]
        finish_time_set = {
            float(seg["finish_time"])
            for index in gap_indexes.values()
            for seg in index.queue
            if not math.isinf(float(seg["finish_time"]))
        }
        finish_timeline = sorted(finish_time_set)

        head_job, head_nodes, head_st, head_ft = fcfs_plan[0]
        head_st = float(head_st)

        if not self._window_feasible(
            head_job,
            head_nodes,
            head_st,
            scheduled_by_id,
            gap_indexes=gap_indexes,
        ):
            return []

        head_ft = self._append_compute_fixed(
            head_job,
            head_nodes,
            head_st,
            scheduled_by_id,
            gap_indexes=gap_indexes,
            finish_timeline=finish_timeline,
            finish_time_set=finish_time_set,
        )
        out = [(head_job, head_nodes, float(head_st), float(head_ft))]

        for job, fcfs_nodes, fcfs_st, fcfs_ft in fcfs_plan[1:]:
            req = int(job["res"])
            if req <= 0:
                continue

            fcfs_st = float(fcfs_st)

            if now + EPS < fcfs_st:
                res_early = self._future_select_nodes_energy_aware_gap(
                    required_nodes=req,
                    _candidates=candidates,
                    scheduled_by_id=scheduled_by_id,
                    min_start_time=now,
                    max_start_time=fcfs_st,
                    job=job,
                    node_static_data=gap_node_selection_static,
                    gap_indexes=gap_indexes,
                    finish_timeline=finish_timeline,
                    available_nodes=available_nodes,
                )
            else:
                res_early = None
            if res_early is not None:
                nodes_early, st_early = res_early
                st_early = float(st_early)
                if st_early + EPS < fcfs_st:
                    ft_early = self._append_compute_fixed(
                        job,
                        nodes_early,
                        st_early,
                        scheduled_by_id,
                        gap_indexes=gap_indexes,
                        finish_timeline=finish_timeline,
                        finish_time_set=finish_time_set,
                    )
                    out.append((job, nodes_early, float(st_early), float(ft_early)))
                    continue

            if self._window_feasible(
                job,
                fcfs_nodes,
                fcfs_st,
                scheduled_by_id,
                gap_indexes=gap_indexes,
            ):
                ft_keep = self._append_compute_fixed(
                    job,
                    fcfs_nodes,
                    fcfs_st,
                    scheduled_by_id,
                    gap_indexes=gap_indexes,
                    finish_timeline=finish_timeline,
                    finish_time_set=finish_time_set,
                )
                out.append((job, fcfs_nodes, float(fcfs_st), float(ft_keep)))
                continue

            res_slide = self._future_select_nodes_energy_aware_gap(
                required_nodes=req,
                _candidates=candidates,
                scheduled_by_id=scheduled_by_id,
                min_start_time=fcfs_st,
                job=job,
                node_static_data=gap_node_selection_static,
                gap_indexes=gap_indexes,
                finish_timeline=finish_timeline,
                available_nodes=available_nodes,
            )
            if res_slide is None:
                break

            nodes_slide, st_slide = res_slide
            ft_slide = self._append_compute_fixed(
                job,
                nodes_slide,
                float(st_slide),
                scheduled_by_id,
                gap_indexes=gap_indexes,
                finish_timeline=finish_timeline,
                finish_time_set=finish_time_set,
            )
            out.append((job, nodes_slide, float(st_slide), float(ft_slide)))

        out.sort(key=lambda x: float(x[2]))
        return out

    def _build_gap_node_selection_static_data(
        self,
        candidates,
        scheduled_by_id,
    ):
        now = float(self.current_time)
        static_data = {}

        for nid in candidates:
            if nid not in scheduled_by_id:
                continue

            node = self.state[nid]
            node_name = self.machines.nodes[nid]["node_name"]
            machine = self.machines.node_specs[node_name]

            if node["state"] == "active" and node.get("job_id") is None:
                state_label = "idle"
                state_priority = 0
            elif node["state"] == "active" and node.get("job_id") is not None:
                state_label = "computing"
                state_priority = 1
            elif node["state"] == "switching_on":
                state_label = "switching_on"
                state_priority = 2
            else:
                state_label = node["state"]
                state_priority = 3

            idle_power = machine["states"]["active"]["power"]
            if idle_power == "from_dvfs":
                idle_power = self._resolve_dvfs_power(
                    machine, node, "idle"
                )

            base_energy_waste = 0.0
            for seg in scheduled_by_id[nid]["queue"]:
                if _COMPUTE_RE.fullmatch(str(seg["phase"])):
                    continue
                start = float(seg["start_time"])
                finish = float(seg["finish_time"])
                if math.isinf(finish):
                    continue
                duration = finish - now if start < now else finish - start
                energy_rate = machine["states"][seg["phase"]]["power"]
                if energy_rate == "from_dvfs":
                    energy_rate = self._resolve_dvfs_power(
                        machine, node, "idle"
                    )
                base_energy_waste += float(energy_rate) * float(duration)

            static_data[nid] = {
                "speed": float(node.get("compute_speed", 1.0)),
                "idle_power": float(idle_power),
                "base": float(base_energy_waste),
                "state_label": state_label,
                "state_priority": int(state_priority),
                "timeout_priority": (
                    -self._remaining_idle_timeout(nid)
                    if state_priority == 0
                    else 0.0
                ),
            }

        return static_data

    def _future_select_nodes_energy_aware_gap(
        self,
        required_nodes,
        _candidates,
        scheduled_by_id,
        min_start_time,
        job,
        max_start_time=None,
        node_static_data=None,
        gap_indexes=None,
        finish_timeline=None,
        available_nodes=None,
    ):
        now = float(self.current_time)
        min_start_time = float(min_start_time)
        if max_start_time is not None:
            max_start_time = float(max_start_time)
            if min_start_time + EPS >= max_start_time:
                return None
        reqtime = float(job["reqtime"])

        cand = [n for n in _candidates if n in scheduled_by_id]
        if len(cand) < required_nodes:
            return None

        if node_static_data is None:
            node_static_data = self._build_gap_node_selection_static_data(
                cand,
                scheduled_by_id,
            )

        if gap_indexes is None:
            gap_indexes = {
                nid: _GapQueueIndex(scheduled_by_id[nid]["queue"])
                for nid in cand
                if nid in node_static_data
            }

        if available_nodes is None:
            available = [
                nid
                for nid in cand
                if nid in node_static_data and nid in gap_indexes
            ]
        else:
            available = available_nodes

        if len(available) < required_nodes:
            return None

        if finish_timeline is None:
            finish_timeline = sorted({
                float(seg["finish_time"])
                for nid in available
                for seg in gap_indexes[nid].queue
                if not math.isinf(float(seg["finish_time"]))
            })

        candidate_times = _iter_gap_candidate_times(
            finish_timeline,
            min_start_time,
            max_start_time=max_start_time,
        )

        node_rows = []
        for nid in available:
            index = gap_indexes[nid]
            dat = node_static_data[nid]
            node_rows.append((
                nid,
                index.queue,
                index.starts,
                index.prefix_max_finish,
                dat,
            ))

        positions = [0] * len(node_rows)
        initialized = False
        block_size = 32

        while True:
            time_block = list(islice(candidate_times, block_size))
            if not time_block:
                break

            pools = [[] for _ in time_block]

            for row_index, (
                nid,
                queue,
                starts,
                prefix_max_finish,
                dat,
            ) in enumerate(node_rows):
                if initialized:
                    position = positions[row_index]
                else:
                    position = bisect_right(
                        starts,
                        float(time_block[0]),
                    )

                queue_len = len(queue)
                speed = float(dat["speed"])
                state_priority = int(dat["state_priority"])
                timeout_priority = float(dat["timeout_priority"])
                state_label = dat["state_label"]
                base_cost = float(dat["base"])
                idle_power = float(dat["idle_power"])

                for time_index, timestamp in enumerate(time_block):
                    t = float(timestamp)
                    while (
                        position < queue_len
                        and starts[position] <= t
                    ):
                        position += 1

                    if (
                        position > 0
                        and prefix_max_finish[position - 1] > t
                    ):
                        continue

                    if position > 0:
                        idle_since = float(
                            queue[position - 1]["finish_time"]
                        )
                    else:
                        idle_since = -math.inf

                    if position < queue_len:
                        free_until = starts[position]
                    else:
                        free_until = math.inf

                    free_len = float(free_until) - t
                    if free_len <= EPS:
                        continue

                    if state_label in (
                        "switching_off",
                        "sleeping",
                    ):
                        cost = base_cost
                    else:
                        idle_ref = max(idle_since, now)
                        cost = (
                            base_cost
                            + idle_power
                            * max(0.0, t - idle_ref)
                        )

                    pools[time_index].append((
                        float(cost),
                        state_priority,
                        timeout_priority,
                        int(nid),
                        speed,
                        float(free_len),
                    ))

                positions[row_index] = position

            initialized = True

            for timestamp, pool in zip(time_block, pools):
                if len(pool) < required_nodes:
                    continue

                pool.sort(
                    key=lambda item: (
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                    )
                )

                cur_pool = pool
                chosen = None
                for _ in range(8):
                    top = cur_pool[:required_nodes]
                    min_speed = min(item[4] for item in top)
                    if min_speed <= 0:
                        chosen = None
                        break

                    wall = reqtime / min_speed
                    filtered = [
                        item
                        for item in cur_pool
                        if item[5] + EPS >= wall
                    ]
                    if len(filtered) < required_nodes:
                        chosen = None
                        break

                    # cur_pool is rank-sorted; filtering preserves order.
                    top2 = filtered[:required_nodes]
                    if top2 == top:
                        chosen = top2
                        break

                    cur_pool = filtered

                if chosen is None:
                    continue

                return (
                    [item[3] for item in chosen],
                    float(timestamp),
                )

        return None
