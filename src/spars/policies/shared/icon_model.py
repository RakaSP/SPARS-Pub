from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing import Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from spars.policies.scheduler.snf.icon import SNFICON


_COMPUTE_RE = re.compile(r"^compute\(job=(\d+)\)$")
EPS = 1e-9


def normalise_pmf(
    pmf: Optional[Mapping[int, float]],
) -> Optional[Dict[int, float]]:
    if pmf is None:
        return None

    cleaned: Dict[int, float] = {}
    for raw_req, raw_probability in pmf.items():
        req = int(raw_req)
        probability = float(raw_probability)
        if req <= 0:
            raise ValueError("resource_pmf keys must be positive integers")
        if probability < 0.0 or not math.isfinite(probability):
            raise ValueError(
                "resource_pmf probabilities must be finite and non-negative"
            )
        if probability > 0.0:
            cleaned[req] = cleaned.get(req, 0.0) + probability

    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("resource_pmf must contain positive probability")
    return {req: probability / total for req, probability in cleaned.items()}


def nonnegative_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def unit_interval_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value


def optional_positive_float(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return positive_float(name, value)


def optional_nonnegative_float(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return nonnegative_float(name, value)


@dataclass
class NodePartition:
    """Node ids by power state, as one ICON plan sees them."""

    idle: List[int]
    sleeping: List[int]
    switching_on: List[int]
    switching_off: List[int]
    computing: List[int]


@dataclass
class ICONForecast:
    """What the ICON scheduler hands the ICON power manager with a decision.

    ``nodes`` is a copy of this plan's node partition and ``next_releases``
    its release plan. The power manager updates its copy of the partition as
    it switches nodes, so later steps of the same decision see the planned
    power state.
    """

    nodes: "NodePartition"
    next_releases: Mapping[int, dict]
    model: "ICONWorkloadModel"
    mode: str  # "markov_power", or "snf_fallback" for plain SNF planning
    markov_fallback_enabled: bool
    markovianity: Mapping[str, object]
    monitor_events: Mapping[str, object]
    idle_before_commit: List[int]
    started_now: Any
    allocated_now: Dict[int, int]
    remaining_jobs: List[dict]
    decision_event: bool
    plan: List[tuple]  # (job, nodes, start_time, finish_time) per queued job


def lognormal_mean_excess(log_mu: float, log_sigma: float, threshold: float) -> float:
    """E[T - threshold | T > threshold] for a log-normal T = exp(N(log_mu, log_sigma^2))."""
    threshold = max(0.0, float(threshold))
    sigma = max(EPS, float(log_sigma))
    if threshold <= EPS:
        return max(
            EPS,
            math.exp(min(700.0, log_mu + 0.5 * sigma * sigma)),
        )

    z = (math.log(threshold) - log_mu) / sigma
    log_denominator = normal_log_survival(z)
    log_numerator = normal_log_survival(z - sigma)
    log_conditional_mean = (
        log_mu
        + 0.5 * sigma * sigma
        + log_numerator
        - log_denominator
    )
    conditional_mean = math.exp(min(700.0, log_conditional_mean))
    return max(EPS, conditional_mean - threshold)


def normal_log_survival(z: float) -> float:
    """log P(Z > z) for a standard normal Z, with an asymptotic tail for large z."""
    z = float(z)
    if z < 8.0:
        survival = 0.5 * math.erfc(z / math.sqrt(2.0))
        return math.log(max(1e-300, survival))
    inv2 = 1.0 / (z * z)
    correction = max(EPS, 1.0 - inv2 + 3.0 * inv2 * inv2)
    return (
        -0.5 * z * z
        - math.log(z)
        - 0.5 * math.log(2.0 * math.pi)
        + math.log(correction)
    )


class ICONWorkloadModel:
    """Online model of the workload seen by the ICON scheduler.

    Estimates arrival and service rates, the requested-node distribution and
    job runtimes from the monitor's arrival and completion records, and tests
    whether recent inter-arrival and service times look Markovian. Running
    jobs and their release times are read from the owning scheduler.
    """

    def __init__(
        self,
        scheduler: "SNFICON",
        *,
        arrival_rate,
        service_rate,
        resource_pmf,
        initial_arrival_rate,
        arrival_ema_alpha,
        service_ema_alpha,
        resource_ema_alpha,
        runtime_log_sigma_floor,
        max_decision_horizon,
        markov_window_seconds,
        markov_min_samples,
        markov_max_samples,
        markov_cv_tolerance,
        markov_max_lag1_autocorrelation,
        markov_ks_multiplier,
        markov_require_service_samples,
        markov_ks_param,
    ):
        self.scheduler = scheduler

        self._configured_arrival_rate = optional_positive_float(
            "arrival_rate", arrival_rate
        )
        self._configured_service_rate = optional_positive_float(
            "service_rate", service_rate
        )
        self.initial_arrival_rate = positive_float(
            "initial_arrival_rate", initial_arrival_rate
        )
        self.arrival_ema_alpha = unit_interval_float(
            "arrival_ema_alpha", arrival_ema_alpha
        )
        self.service_ema_alpha = unit_interval_float(
            "service_ema_alpha", service_ema_alpha
        )
        self.resource_ema_alpha = unit_interval_float(
            "resource_ema_alpha", resource_ema_alpha
        )
        self.runtime_log_sigma_floor = nonnegative_float(
            "runtime_log_sigma_floor", runtime_log_sigma_floor
        )
        self.max_decision_horizon = positive_float(
            "max_decision_horizon", max_decision_horizon
        )

        self.markov_ks_param = markov_ks_param
        self.markov_window_seconds = positive_float(
            "markov_window_seconds", markov_window_seconds
        )
        self.markov_min_samples = int(markov_min_samples)
        self.markov_max_samples = int(markov_max_samples)
        if self.markov_min_samples < 3:
            raise ValueError("markov_min_samples must be at least 3")
        if self.markov_max_samples < self.markov_min_samples:
            raise ValueError(
                "markov_max_samples must be greater than or equal to "
                "markov_min_samples"
            )
        self.markov_cv_tolerance = nonnegative_float(
            "markov_cv_tolerance", markov_cv_tolerance
        )
        self.markov_max_lag1_autocorrelation = nonnegative_float(
            "markov_max_lag1_autocorrelation",
            markov_max_lag1_autocorrelation,
        )
        if self.markov_max_lag1_autocorrelation > 1.0:
            raise ValueError(
                "markov_max_lag1_autocorrelation must not exceed 1"
            )
        self.markov_ks_multiplier = positive_float(
            "markov_ks_multiplier", markov_ks_multiplier
        )
        self.markov_require_service_samples = bool(
            markov_require_service_samples
        )

        self._fixed_resource_pmf = normalise_pmf(resource_pmf)

        self.last_arrival_event_time: Optional[float] = None
        self._ema_interarrival = 1.0 / self.initial_arrival_rate
        self._ema_requested_service_time: Optional[float] = None
        self._ema_service_time: Optional[float] = None

        self._recent_interarrivals = deque(maxlen=self.markov_max_samples)
        self._recent_service_times = deque(maxlen=self.markov_max_samples)

        self.ema_log_runtime_ratio: Optional[float] = None
        self.ema_log_runtime_ratio_sq: Optional[float] = None
        self.runtime_ratio_samples = 0

        self._resource_weights: Dict[int, float] = {}

    def _machine_for(self, node_id: int):
        node_name = self.scheduler.machines.nodes[node_id]["node_name"]
        return self.scheduler.machines.node_specs[node_name]

    def observe_arrivals(self, arrival_records, now: float) -> None:
        # Update EMA statistics from every monitor arrival record
        if not arrival_records:
            return

        speeds = [
            float(node.get("compute_speed", 1.0))
            for node in self.scheduler.state.values()
            if float(node.get("compute_speed", 1.0)) > 0.0
        ]
        representative_speed = (
            sum(speeds) / len(speeds)
            if speeds
            else 1.0
        )

        records = sorted(
            arrival_records,
            key=lambda job: (
                float(job.get("subtime", now)),
                int(job.get("job_id", -1)),
            ),
        )

        for job in records:
            event_time = float(job.get("subtime", now))

            if (
                self.last_arrival_event_time is not None
                and event_time + EPS < self.last_arrival_event_time
            ):
                raise RuntimeError(
                    "Arrival log is not time ordered: "
                    f"{event_time} < {self.last_arrival_event_time}"
                )

            if self.last_arrival_event_time is not None:
                interarrival = max(
                    0.0,
                    event_time - self.last_arrival_event_time,
                )
                self._ema_interarrival = self._ema_update(
                    self._ema_interarrival,
                    interarrival,
                    self.arrival_ema_alpha,
                )
                # Associate an interarrival interval with its later arrival.
                # It is therefore retained while that arrival lies in the
                # current last-T observation window.
                self._recent_interarrivals.append(
                    (float(event_time), float(interarrival))
                )

            self.last_arrival_event_time = event_time

            requested_nodes = int(job["res"])
            if requested_nodes <= 0:
                raise ValueError("Requested node can't be <= 0")
            self._update_resource_ema(requested_nodes)

            reqtime = float(job.get("reqtime", 0.0))
            if reqtime > 0.0:
                requested_service = reqtime / representative_speed
                self._ema_requested_service_time = self._ema_update(
                    self._ema_requested_service_time,
                    requested_service,
                    self.service_ema_alpha,
                )

    def observe_completions(self, completion_records) -> None:
        # Learn actual service time and actual/requested runtime ratio.
        records = sorted(
            completion_records,
            key=lambda job: (
                float(job.get("finish_time"))
                if job.get("finish_time") is not None
                else math.inf,
                int(job.get("job_id", -1)),
            ),
        )
        for job in records:
            start_time = job.get("start_time")
            finish_time = job.get("finish_time")
            if start_time is None or finish_time is None:
                continue

            duration = float(finish_time) - float(start_time)
            if duration <= EPS or not math.isfinite(duration):
                continue

            self._ema_service_time = self._ema_update(
                self._ema_service_time,
                duration,
                self.service_ema_alpha,
            )
            # Associate service duration with completion time so the
            # classifier uses completions observed within the last T seconds.
            self._recent_service_times.append(
                (float(finish_time), float(duration))
            )

            requested_duration = self._requested_walltime(job)
            if requested_duration is None or requested_duration <= EPS:
                continue

            ratio = duration / requested_duration
            if ratio <= EPS or not math.isfinite(ratio):
                continue

            log_ratio = math.log(ratio)
            self.ema_log_runtime_ratio = self._ema_update_signed(
                self.ema_log_runtime_ratio,
                log_ratio,
                self.service_ema_alpha,
            )
            self.ema_log_runtime_ratio_sq = self._ema_update(
                self.ema_log_runtime_ratio_sq,
                log_ratio * log_ratio,
                self.service_ema_alpha,
            )
            self.runtime_ratio_samples += 1

    @staticmethod
    def _ema_update(
        previous: Optional[float],
        sample: float,
        smoothing: float,
    ) -> float:
        sample = max(0.0, float(sample))
        if previous is None:
            return sample
        return (
            float(smoothing) * sample
            + (1.0 - float(smoothing)) * float(previous)
        )

    @staticmethod
    def _ema_update_signed(
        previous: Optional[float],
        sample: float,
        smoothing: float,
    ) -> float:
        sample = float(sample)
        if previous is None:
            return sample
        return (
            float(smoothing) * sample
            + (1.0 - float(smoothing)) * float(previous)
        )

    def _update_resource_ema(self, requested_nodes: int) -> None:
        if not self._resource_weights:
            self._resource_weights[requested_nodes] = 1.0
            return

        keep = 1.0 - self.resource_ema_alpha
        for key in list(self._resource_weights):
            self._resource_weights[key] *= keep
            if self._resource_weights[key] <= EPS:
                del self._resource_weights[key]

        self._resource_weights[requested_nodes] = (
            self._resource_weights.get(requested_nodes, 0.0)
            + self.resource_ema_alpha
        )

    def _prune_markov_history(self, now: float) -> None:
        # Discard Markov-classification samples older than the last T seconds
        cutoff = float(now) - self.markov_window_seconds
        for history in (
            self._recent_interarrivals,
            self._recent_service_times,
        ):
            while history and float(history[0][0]) < cutoff:
                history.popleft()

    @staticmethod
    def _window_values(history) -> List[float]:
        # Return sample values from a timestamped ``(time, value)`` window
        return [float(value) for _, value in history]

    def recent_markovianity(self, now: float) -> Dict[str, object]:
        #Classify workload observations from the last T simulation seconds
        now = float(now)
        self._prune_markov_history(now)

        arrival = self._series_markovianity(
            self._window_values(self._recent_interarrivals),
            name="interarrival",
            required=True,
        )
        service = self._series_markovianity(
            self._window_values(self._recent_service_times),
            name="service",
            required=self.markov_require_service_samples,
        )

        well_markovian = bool(
            arrival["accepted"]
            and (
                service["accepted"]
                or not self.markov_require_service_samples
            )
        )
        reasons = []
        if not arrival["accepted"]:
            reasons.append(f"arrival:{arrival['reason']}")
        if self.markov_require_service_samples and not service["accepted"]:
            reasons.append(f"service:{service['reason']}")

        return {
            "well_markovian": well_markovian,
            "reason": "accepted" if well_markovian else ";".join(reasons),
            "window_seconds": self.markov_window_seconds,
            "window_start": now - self.markov_window_seconds,
            "window_end": now,
            "minimum_samples": self.markov_min_samples,
            "maximum_samples": self.markov_max_samples,
            "arrival": arrival,
            "service": service,
        }

    def _series_markovianity(
        self,
        samples: Sequence[float],
        *,
        name: str,
        required: bool,
    ) -> Dict[str, object]:
        values = [
            float(value)
            for value in samples
            if math.isfinite(float(value)) and float(value) >= 0.0
        ]
        sample_count = len(values)
        base = {
            "name": name,
            "required": bool(required),
            "sample_count": sample_count,
        }

        if not required and sample_count < self.markov_min_samples:
            return {
                **base,
                "accepted": True,
                "reason": "optional_insufficient_samples_in_time_window",
                "failed_checks": [],
            }

        if sample_count < self.markov_min_samples:
            return {
                **base,
                "accepted": False,
                "reason": "insufficient_samples_in_time_window",
                "failed_checks": [
                    "insufficient_samples_in_time_window",
                ],
            }

        mean = sum(values) / sample_count
        if mean <= EPS:
            return {
                **base,
                "accepted": False,
                "reason": "non_positive_mean",
                "failed_checks": ["non_positive_mean"],
                "mean": float(mean),
            }

        variance = sum((value - mean) ** 2 for value in values) / sample_count
        coefficient_of_variation = math.sqrt(max(0.0, variance)) / mean

        left = values[:-1]
        right = values[1:]
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (x - left_mean) * (y - right_mean)
            for x, y in zip(left, right)
        )
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        lag1_autocorrelation = (
            numerator / denominator if denominator > EPS else 1.0
        )

        sorted_values = sorted(values)
        ks_distance = 0.0
        for index, value in enumerate(sorted_values, start=1):
            fitted_cdf = 1.0 - math.exp(-value / mean)
            empirical_before = (index - 1) / sample_count
            empirical_after = index / sample_count
            ks_distance = max(
                ks_distance,
                abs(empirical_before - fitted_cdf),
                abs(empirical_after - fitted_cdf),
            )

        cv_ok = (
            abs(coefficient_of_variation - 1.0)
            <= self.markov_cv_tolerance
        )
        autocorrelation_ok = (
            abs(lag1_autocorrelation)
            <= self.markov_max_lag1_autocorrelation
        )
        ks_limit = (
            self.markov_ks_multiplier
            * self.markov_ks_param
            / math.sqrt(sample_count)
        )
        ks_ok = ks_distance <= ks_limit
        accepted = bool(cv_ok and autocorrelation_ok and ks_ok)

        failed_checks = []
        if not cv_ok:
            failed_checks.append("cv")
        if not autocorrelation_ok:
            failed_checks.append("lag1_autocorrelation")
        if not ks_ok:
            failed_checks.append("exponential_ks")

        return {
            **base,
            "accepted": accepted,
            "reason": "accepted" if accepted else ",".join(failed_checks),
            "failed_checks": list(failed_checks),
            "mean": float(mean),
            "coefficient_of_variation": float(coefficient_of_variation),
            "cv_range": [
                max(0.0, 1.0 - self.markov_cv_tolerance),
                1.0 + self.markov_cv_tolerance,
            ],
            "lag1_autocorrelation": float(lag1_autocorrelation),
            "max_abs_lag1_autocorrelation": (
                self.markov_max_lag1_autocorrelation
            ),
            "ks_distance": float(ks_distance),
            "ks_limit": float(ks_limit),
        }

    def arrival_rate(self) -> float:
        if self._configured_arrival_rate is not None:
            return self._configured_arrival_rate
        return 1.0 / max(EPS, self._ema_interarrival)

    def service_rate(self) -> float:
        if self._configured_service_rate is not None:
            return self._configured_service_rate

        mean_service = (
            self._ema_service_time
            if self._ema_service_time is not None
            else self._ema_requested_service_time
        )
        if mean_service is None:
            return 0.0

        return 1.0 / max(EPS, mean_service)

    def resource_pmf(self) -> Dict[int, float]:
        if self._fixed_resource_pmf is not None:
            return dict(self._fixed_resource_pmf)

        total = sum(self._resource_weights.values())
        if total <= EPS:
            return {1: 1.0}

        return {
            int(requested_nodes): float(weight) / total
            for requested_nodes, weight
            in sorted(self._resource_weights.items())
            if weight > EPS
        }

    def runtime_ratio_median(self) -> float:
        # Return exp(E[log(actual/requested)]), or 1.0 at cold start
        if self.ema_log_runtime_ratio is None:
            return 1.0
        return max(EPS, math.exp(min(700.0, self.ema_log_runtime_ratio)))

    def _runtime_ratio_distribution(self):
        if (
            self.ema_log_runtime_ratio is None
            or self.ema_log_runtime_ratio_sq is None
        ):
            return None
        mu = float(self.ema_log_runtime_ratio)
        variance = max(
            0.0,
            float(self.ema_log_runtime_ratio_sq) - mu * mu,
        )
        sigma = max(self.runtime_log_sigma_floor, math.sqrt(variance))
        return mu, sigma

    def predicted_job_duration(
        self,
        job: Mapping[str, object],
        *,
        compute_speed: Optional[float] = None,
    ) -> float:
        requested = self._requested_walltime(
            job,
            compute_speed=compute_speed,
        )
        if requested is None:
            if self._ema_service_time is not None:
                return max(EPS, float(self._ema_service_time))
            raise ValueError(
                f"Job {job.get('job_id')} has no valid requested duration"
            )
        return max(EPS, requested * self.runtime_ratio_median())

    def active_job_predictions(self) -> List[Dict[str, object]]:
        predictions = [
            self._predict_active_job_remaining(job)
            for job in self._scheduler_active_jobs().values()
        ]
        return sorted(predictions, key=lambda item: item["job_id"])

    def _scheduler_active_jobs(self) -> Dict[int, Dict[str, object]]:
        active: Dict[int, Dict[str, object]] = {}

        for job in getattr(self.scheduler.jobs_manager, "active_jobs", []):
            job_id = int(job["job_id"])
            active[job_id] = dict(job)
            active[job_id]["job_id"] = job_id
            active[job_id]["nodes"] = list(job.get("nodes", []))

        for job_id, job in self.scheduler._started_this_schedule.items():
            active[int(job_id)] = dict(job)
            active[int(job_id)]["nodes"] = list(job.get("nodes", []))

        now = float(self.scheduler.current_time)
        for node_id in self.scheduler.computing:
            job_id = self.scheduler.state[node_id].get("job_id")
            if job_id is None:
                entry = self.scheduler.next_releases.get(node_id, {})
                for segment in entry.get("queue", []):
                    candidate = self._phase_job_id(segment.get("phase"))
                    if candidate is not None:
                        job_id = candidate
                        break
            if job_id is None:
                continue

            job_id = int(job_id)
            record = active.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "start_time": now,
                    "nodes": [],
                },
            )
            record.setdefault("nodes", [])
            if node_id not in record["nodes"]:
                record["nodes"].append(node_id)

            entry = self.scheduler.next_releases.get(node_id, {})
            for segment in entry.get("queue", []):
                if self._phase_job_id(segment.get("phase")) != job_id:
                    continue
                segment_start = float(segment.get("start_time", now))
                record["start_time"] = min(
                    float(record.get("start_time", segment_start)),
                    segment_start,
                )
                if "reqtime" not in record:
                    segment_finish = float(
                        segment.get("finish_time", segment_start)
                    )
                    requested_duration = segment_finish - segment_start
                    if requested_duration > EPS:
                        record["requested_finish_time"] = segment_finish
                break

        for record in active.values():
            record["nodes"] = sorted(set(record.get("nodes", [])))
        return active

    def _predict_active_job_remaining(
        self,
        job: Mapping[str, object],
    ) -> Dict[str, object]:
        now = float(self.scheduler.current_time)
        start_time = float(job.get("start_time", now))
        elapsed = max(0.0, now - start_time)
        requested = self._requested_walltime(job)

        if requested is None:
            predicted_total = (
                float(self._ema_service_time)
                if self._ema_service_time is not None
                else self.max_decision_horizon
            )
            remaining = max(EPS, predicted_total - elapsed)
            source = "absolute_service_fallback"
        else:
            distribution = self._runtime_ratio_distribution()
            if distribution is None:
                predicted_total = requested
                remaining = max(EPS, predicted_total - elapsed)
                if elapsed >= predicted_total:
                    # No learned ratio exists yet.  Stay conservative, but keep
                    # the fallback proportional to this job's own request.
                    remaining = max(EPS, requested)
                source = "requested_duration_cold_start"
            else:
                ratio_mu, ratio_sigma = distribution
                log_total_mu = math.log(max(EPS, requested)) + ratio_mu
                predicted_total = requested * math.exp(
                    min(700.0, ratio_mu)
                )
                remaining = lognormal_mean_excess(
                    log_total_mu,
                    ratio_sigma,
                    elapsed,
                )
                source = "conditional_lognormal_runtime_ratio"

        return {
            "job_id": int(job["job_id"]),
            "nodes": list(job.get("nodes", [])),
            "requested_duration": requested,
            "elapsed": float(elapsed),
            "predicted_total_duration": float(predicted_total),
            "predicted_remaining": float(remaining),
            "prediction_source": source,
        }

    def _requested_walltime(
        self,
        job: Mapping[str, object],
        *,
        compute_speed: Optional[float] = None,
    ) -> Optional[float]:
        # Return this job's requested wall-clock duration.
        start_time = job.get("start_time")
        for key in ("req_finish_time", "requested_finish_time"):
            requested_finish = job.get(key)
            if start_time is not None and requested_finish is not None:
                duration = float(requested_finish) - float(start_time)
                if duration > EPS and math.isfinite(duration):
                    return duration

        reqtime = job.get("reqtime")
        if reqtime is None:
            return None
        reqtime = float(reqtime)
        if reqtime <= EPS or not math.isfinite(reqtime):
            return None

        if compute_speed is None:
            nodes = list(job.get("nodes", []))
            speeds = [
                self.active_compute_speed(int(node_id))
                for node_id in nodes
                if int(node_id) in self.scheduler.state
            ]
            if speeds:
                compute_speed = min(speeds)

        if compute_speed is None or compute_speed <= EPS:
            compute_speed = 1.0

        duration = reqtime / float(compute_speed)
        return duration if duration > EPS and math.isfinite(duration) else None

    @staticmethod
    def _phase_job_id(phase) -> Optional[int]:
        match = _COMPUTE_RE.fullmatch(str(phase))
        return int(match.group(1)) if match is not None else None

    def active_compute_speed(self, node_id: int) -> float:
        # Return a node's active-state compute speed.
        node = self.scheduler.state[node_id]
        current = float(node.get("compute_speed", 0.0))
        if node.get("state") == "active" and current > EPS:
            return current

        machine = self._machine_for(node_id)
        active = machine["states"]["active"]
        speed = active.get("compute_speed")
        if speed == "from_dvfs":
            speed = machine["dvfs_profiles"][node["dvfs_mode"]][
                "compute_speed"
            ]
        speed = float(speed)
        if speed <= EPS:
            raise RuntimeError(
                f"Active compute speed for node {node_id} must be > 0"
            )
        return speed

    def prediction_release_map(self):
        release_map = {
            node_id: {
                "queue": [
                    dict(segment) for segment in entry.get("queue", [])
                ],
                "release_time": float(
                    entry.get("release_time", self.scheduler.current_time)
                ),
            }
            for node_id, entry in self.scheduler.next_releases.items()
        }

        now = float(self.scheduler.current_time)
        for prediction in self.active_job_predictions():
            job_id = int(prediction["job_id"])
            predicted_release = now + float(
                prediction["predicted_remaining"]
            )
            for node_id in prediction["nodes"]:
                entry = release_map.get(node_id)
                if entry is None:
                    continue
                matched = False
                for segment in entry["queue"]:
                    if self._phase_job_id(segment.get("phase")) == job_id:
                        segment["finish_time"] = predicted_release
                        matched = True
                if not matched:
                    entry["queue"].append(
                        {
                            "phase": f"compute(job={job_id})",
                            "start_time": now - float(prediction["elapsed"]),
                            "finish_time": predicted_release,
                        }
                    )
                entry["release_time"] = predicted_release

        return release_map
