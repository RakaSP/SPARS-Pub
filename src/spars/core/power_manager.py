from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from spars.core.registry import build_component, load_policy
from spars.policies.discovery import canonical_name
from spars.core.decision import SchedulingDecision

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.platform_control import PlatformControl
    from spars.core.trigger import PlanContext
    import os
    from spars.core.simulator.jobs_manager import JobsManager


class PowerManager:
    """Resolve and delegate to the configured power-management policy."""

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        name: str,
        start_time: float,
        config: dict[str, Any],
        platform_control: "PlatformControl" = None,
        output_dir: "str | os.PathLike[str] | None" = None,
    ) -> None:
        power_manager_name = canonical_name("power_management", name)
        self.name = power_manager_name

        if not isinstance(config, dict):
            raise TypeError(
                f"config for {power_manager_name!r} must be a dictionary"
            )

        policy_class = load_policy("power_management", power_manager_name)
        runtime_values = {
            "machines": machines,
            "jobs_manager": jobs_manager,
            "start_time": start_time,
            "platform_control": platform_control,
            # The run's output folder, for policies that write their own files.
            "output_dir": output_dir,
        }

        self.policy = build_component(
            component_class=policy_class,
            runtime_values=runtime_values,
            config=config,
            component_name=power_manager_name,
        )

    def prepare(self, current_time: float) -> None:
        self.policy.prepare(current_time)

    def remaining_idle_timeout(
        self,
        node_id: int,
        current_time: float,
    ) -> float:
        return self.policy.remaining_idle_timeout(
            node_id=node_id,
            current_time=current_time,
        )

    def needs_replan(self, ctx: "PlanContext") -> bool:
        # A power manager without its own rule always needs a replan (safe).
        rule = getattr(self.policy, "needs_replan", None)
        return bool(rule(ctx)) if rule is not None else True

    def allow_wake(
        self,
        node_ids: Sequence[int],
        current_time: float,
    ) -> set[int]:
        """The sleeping nodes this policy lets the scheduler wake.

        A scheduler that wakes nodes on demand asks before it plans on a
        sleeping node. A policy may refuse (it defines ``allow_wake``); the
        scheduler then plans without that node. Without the method, every
        wake is allowed.
        """
        rule = getattr(self.policy, "allow_wake", None)
        if rule is None:
            return set(node_ids)
        return set(rule(node_ids, current_time))

    def idle_decision(
        self,
        node_id: int,
        idle_at: float,
        next_use_at: float,
        episode_id: int,
    ) -> Any:
        """How a node spends the gap from idle_at to next_use_at, and its energy.

        Only a policy that plans with the complete workload (the oracle)
        defines ``idle_decision``.
        """
        rule = getattr(self.policy, "idle_decision", None)
        if rule is None:
            raise RuntimeError(
                f"Power manager {self.name!r} does not provide idle_decision; "
                "pair SNFOraclePSAS with OraclePowerManager"
            )
        return rule(
            node_id=node_id,
            idle_at=idle_at,
            next_use_at=next_use_at,
            episode_id=episode_id,
        )

    def decide(
        self,
        current_time: float,
        scheduling_decision: SchedulingDecision,
    ) -> Any:
        return self.policy.decide(
            current_time=current_time,
            scheduling_decision=scheduling_decision,
        )

    def apply_rl_action(
        self,
        action: Any,
        current_time: float,
    ) -> Any:
        apply_action = getattr(self.policy, "apply_rl_action", None)
        if apply_action is None:
            raise RuntimeError(
                f"Power manager {self.name!r} does not accept RL actions"
            )
        return apply_action(action, current_time)

    def get_node_action_mask(
        self,
        current_time: float,
    ) -> dict[str, set[int]]:
        get_action_mask = getattr(
            self.policy,
            "get_node_action_mask",
            None,
        )
        if get_action_mask is None:
            raise RuntimeError(
                f"Power manager {self.name!r} does not provide "
                "an RL node-action mask"
            )
        return get_action_mask(current_time)

    def active_target_bounds(self, current_time: float) -> tuple[int, int]:
        bounds = getattr(self.policy, "active_target_bounds", None)
        if bounds is None:
            raise TypeError(f"Power controller {self.name!r} does not support capacity targets")
        return bounds(current_time)
