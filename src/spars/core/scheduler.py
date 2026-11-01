from typing import TYPE_CHECKING, Any

from spars.core.registry import build_component, load_policy
from spars.policies.discovery import canonical_name

if TYPE_CHECKING:
    from spars.core.simulator.machine import Machine
    from spars.core.simulator.monitor import Monitor
    from spars.core.simulator.platform_control import PlatformControl
    from spars.core.trigger import PlanContext
    import os
    from collections.abc import Mapping
    from spars.core.power_manager import PowerManager
    from spars.core.simulator.jobs_manager import JobsManager


class Scheduler:
    """Resolve and delegate to the configured scheduling algorithm."""

    def __init__(
        self,
        machines: "Machine",
        jobs_manager: "JobsManager",
        name: str,
        start_time: float,
        config: dict[str, Any],
        workload: "dict[str, Any]" = None,
        platform: "dict[int, dict[str, Any]]" = None,
        monitor: "Monitor" = None,
        platform_control: "PlatformControl" = None,
        output_dir: "str | os.PathLike[str] | None" = None,
    ) -> None:
        scheduler_name = canonical_name("scheduler", name)
        scheduler_class = load_policy("scheduler", scheduler_name)
        runtime_values = {
            "machines": machines,
            "jobs_manager": jobs_manager,
            "start_time": start_time,
            "workload": workload,
            "platform": platform,
            "monitor": monitor,
            "platform_control": platform_control,
            # The run's output folder, for policies that write their own files.
            "output_dir": output_dir,
        }

        self.algorithm = build_component(
            component_class=scheduler_class,
            runtime_values=runtime_values,
            config=config,
            component_name=scheduler_name,
        )

    def schedule(
        self,
        current_time: float,
        power_state: "PowerManager",
    ) -> Any:
        self.algorithm.set_time(current_time)
        set_power_state = getattr(self.algorithm, "set_power_state", None)
        if set_power_state is not None:
            set_power_state(power_state)
        return self.algorithm.schedule()

    def observe(
        self,
        current_time: float,
        power_state: "PowerManager",
    ) -> None:
        observe = getattr(self.algorithm, "observe", None)
        if observe is None:
            return
        self.algorithm.set_time(current_time)
        set_power_state = getattr(self.algorithm, "set_power_state", None)
        if set_power_state is not None:
            set_power_state(power_state)
        observe()

    def plan_deviation(
        self,
        current_time: float,
        ignored_nodes: "Mapping[int, str] | None" = None,
    ) -> bool:
        # Without a plan model, deviation cannot be judged: always replan.
        check = getattr(self.algorithm, "plan_deviation", None)
        if check is None:
            return True
        return bool(check(current_time, ignored_nodes))

    def needs_replan(self, ctx: "PlanContext") -> bool:
        rule = getattr(self.algorithm, "needs_replan", None)
        return bool(rule(ctx)) if rule is not None else False
