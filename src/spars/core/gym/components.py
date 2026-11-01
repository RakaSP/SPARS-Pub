import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from spars.core.registry import load_object

if TYPE_CHECKING:
    from spars.core.simulator.simulator import Simulator


Component = Callable[..., Any]
RewardSpec = type | str | Mapping[str, Any] | Any


def _unconfigured(component_name: str) -> NoReturn:
    raise RuntimeError(
        f"spars.core.gym.components.{component_name} is not configured. "
        "Import your config (which sets these symbols) BEFORE "
        "importing/creating the env."
    )


def feature_extraction(*args: Any, **kwargs: Any) -> NoReturn:
    return _unconfigured("feature_extraction")


def action_translator(*args: Any, **kwargs: Any) -> NoReturn:
    return _unconfigured("action_translator")


def learn(*args: Any, **kwargs: Any) -> NoReturn:
    return _unconfigured("learn")


def Reward(*args: Any, **kwargs: Any) -> NoReturn:
    return _unconfigured("Reward")


def reset_component_state(component: Component) -> None:
    """Reset optional episode-local state owned by a configured component."""
    reset_state = getattr(component, "reset_state", None)

    if reset_state is None:
        module_name = getattr(component, "__module__", None)
        if module_name:
            module = importlib.import_module(module_name)
            reset_state = getattr(module, "reset_state", None)

    if reset_state is None:
        return

    if not callable(reset_state):
        raise TypeError(f"{component!r} exposes a non-callable reset_state")

    reset_state()


def make_reward(
    spec: RewardSpec,
    **overrides: Any,
) -> Any:
    """Build a reward instance from a class, import path, config, or instance."""
    parameters: dict[str, Any] = {}

    if isinstance(spec, dict):
        reward_target = spec.get("name")
        parameters = dict(spec.get("params", {}))
        parameters.update(overrides)

        if inspect.isclass(reward_target):
            return reward_target(**parameters)

        if isinstance(reward_target, str):
            reward_class = load_object(reward_target)
            return reward_class(**parameters)

        if hasattr(reward_target, "calculate_reward"):
            return reward_target

        raise TypeError(
            "reward dict 'name' must be a class, dotted string, or instance"
        )

    if inspect.isclass(spec):
        parameters.update(overrides)
        return spec(**parameters)

    if isinstance(spec, str):
        reward_class = load_object(spec)
        parameters.update(overrides)
        return reward_class(**parameters)

    if hasattr(spec, "calculate_reward"):
        return spec

    raise TypeError("Unsupported reward spec")


def resolve_components(
    cfg: Mapping[str, Any],
) -> tuple[Component, Component, Any, Component]:
    """Resolve configured feature, translator, reward, and learner components."""

    def require_component(component_name: str) -> Any:
        if component_name not in cfg:
            raise KeyError(
                f"Missing {component_name!r} in cfg for resolve_components(...)"
            )
        return cfg[component_name]

    def resolve_callable(component: Any) -> Component:
        if callable(component) and not isinstance(component, str):
            return component

        if isinstance(component, str):
            return load_object(component)

        raise TypeError(
            f"Expected callable or dotted string for component, got {type(component)}"
        )

    feature_extractor = resolve_callable(require_component("feature_extractor"))
    translator = resolve_callable(require_component("translator"))
    learner = resolve_callable(require_component("learner"))
    reward = make_reward(require_component("reward"))

    return feature_extractor, translator, reward, learner


__all__ = [
    "Reward",
    "feature_extraction",
    "action_translator",
    "learn",
    "reset_component_state",
    "load_object",
    "make_reward",
    "resolve_components",
]


def validate_component_contracts(model: Any, simulator: "Simulator") -> None:
    """Validate configurable interfaces before starting a training episode."""

    def contract(component, key):
        module = importlib.import_module(component.__module__)
        return getattr(module, key)

    observation_kind = contract(feature_extraction, "OBSERVATION_KIND")
    action_kind = contract(action_translator, "ACTION_KIND")
    learner_kind = contract(learn, "ACTION_KIND")
    if observation_kind != model.observation_kind:
        raise ValueError(
            f"Feature/model observation mismatch: {observation_kind} / {model.observation_kind}"
        )
    if not action_kind == learner_kind == model.action_kind:
        raise ValueError("Translator, learner, and model action contracts do not match")
    feature_module = importlib.import_module(feature_extraction.__module__)
    schema = getattr(feature_module, "OBSERVATION_SCHEMA", None)
    if schema != getattr(model, "observation_schema", None):
        raise ValueError("Feature/model observation schema does not match")
    feature_width = getattr(feature_module, "OBSERVATION_DIM", None)
    if feature_width is not None and feature_width != getattr(model, "input_dim", None):
        raise ValueError("Feature/model observation width does not match")
    maximum = getattr(model, "max_nodes", None)
    if maximum is not None and maximum != len(simulator.platform_control.machine.nodes):
        raise ValueError("Policy max_nodes must equal the platform node count")
    manager = simulator.planner.power_manager.policy
    for method in contract(action_translator, "CONTROLLER_METHODS"):
        if not callable(getattr(manager, method, None)):
            raise TypeError(f"Selected power controller must implement {method}")
