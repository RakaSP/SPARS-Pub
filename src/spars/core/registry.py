import importlib
from collections.abc import Mapping
from inspect import Parameter, signature
from typing import Any, TypeVar

from spars.policies.discovery import canonical_name, resolve


Component = TypeVar("Component")


def validate_component_config(
    config: dict[str, Any],
    component_name: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    """Validate and return a component's single config object."""
    if not isinstance(config, dict):
        raise TypeError(
            f"config for {component_name!r} must be a dictionary"
        )

    missing_keys = sorted(expected_keys - set(config))
    unexpected_keys = sorted(set(config) - expected_keys)
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys: {missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected keys: {unexpected_keys}")
        raise ValueError(
            f"Invalid config for {component_name!r}: "
            + "; ".join(details)
        )

    return config


def config_with_defaults(
    config: dict[str, Any],
    component_name: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """A component's config, with ``defaults`` for the keys it leaves out."""
    if not isinstance(config, dict):
        raise TypeError(
            f"config for {component_name!r} must be a dictionary"
        )

    unexpected_keys = sorted(set(config) - set(defaults))
    if unexpected_keys:
        raise ValueError(
            f"Invalid config for {component_name!r}: "
            f"unexpected keys: {unexpected_keys}"
        )

    return {**defaults, **config}


def resolve_run_components(
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the scheduler and power-management specs of a run config."""
    if "algorithm" in run_config or "algo_config" in run_config:
        raise ValueError(
            "run.algorithm and run.algo_config are no longer supported: give "
            'run.scheduler and run.power_management, each {"name": ..., "config": {...}}'
        )
    for component_name in ("scheduler", "power_management"):
        component = run_config.get(component_name)
        if not isinstance(component, dict):
            raise TypeError(f"run.{component_name} must be a dictionary")
        if "name" not in component or "config" not in component:
            raise KeyError(f"run.{component_name} requires name and config")
        if not isinstance(component["name"], str) or not component["name"]:
            raise ValueError(f"run.{component_name}.name must be a non-empty string")
        if not isinstance(component["config"], dict):
            raise TypeError(f"run.{component_name}.config must be a dictionary")

    scheduler = run_config["scheduler"]
    power_management = run_config["power_management"]
    return (
        {
            "name": canonical_name("scheduler", scheduler["name"]),
            "config": dict(scheduler["config"]),
        },
        {
            "name": canonical_name("power_management", power_management["name"]),
            "config": dict(power_management["config"]),
        },
    )


def get_run_timeout(run_config: dict[str, Any]) -> Any:
    """Return the effective timeout used for run labels and metadata."""
    _, power_management = resolve_run_components(run_config)
    if "timeout" in power_management["config"]:
        return power_management["config"]["timeout"]
    return None


def build_component(
    component_class: type[Component],
    runtime_values: dict[str, Any],
    config: dict[str, Any],
    component_name: str,
) -> Component:
    """Instantiate a component with runtime values and one config object."""
    constructor_parameters = signature(component_class.__init__).parameters
    parameter_names = set(constructor_parameters) - {"self"}

    for parameter_name, parameter in constructor_parameters.items():
        if parameter_name == "self":
            continue

        if parameter_name == "config":
            continue

        if parameter.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{component_class.__name__}.__init__ cannot use "
                "*args or **kwargs"
            )

        if (
            parameter_name not in runtime_values
            and parameter.default is Parameter.empty
        ):
            raise TypeError(
                f"Missing runtime value {parameter_name!r} for "
                f"{component_class.__name__}"
            )

    if "config" not in parameter_names:
        raise TypeError(
            f"{component_class.__name__}.__init__ must accept a config object"
        )

    if not isinstance(config, dict):
        raise TypeError(
            f"config for {component_name!r} must be a dictionary"
        )

    return component_class(
        **{
            name: value
            for name, value in runtime_values.items()
            if name in parameter_names and name != "config"
        },
        config=config,
    )


def load_object(import_path: str) -> Any:
    """Load ``pkg.mod:obj`` or ``pkg.mod.obj`` into a Python object."""
    if ":" in import_path:
        module_name, object_name = import_path.split(":", 1)
    else:
        module_name, _, object_name = import_path.rpartition(".")
        if not module_name:
            raise ValueError(
                f"Cannot import {import_path!r}. Use 'pkg.mod:obj' or 'pkg.mod.obj'."
            )

    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_policy(kind: str, name: str) -> Any:
    """Return the class for a class name or a "module:Class" path.

    ``kind`` is "scheduler" or "power_management"; see spars.policies.discovery.
    """
    return load_object(resolve(kind, name))
