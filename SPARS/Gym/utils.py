# SPARS/Gym/utils.py
# Minimal shim with ZERO defaults/registries.
# Your config/bootstrap must set these names BEFORE the env is imported,
# e.g., in config.py: from SPARS.Gym import utils as G; G.feature_extraction = ...

from typing import Any, Dict, Union, Tuple, Callable
import importlib
import inspect


def _unconfigured(name: str):
    raise RuntimeError(
        f"SPARS.Gym.utils.{name} is not configured. "
        "Import your config (which sets these symbols) BEFORE importing/creating the env."
    )


def feature_extraction(
    *args, **kwargs): return _unconfigured("feature_extraction")
def action_translator(
    *args, **kwargs): return _unconfigured("action_translator")
def learn(*args, **kwargs): return _unconfigured("learn")
def Reward(*args, **kwargs): return _unconfigured("Reward")


def _load_object(spec: str) -> Any:
    """Load 'pkg.mod:obj' or 'pkg.mod.obj' into a Python object."""
    if ":" in spec:
        module_path, obj_name = spec.split(":", 1)
    else:
        module_path, _, obj_name = spec.rpartition(".")
        if not module_path:
            raise ValueError(
                f"Cannot import '{spec}'. Use 'pkg.mod:obj' or 'pkg.mod.obj'.")
    mod = importlib.import_module(module_path)
    return getattr(mod, obj_name)


def make_reward(spec: Union[type, str, Dict[str, Any], Any], **overrides) -> Any:
    """
    Build a reward instance without any central registry.

    Accepts:
      - class (callable):        RewardClass
      - dotted string:          "pkg.mod:RewardClass" or "pkg.mod.RewardClass"
      - dict:                   {"name": <class|dotted_str|instance>, "params": {...}}
      - instance:               object having .calculate_reward(...)
    """
    params: Dict[str, Any] = {}

    if isinstance(spec, dict):
        name_or_obj = spec.get("name")
        params = dict(spec.get("params", {}))
        params.update(overrides)
        if inspect.isclass(name_or_obj):
            return name_or_obj(**params)
        if isinstance(name_or_obj, str):
            cls = _load_object(name_or_obj)
            return cls(**params)
        if hasattr(name_or_obj, "calculate_reward"):
            return name_or_obj
        raise TypeError(
            "reward dict 'name' must be a class, dotted string, or instance")

    if inspect.isclass(spec):
        params.update(overrides)
        return spec(**params)

    if isinstance(spec, str):
        cls = _load_object(spec)
        params.update(overrides)
        return cls(**params)

    if hasattr(spec, "calculate_reward"):
        return spec

    raise TypeError("Unsupported reward spec")


def resolve_components(cfg: Dict[str, Any]) -> Tuple[Callable, Callable, Any, Callable, Callable, Callable]:
    """
    Resolve components WITHOUT defaults/registries.
    All entries must be provided as callables or dotted strings.

    Required cfg keys:
      - feature_extractor
      - translator
      - reward   (class/instance/dotted or dict {'name':..., 'params':{...}})
      - learner

    Returns: (feature_extractor, translator, reward_instance, learner)
    """
    def _need(key: str):
        if key not in cfg:
            raise KeyError(
                f"Missing '{key}' in cfg for resolve_components(...)")
        return cfg[key]

    def _resolve_fn_or_path(x):
        if callable(x) and not isinstance(x, str):
            return x
        if isinstance(x, str):
            return _load_object(x)
        raise TypeError(
            f"Expected callable or dotted string for component, got {type(x)}")

    fe = _resolve_fn_or_path(_need("feature_extractor"))
    tr = _resolve_fn_or_path(_need("translator"))
    lr = _resolve_fn_or_path(_need("learner"))
    rw = make_reward(_need("reward"))

    return fe, tr, rw, lr


__all__ = [
    "Reward", "feature_extraction",
    "action_translator", "learn",
    "_load_object", "make_reward", "resolve_components",
]
