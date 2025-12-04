import os
from SPARS.Gym import utils as G
from SPARS.Logger import log_info

FEATURE_EXTRACTORS = {
    "global_node_11d": "SPARS.Gym.features.global_node_11d:feature_extraction",
}

TRANSLATORS = {
    "scalar_active_target": "SPARS.Gym.translators.scalar_active_target:action_translator",
}

REWARDS = {
    "energy_wait_time": "SPARS.Gym.rewards.energy_wait_time:Reward",
}

LEARNERS = {
    "a2c": "SPARS.Gym.learners.a2c:learn",
}



CFG = {
    "feature_extractor": "global_node_11d",
    "translator": "scalar_active_target",
    "reward": {"name": "energy_wait_time", "params": {"alpha": 0.5, "beta": 0.5, "device": "cuda"}},
    "learner": "a2c",
}

def _resolve_from_map(mapping: dict, key_or_obj):
    if callable(key_or_obj) and not isinstance(key_or_obj, str):
        return key_or_obj
    if isinstance(key_or_obj, str):
        # allow dotted path directly
        target = mapping.get(key_or_obj, key_or_obj)
        return G._load_object(target) if isinstance(target, str) else target
    raise TypeError(
        f"Expected callable or string key/path, got {type(key_or_obj)}")


def _resolve_reward(spec):
    if isinstance(spec, dict):
        name = spec.get("name")
        params = dict(spec.get("params", {}))
        if isinstance(name, str):
            name = REWARDS.get(name, name)
        return G.make_reward({"name": name, "params": params})
    if isinstance(spec, str):
        spec = REWARDS.get(spec, spec)
    return G.make_reward(spec)


feature_extractor = _resolve_from_map(
    FEATURE_EXTRACTORS, CFG["feature_extractor"])
translator = _resolve_from_map(TRANSLATORS,       CFG["translator"])
learner = _resolve_from_map(LEARNERS,          CFG["learner"])
reward_instance = _resolve_reward(CFG["reward"])

G.feature_extraction = feature_extractor
G.action_translator = translator
G.learn = learner


def _reward_factory():
    return _resolve_reward(CFG["reward"])


G.Reward = _reward_factory

SELECTED = CFG 


def _dotted(obj):
    try:
        mod = getattr(obj, "__module__", None) or type(obj).__module__
        qn = getattr(obj, "__qualname__", None) or type(obj).__qualname__
        return f"{mod}:{qn}"
    except Exception:
        return str(obj)


def _format_reward(spec):
    if isinstance(spec, dict):
        name = spec.get("name")
        params = spec.get("params", {})
        mapped = REWARDS.get(name, name)
        return f"name={mapped}, params={params}"
    return str(REWARDS.get(spec, spec))


def _announce_selected():
    lines = [
        "SPARS.Gym config selected:",
        f"  feature_extractor = {_dotted(feature_extractor)}",
        f"  translator       = {_dotted(translator)}",
        f"  learner          = {_dotted(learner)}",
        f"  reward           = {_format_reward(CFG['reward'])}",
    ]
    msg = "\n".join(lines)
    if os.getenv("SPARS_CONFIG_PRINT", "0") == "1":
        print(msg)
    else:
        log_info(msg)

if os.getenv("SPARS_CONFIG_SILENT", "0") != "1":
    _announce_selected()
