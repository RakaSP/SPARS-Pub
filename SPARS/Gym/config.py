import copy
import os

from SPARS.Gym import utils as G
from SPARS.Logger import log_info


FEATURE_EXTRACTORS = {
    "Budiarjo": "SPARS.Gym.features.Budiarjo:feature_extraction",
}

TRANSLATORS = {
    "Budiarjo": "SPARS.Gym.translators.Budiarjo:action_translator",
}

REWARDS = {
    "Budiarjo": "SPARS.Gym.rewards.Budiarjo:Reward",
}

LEARNERS = {
    "Budiarjo": "SPARS.Gym.learners.Budiarjo:learn",
}


CFG = {
    "feature_extractor": "Budiarjo",
    "translator": "Budiarjo",
    "reward": {
        "name": "Budiarjo",
        "params": {
            "weight1": 0.15,
            "weight2": 0.30,
            "device": "cuda",
        },
    },
    "learner": "Budiarjo",
}


def _resolve_from_map(mapping: dict, key_or_obj):
    if callable(key_or_obj) and not isinstance(key_or_obj, str):
        return key_or_obj

    if isinstance(key_or_obj, str):
        target = mapping.get(key_or_obj, key_or_obj)

        if isinstance(target, str):
            return G._load_object(target)

        return target

    raise TypeError(
        f"Expected callable or string key/path, "
        f"got {type(key_or_obj)}"
    )


def _resolve_reward(spec):
    if isinstance(spec, dict):
        name = spec["name"]
        params = copy.deepcopy(spec["params"])

        if isinstance(name, str):
            name = REWARDS.get(name, name)

        return G.make_reward({
            "name": name,
            "params": params,
        })

    if isinstance(spec, str):
        spec = REWARDS.get(spec, spec)

    return G.make_reward(spec)


def _reward_factory():
    return _resolve_reward(CFG["reward"])


feature_extractor = _resolve_from_map(
    FEATURE_EXTRACTORS,
    CFG["feature_extractor"],
)

translator = _resolve_from_map(
    TRANSLATORS,
    CFG["translator"],
)

learner = _resolve_from_map(
    LEARNERS,
    CFG["learner"],
)

reward_instance = _reward_factory()


def apply_feature_extractor_config(feature_extractor_cfg):
    global feature_extractor

    feature_extractor = _resolve_from_map(
        FEATURE_EXTRACTORS,
        feature_extractor_cfg,
    )

    CFG["feature_extractor"] = copy.deepcopy(
        feature_extractor_cfg
    )

    G.feature_extraction = feature_extractor

    return feature_extractor


def apply_translator_config(translator_cfg):
    global translator

    translator = _resolve_from_map(
        TRANSLATORS,
        translator_cfg,
    )

    CFG["translator"] = copy.deepcopy(translator_cfg)
    G.action_translator = translator

    return translator


def apply_reward_config(reward_cfg):
    global reward_instance

    if not isinstance(reward_cfg, dict):
        raise TypeError(
            "reward configuration must be a dictionary"
        )

    if "name" not in reward_cfg:
        raise KeyError(
            "Missing reward configuration field: name"
        )

    if "params" not in reward_cfg:
        raise KeyError(
            "Missing reward configuration field: params"
        )

    if not isinstance(reward_cfg["params"], dict):
        raise TypeError(
            "reward params must be a dictionary"
        )

    CFG["reward"] = copy.deepcopy(reward_cfg)

    reward_instance = _reward_factory()
    G.Reward = _reward_factory

    return reward_instance


def apply_learner_config(learner_cfg):
    global learner

    learner = _resolve_from_map(
        LEARNERS,
        learner_cfg,
    )

    CFG["learner"] = copy.deepcopy(learner_cfg)
    G.learn = learner

    return learner


def apply_gym_config(gym_cfg):
    if not isinstance(gym_cfg, dict):
        raise TypeError(
            "gym configuration must be a dictionary"
        )

    feature_extractor_cfg = gym_cfg["feature_extractor"]
    translator_cfg = gym_cfg["translator"]
    reward_cfg = gym_cfg["reward"]
    learner_cfg = gym_cfg["learner"]

    apply_feature_extractor_config(
        feature_extractor_cfg
    )

    apply_translator_config(
        translator_cfg
    )

    apply_reward_config(
        reward_cfg
    )

    apply_learner_config(
        learner_cfg
    )

    return CFG


G.feature_extraction = feature_extractor
G.action_translator = translator
G.Reward = _reward_factory
G.learn = learner

SELECTED = CFG


def _dotted(obj):
    module = (
        getattr(obj, "__module__", None)
        or type(obj).__module__
    )

    qualname = (
        getattr(obj, "__qualname__", None)
        or type(obj).__qualname__
    )

    return f"{module}:{qualname}"


def _format_reward(spec):
    if isinstance(spec, dict):
        name = spec["name"]
        params = spec["params"]
        mapped = REWARDS.get(name, name)

        return f"name={mapped}, params={params}"

    return str(REWARDS.get(spec, spec))


def _announce_selected():
    lines = [
        "SPARS.Gym config selected:",
        (
            "  feature_extractor = "
            f"{_dotted(feature_extractor)}"
        ),
        (
            "  translator        = "
            f"{_dotted(translator)}"
        ),
        (
            "  learner           = "
            f"{_dotted(learner)}"
        ),
        (
            "  reward            = "
            f"{_format_reward(CFG['reward'])}"
        ),
    ]

    message = "\n".join(lines)

    if os.getenv("SPARS_CONFIG_PRINT", "0") == "1":
        print(message)
    else:
        log_info(message)


if os.getenv("SPARS_CONFIG_SILENT", "0") != "1":
    _announce_selected()