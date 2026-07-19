"""Strict scheduler algorithm configuration handling."""


def require_algo_config(algorithm_name, algo_config, configurable_parameters):
    if not isinstance(algo_config, dict):
        raise TypeError(
            f"algo_config for {algorithm_name!r} must be a dictionary"
        )

    required = tuple(configurable_parameters)
    required_set = set(required)
    supplied_set = set(algo_config)
    missing = sorted(required_set - supplied_set)
    unexpected = sorted(supplied_set - required_set)

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise ValueError(
            f"Invalid algo_config for {algorithm_name!r}: "
            + "; ".join(details)
        )

    return {key: algo_config[key] for key in required}
