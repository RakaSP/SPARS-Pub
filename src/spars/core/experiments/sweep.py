import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def expand_sweep(
    sweep: Mapping[str, Sequence[Any]] | None,
) -> Iterator[dict[str, Any]]:
    """Yield every configuration combination in a parameter sweep."""
    if not sweep:
        yield {}
        return

    parameter_names = list(sweep)
    parameter_values: list[Sequence[Any]] = []

    for parameter_name in parameter_names:
        values = sweep[parameter_name]

        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"Sweep parameter {parameter_name} "
                "must contain a list or tuple"
            )

        if not values:
            raise ValueError(
                f"Sweep parameter {parameter_name} is empty"
            )

        parameter_values.append(values)

    for combination in itertools.product(*parameter_values):
        yield dict(zip(parameter_names, combination))
