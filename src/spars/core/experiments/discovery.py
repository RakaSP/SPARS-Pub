from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# Definitions live in spars/experiments, outside the core.
DEFINITIONS_ROOT = Path(__file__).resolve().parents[2] / "experiments"
DEFINITIONS_PACKAGE = "spars.experiments"
SELECTED_DEFINITIONS_ENV = "SPARS_SELECTED_DEFINITIONS"
SELECTED_RESEARCH_ENV = "SPARS_SELECTED_RESEARCH"

ExperimentConfig = dict[str, Any]


@dataclass(frozen=True)
class ExperimentDefinition:
    key: str
    name: str
    module_name: str
    experiments: tuple[ExperimentConfig, ...]
    order: int = 1000

    @property
    def label(self) -> str:
        """The experiment's name for a one-experiment file, else the file name and the count."""
        if len(self.experiments) == 1:
            experiment = self.experiments[0]
            return str(experiment.get("display_name") or experiment.get("name") or self.name)
        return f"{self.name.replace('_', ' ')}  ({len(self.experiments)} experiments)"


def discover_research_names() -> list[str]:
    """Return the set folders: every directory with a valid name, even an empty one (a new set starts empty)."""
    if not DEFINITIONS_ROOT.is_dir():
        return []

    research_names: list[str] = []
    for directory in sorted(DEFINITIONS_ROOT.iterdir(), key=lambda path: path.name):
        if (
            not directory.is_dir()
            or directory.name.startswith((".", "_"))
            or not directory.name.isidentifier()
        ):
            continue

        research_names.append(directory.name)

    return research_names


def _module_name_for_path(path: Path) -> tuple[str, str] | None:
    relative_path = path.relative_to(DEFINITIONS_ROOT).with_suffix("")
    parts = relative_path.parts

    # Only import normal Python packages/modules. This also ignores accidental
    # copied folders such as "icon copy".
    if not parts or any(not part.isidentifier() for part in parts):
        return None

    key = "/".join(parts)
    module_name = ".".join((DEFINITIONS_PACKAGE, *parts))
    return key, module_name


def discover_experiment_definitions(
    research: str | None = None,
    errors: list[str] | None = None,
) -> list[ExperimentDefinition]:
    """Discover selectable definitions, optionally within one set.

    With an ``errors`` list, a file that cannot be imported is skipped and its
    error appended (the dashboard still shows the other files); without, it raises.
    """
    if not DEFINITIONS_ROOT.is_dir():
        return []

    if research is None:
        paths = DEFINITIONS_ROOT.rglob("*.py")
    else:
        if research not in discover_research_names():
            available = ", ".join(discover_research_names()) or "none"
            raise ValueError(
                f"Unknown set {research!r}. Available: {available}"
            )
        paths = (DEFINITIONS_ROOT / research).glob("*.py")

    definitions: list[ExperimentDefinition] = []

    for path in sorted(paths, key=lambda item: str(item)):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue

        module_identity = _module_name_for_path(path)
        if module_identity is None:
            continue

        key, module_name = module_identity
        try:
            module = importlib.import_module(module_name)
            experiments = getattr(module, "EXPERIMENTS", None)
            if experiments is not None and not isinstance(experiments, (list, tuple)):
                raise TypeError(f"{module_name}.EXPERIMENTS must be a list or tuple")
            if experiments is not None and not all(isinstance(e, dict) for e in experiments):
                raise TypeError(f"{module_name}.EXPERIMENTS must contain dictionaries")
        except Exception as error:  # a hand-edited file can hold anything
            if errors is None:
                raise
            errors.append(f"{path.relative_to(DEFINITIONS_ROOT).as_posix()}: {type(error).__name__}: {error}")
            continue

        # Helper modules are allowed. Only modules that explicitly expose
        # EXPERIMENTS become selectable in the interactive runner.
        if experiments is None:
            continue

        definitions.append(
            ExperimentDefinition(
                key=key,
                name=path.stem,
                module_name=module_name,
                experiments=tuple(experiments),
                order=int(getattr(module, "EXPERIMENT_SET_ORDER", 1000)),
            )
        )

    # Preserve configured order. The key is a deterministic tie-breaker when
    # multiple files have the same order value.
    definitions.sort(key=lambda definition: (definition.order, definition.key))
    return definitions


def load_experiments(
    definition_keys: Iterable[str] | None = None,
) -> list[ExperimentConfig]:
    """Load copied experiment definitions, optionally restricted by key.

    With keys, only the sets they belong to are imported, so a broken file in
    another set does not stop this run.
    """
    if definition_keys is None:
        selected = discover_experiment_definitions()
    else:
        requested_keys = list(dict.fromkeys(definition_keys))
        sets = dict.fromkeys(key.split("/", 1)[0] for key in requested_keys)
        known = set(discover_research_names())
        available_by_key = {
            definition.key: definition
            for research in sets
            if research in known
            # errors=[]: a broken file elsewhere in the set must not stop a
            # requested, working experiment from loading (it was already
            # reported and left out before this run was started).
            for definition in discover_experiment_definitions(research, errors=[])
        }
        unknown_keys = [
            key for key in requested_keys if key not in available_by_key
        ]
        if unknown_keys:
            raise ValueError(
                "Unknown experiment definition(s): " + ", ".join(unknown_keys)
            )
        selected = [available_by_key[key] for key in requested_keys]

    experiments: list[ExperimentConfig] = []
    for definition in selected:
        experiments.extend(copy.deepcopy(definition.experiments))

    return experiments
