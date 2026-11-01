"""Read, write and validate experiment definition files for the dashboard.

A definition is a normal module: an optional docstring, EXPERIMENT_SET_ORDER and
EXPERIMENTS as literal dicts. Only files made of exactly that are editable;
a file with helper code or computed values is left alone. Saving patches the
values in place, so comments survive; only when that is not possible is the file
regenerated and its comments lost (``write_definition`` then returns False).
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import pprint
import re
import tokenize
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spars.policies import discovery
from spars.policies.discovery import resolve

from .configuration import merge_overrides, set_nested, validate_config
from .defaults import BASE_CONFIG, PROJECT_ROOT
from .discovery import DEFINITIONS_ROOT, discover_research_names
from .source_patch import patch_definition
from .sweep import expand_sweep


@dataclass
class DefinitionFile:
    path: Path
    docstring: str | None = None
    order: int | None = None
    experiments: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""  # why the file is read-only; empty when editable
    has_comments: bool = False

    @property
    def editable(self) -> bool:
        return not self.reason

    @property
    def key(self) -> str:
        return f"{self.path.parent.name}/{self.path.stem}"


def read_definition(path: Path) -> DefinitionFile | None:
    """Parse a definition file without importing it; None if it has no EXPERIMENTS."""
    source = Path(path).read_text(encoding="utf-8")
    definition = DefinitionFile(path=Path(path))
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        definition.reason = f"syntax error: {error.msg}"
        return definition

    found = False
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(
            getattr(node, "value", None), ast.Constant
        ) and isinstance(node.value.value, str):
            definition.docstring = node.value.value
            continue
        target = (
            node.targets[0].id
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            else None
        )
        if target not in ("EXPERIMENTS", "EXPERIMENT_SET_ORDER"):
            definition.reason = definition.reason or "contains code besides plain data"
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            definition.reason = definition.reason or f"{target} is computed, not literal"
            continue
        if target == "EXPERIMENTS":
            found = True
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, dict) for item in value
            ):
                definition.reason = definition.reason or "EXPERIMENTS is not a list of dicts"
            else:
                definition.experiments = [dict(item) for item in value]
        else:
            definition.order = value

    if not found:
        # EXPERIMENTS may still exist but be computed (assigned in a way we skipped).
        if "EXPERIMENTS" not in source:
            return None
        definition.reason = definition.reason or "EXPERIMENTS is computed, not literal"
    definition.has_comments = any(
        token.type == tokenize.COMMENT
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
    )
    return definition


def render_definition(definition: DefinitionFile) -> str:
    parts = []
    if definition.docstring:
        parts.append(f'"""{definition.docstring}"""\n')
    if definition.order is not None:
        parts.append(f"EXPERIMENT_SET_ORDER = {definition.order}\n")
    body = pprint.pformat(definition.experiments, sort_dicts=False, width=88)
    parts.append(f"EXPERIMENTS = {body}\n")
    return "\n".join(parts)


def write_definition(definition: DefinitionFile, *, create: bool = False) -> bool:
    """Atomically write ``definition``; ``create`` refuses to replace a file.

    Returns False when the file had comments that could not be kept.
    """
    if not create and not definition.editable:
        raise ValueError(f"{definition.path.name} is read-only: {definition.reason}")
    if create and definition.path.exists():
        raise FileExistsError(f"{definition.path.name} already exists")
    kept_comments = True
    try:
        if create:
            raise ValueError("new file")
        text = patch_definition(
            definition.path.read_text(encoding="utf-8"),
            definition.experiments,
            definition.order,
            definition.docstring,
        )
    except (ValueError, SyntaxError):
        text = render_definition(definition)
        ast.literal_eval(text.split("EXPERIMENTS = ", 1)[1])  # must round-trip
        kept_comments = not (not create and definition.has_comments)
    temporary = definition.path.with_name(f".{definition.path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, definition.path)
    return kept_comments


def list_definition_files(research: str) -> list[DefinitionFile]:
    directory = DEFINITIONS_ROOT / research
    files = []
    for path in sorted(directory.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        definition = read_definition(path)
        if definition is not None:
            files.append(definition)
    return files


def _check_new_name(research: str, stem: str) -> Path:
    if not stem.isidentifier() or stem.startswith("_"):
        raise ValueError("File name must be a Python identifier (letters, digits, _)")
    if research not in discover_research_names():
        raise ValueError(f"Unknown set {research!r}")
    return DEFINITIONS_ROOT / research / f"{stem}.py"


def create_definition(research: str, stem: str, docstring: str = "") -> DefinitionFile:
    definition = DefinitionFile(
        path=_check_new_name(research, stem), docstring=docstring or None, order=1000
    )
    write_definition(definition, create=True)
    return definition


TRASH = PROJECT_ROOT / ".spars" / "trash"


def delete_definition(definition: DefinitionFile) -> Path:
    """Move the file to the trash folder (recoverable); returns where it went."""
    if not definition.editable:
        raise ValueError(f"{definition.path.name} is read-only: {definition.reason}")
    TRASH.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = TRASH / f"{stamp}-{definition.path.parent.name}-{definition.path.name}"
    os.replace(definition.path, target)
    return target


def create_research(name: str) -> str:
    """Make an (empty) set folder; experiments are added to it as files."""
    name = name.strip()
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("A set name must start with a letter and use only letters, digits and _")
    directory = DEFINITIONS_ROOT / name
    if directory.exists():
        raise FileExistsError(f"Set {name!r} already exists")
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(f'"""{name} experiments."""\n', encoding="utf-8")
    return name


def experiment_names(research: str) -> set[str]:
    """Every experiment name used in a set (names must be unique in a set)."""
    return {
        str(experiment.get("name"))
        for definition in list_definition_files(research)
        for experiment in definition.experiments
    }


def unused_name(base: str, taken: set[str]) -> str:
    name, number = base, 1
    while name in taken:
        number += 1
        name = f"{base}{number}"
    return name


def duplicate_experiment(source: DefinitionFile, name: str) -> DefinitionFile:
    """Copy a one-experiment file as a new file holding the same experiment under ``name``."""
    experiment = copy.deepcopy(source.experiments[0])
    experiment["name"] = name
    experiment["display_name"] = name
    copied = DefinitionFile(
        path=source.path.with_name(f"{file_stem(name)}.py"),
        docstring=source.docstring,
        order=source.order,
        experiments=[experiment],
    )
    write_definition(copied, create=True)
    return copied


def new_experiment(name: str, workload: str, platform: str) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": name,
        "enabled": True,
        "overrides": {"paths": {"workload": workload, "platform": platform}},
    }


# --- policy config defaults -------------------------------------------------


def _class_node(import_path: str) -> ast.ClassDef | None:
    """The class definition behind ``pkg.mod:Class``, found without importing anything.

    Importing would run the module's top-level code (torch, a user's own
    policy) inside the dashboard, so the source file is parsed instead.
    """
    from importlib.machinery import PathFinder

    module, _, name = import_path.replace(":", ".").rpartition(".")
    search, spec = None, None
    for part in module.split("."):
        spec = PathFinder.find_spec(part, search)  # locates the file, runs no code
        if spec is None:
            return None
        search = spec.submodule_search_locations
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    return next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None
    )


_PLAIN = (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List, ast.operator, ast.unaryop, ast.expr_context)


def _literal(node: ast.AST | None) -> Any:
    """A constant, a list or tuple of them, or plain arithmetic on them (``12.0 * 3600``).

    In a dict, values that are not plain (a call, a name) are left out.
    """
    if node is None:
        return None
    if isinstance(node, ast.Dict):
        keys = [_literal(k) for k in node.keys]
        values = [_literal(v) for v in node.values]
        return {
            k: v for k, v, n in zip(keys, values, node.values)
            if isinstance(k, str) and (v is not None or _is_none(n))
        }
    if all(isinstance(n, _PLAIN) for n in ast.walk(node)):
        try:
            return eval(compile(ast.Expression(node), "<defaults>", "eval"), {"__builtins__": {}})
        except (ArithmeticError, TypeError, ValueError):  # e.g. 1 / 0, "a" - 1
            return None
    return None


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def policy_defaults(kind: str, name: str) -> dict[str, Any]:
    """Config keys and defaults of a policy (its ``DEFAULTS``); {} when it declares none."""
    return declared_defaults(kind, name) or {}


def rl_objects(name: str) -> list[str]:
    """``module:name`` paths of the top-level functions or classes called ``name`` under policies/power_management/rl/.

    The RL form offers these for the gym parts (feature extractor ``feature_extraction``,
    translator ``action_translator``, learner ``learn``, reward ``Reward``).
    """
    root = Path(discovery.__file__).parent  # spars/policies
    found = []
    for path in sorted((root / "power_management" / "rl").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        if any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name for n in tree.body):
            module = ".".join(("spars", "policies", *path.relative_to(root).with_suffix("").parts))
            found.append(f"{module}:{name}")
    return found


def declared_defaults(kind: str, name: str) -> dict[str, Any] | None:
    """The policy class's ``DEFAULTS`` dict, read from its source; None when it has none."""
    try:
        path = resolve(kind, name)
    except ValueError:
        path = None
    node = _class_node(path) if path else None
    for statement in node.body if node else ():
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "DEFAULTS"
        ):
            defaults = _literal(statement.value)
            return copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
    return None


def model_defaults(class_path: str) -> dict[str, Any]:
    """Constructor parameters of an RL model class that have plain default values.

    ``device`` is left out: the runner sets it from rl.device. A class that
    inherits its ``__init__`` shows no parameters.
    """
    node = _class_node(class_path)
    init = next(
        (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    ) if node else None
    if init is None:
        return {}
    arguments = init.args
    positional = arguments.posonlyargs + arguments.args
    pairs = list(zip(positional[len(positional) - len(arguments.defaults):], arguments.defaults))
    pairs += [(a, d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is not None]
    defaults = {}
    for argument, default in pairs:
        value = _literal(default)
        if argument.arg != "device" and (value is None or isinstance(value, (bool, int, float, str))):
            defaults[argument.arg] = value
    return defaults


_SUMMARY_CACHE: dict[tuple[str, float], str] = {}
_SCAN_LIMIT = 8_000_000  # bytes; bigger workloads are not parsed in the UI thread


def resolve_data_file(value: str) -> Path | None:
    """The file an experiment path means, found the way the planner finds it (never from the current folder)."""
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, PROJECT_ROOT / "data" / path]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def workload_summary(workload: str, platform: str = "") -> str:
    """One line about a workload (jobs, size, span) and whether the platform is big enough."""
    path = resolve_data_file(workload)
    if path is None:
        return f"{workload or '(empty)'}: file not found"
    size = path.stat().st_size
    key = (str(path), path.stat().st_mtime)
    if key not in _SUMMARY_CACHE:
        if size > _SCAN_LIMIT:
            _SUMMARY_CACHE[key] = f"{path.name}: {size / 1e6:.0f} MB, too large to scan here"
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                jobs = data["jobs"]
                span = max((job["subtime"] for job in jobs), default=0) / 86400
                _SUMMARY_CACHE[key] = (
                    f"{path.name}: {len(jobs)} jobs, up to {data['nb_res']} nodes per job, "
                    f"submissions over {span:.1f} days"
                )
            except (OSError, ValueError, KeyError, TypeError):
                _SUMMARY_CACHE[key] = f"{path.name}: not a valid SPARS workload"
    text = _SUMMARY_CACHE[key]
    platform_file = resolve_data_file(platform)
    if platform_file is not None and "jobs" in text:
        try:
            nodes = platform_node_count(str(platform_file))
            needed = int(text.split("up to ")[1].split(" nodes")[0])
            text += f"; platform has {nodes} nodes" + (" (too small!)" if needed > nodes else " (ok)")
        except (OSError, ValueError, KeyError, IndexError):
            pass
    return text


# --- platform / validation --------------------------------------------------


def platform_node_count(platform_file: str) -> int:
    from spars.core.simulator.machine import Machine

    spec = json.loads(Path(platform_file).read_text(encoding="utf-8"))
    # Machine.id also accepts a bare int or a bare "start-end" string (not only a
    # list), so this must expand ids the same way the simulator does.
    return sum(len(Machine._expand_node_ids(machine["id"])) for machine in spec["machines"])


_OUTPUT_PLACEHOLDER = "results/_validation"
_NAME = re.compile(r"[\w.\-+ ()]+")


def name_problems(name: Any, other_names: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Problems with an experiment name; they stop a save (the name labels files and folders)."""
    if not isinstance(name, str) or not name.strip():
        return ["Name must not be empty"]
    if name.strip() in other_names:
        return [f"Name {name.strip()!r} is already used in this set"]
    if not _NAME.fullmatch(name.strip()):
        return ["Name may only contain letters, digits and _ . - + ( ) space"]
    return []


def file_stem(name: str) -> str:
    """A definition file name for an experiment name: FCFS-1800 -> FCFS_1800."""
    stem = re.sub(r"\W+", "_", name.strip()).strip("_") or "experiment"
    return stem if not stem[0].isdigit() else f"e_{stem}"


def _is_baseline(kind: str, name: str) -> bool:
    """A scheduler built on BaselineScheduler (it never switches nodes on)."""
    try:
        node = _class_node(resolve(kind, name))
    except ValueError:
        return False
    bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases] if node else []
    return any("Baseline" in base for base in bases)


def quick_problems(experiment: dict[str, Any]) -> list[str]:
    """Problems found without reading the workload: missing files, unknown policies.

    Cheap enough for every row of a list.
    """
    problems = []
    try:
        cfg = merge_overrides(BASE_CONFIG, experiment.get("overrides", {}))
    except Exception as error:  # a hand-edited file can hold anything
        return [f"{type(error).__name__}: {error}"]
    for key, label in (("workload", "Workload"), ("platform", "Platform")):
        value = cfg["paths"].get(key)
        if not value:
            problems.append(f"{label} is not chosen")
        elif resolve_data_file(str(value)) is None:
            problems.append(f"{label} file not found: {value}")
    run = cfg.get("run", {})
    for kind, label in (("scheduler", "Scheduler"), ("power_management", "Power manager")):
        name = (run.get(kind) or {}).get("name")
        if not name:
            problems.append(f"{label} is not chosen")
            continue
        try:
            resolve(kind, name)
        except ValueError as error:
            problems.append(str(error))
    if cfg.get("rl", {}).get("enabled"):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            problems.append("RL is on but PyTorch is not installed (python install.py --torch)")
    return problems


def validate_experiment(
    experiment: dict[str, Any], other_names: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Human-readable problems that would stop this experiment from being saved or run."""
    return name_problems(experiment.get("name"), other_names) + run_problems(experiment)


def blocked_definitions(definitions) -> dict[str, list[str]]:
    """Definition key -> why it cannot run now; the definitions that can run are absent.

    A file is left out as a whole, so a hand-written file with several
    experiments loses all of them when one has a problem.
    """
    blocked = {}
    for definition in definitions:
        enabled = [e for e in definition.experiments if e.get("enabled", True)]
        problems = [f"{definition.label}: no enabled experiment"] if not enabled else []
        for experiment in enabled:
            problems += [f"{experiment.get('name')}: {p}" for p in run_problems(experiment)]
        if problems:
            blocked[definition.key] = problems
    return blocked


def run_problems(experiment: dict[str, Any]) -> list[str]:
    """Problems that would make this experiment fail when it runs (its name is not checked).

    Every check runs, so one missing file does not hide the other problems.
    """
    from spars.core.experiments.planner import resolve_input_paths
    from spars.core.simulator.simulator import parse_start_time, validate_workload

    quick = quick_problems(experiment)
    errors = list(quick)
    try:
        cfg = merge_overrides(BASE_CONFIG, experiment.get("overrides", {}))
    except Exception:
        return errors
    def _check_inputs(checked: dict[str, Any]) -> None:
        checked["paths"]["output"] = _OUTPUT_PLACEHOLDER
        resolve_input_paths(checked)
        validate_config(checked)
        workload = json.loads(Path(checked["paths"]["workload"]).read_text(encoding="utf-8"))
        validate_workload(workload, platform_node_count(checked["paths"]["platform"]))

    checks = []
    if not quick:
        checks.append(lambda: _check_inputs(copy.deepcopy(cfg)))

        sweep = experiment.get("sweep") or {}
        if "paths.workload" in sweep or "paths.platform" in sweep:
            def swept_inputs():
                # A swept workload/platform is not the base one checked above, and a
                # bad value here would otherwise only fail once the batch is running.
                for sweep_values in expand_sweep(experiment.get("sweep")):
                    checked = copy.deepcopy(cfg)
                    for config_path, value in sweep_values.items():
                        set_nested(checked, config_path, value)
                    _check_inputs(checked)
            checks.append(swept_inputs)

    def policies():
        run = cfg["run"]
        for kind in ("scheduler", "power_management"):
            declared = declared_defaults(kind, run[kind]["name"])
            extra = set(run[kind]["config"]) - set(declared or {})
            if extra and declared is not None:  # only a policy with DEFAULTS lists its settings
                errors.append(f"{run[kind]['name']} has no setting {sorted(extra)}")
    checks += [
        policies,
        lambda: parse_start_time(cfg["run"]["start_time"]),
        lambda: list(expand_sweep(experiment.get("sweep"))),
    ]
    for check in checks:
        try:
            check()
        except Exception as error:  # every failure is reported to the user
            errors.append(f"{type(error).__name__}: {error}")
    return errors


def experiment_notes(experiment: dict[str, Any]) -> list[str]:
    """Settings that run but are probably not what was meant."""
    try:
        cfg = merge_overrides(BASE_CONFIG, experiment.get("overrides", {}))
        scheduler, power = cfg["run"]["scheduler"]["name"], cfg["run"]["power_management"]["name"]
    except Exception:
        return []
    notes = []
    if _is_baseline("scheduler", scheduler) and resolve_or_none("power_management", power) not in (
        None, resolve_or_none("power_management", "NoPowerManager")
    ):
        notes.append(
            f"{scheduler} never switches nodes on: with {power} the nodes it switches off stay off "
            "and the run stalls. Pair a Baseline scheduler with NoPowerManager."
        )
    return notes


def resolve_or_none(kind: str, name: str) -> str | None:
    try:
        return resolve(kind, name)
    except ValueError:
        return None


def policy_summary(kind: str, name: str) -> str:
    """First line of a policy class's docstring, read from its source (nothing is imported)."""
    path = resolve_or_none(kind, name)
    node = _class_node(path) if path else None
    doc = ast.get_docstring(node) if node else None
    return doc.strip().splitlines()[0] if doc else ""
