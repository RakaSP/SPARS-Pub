"""Find schedulers and power managers by class name; no registration needed.

A scheduler or power manager is any top-level class in a file under
policies/<kind>/ whose ``__init__`` takes ``config``, or that has no
``__init__`` and subclasses another class (names starting with ``_`` or
``Base`` are skipped). The files are parsed with ``ast``; nothing is
imported. A definition names a policy by its class name (any letter case) or
by a full ``module:Class`` path; a class name used in two files is ambiguous
and must be given as its path.
"""

from __future__ import annotations

import ast
from pathlib import Path

KINDS = ("scheduler", "power_management")
_ROOT = Path(__file__).resolve().parent
_PARSED: dict[tuple[Path, int], tuple[list[str], list[tuple[str, str]]]] = {}  # (file, mtime) -> (classes, [(class or "", why left out)])


def _takes_config(node: ast.ClassDef) -> bool:
    """A policy writes ``__init__(..., config)``, or subclasses one and inherits its constructor."""
    init = next(
        (item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
        None,
    )
    if init is None:
        return bool(node.bases)
    return any(a.arg == "config" for a in init.args.args + init.args.kwonlyargs)


def _classes(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """The policy classes of a file, and why a file or class that looks like one was left out."""
    key = (path, path.stat().st_mtime_ns)
    if key not in _PARSED:
        classes, left_out = [], []
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            line = f" line {error.lineno}" if getattr(error, "lineno", None) else ""
            left_out.append(("", f"{path.name} cannot be read: {type(error).__name__}{line}"))
            tree = ast.Module(body=[], type_ignores=[])
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith(("_", "Base")):
                continue
            if _takes_config(node):
                classes.append(node.name)
            else:
                left_out.append((node.name, f"{node.name} in {path.name} is not used: its __init__ has no `config` argument"))
        _PARSED[key] = (classes, left_out)
    return _PARSED[key]


def _files(kind: str):
    for path in sorted((_ROOT / kind).rglob("*.py")):
        if path.name != "__init__.py" and not path.name.startswith("_"):
            yield path


def discover(kind: str) -> dict[str, list[str]]:
    """Lower-case class name -> ``module:Class`` paths of the classes under policies/<kind>/."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    found: dict[str, list[str]] = {}
    for path in _files(kind):
        module = ".".join(("spars", "policies", *path.relative_to(_ROOT).with_suffix("").parts))
        for name in _classes(path)[0]:
            found.setdefault(name.lower(), []).append(f"{module}:{name}")
    return found


def _label(kind: str) -> str:
    return "power manager" if kind == "power_management" else kind


def resolve(kind: str, name: str) -> str:
    """The ``module:Class`` path behind a class name or a path."""
    if ":" in name:
        return name
    paths = discover(kind).get(name.lower(), [])
    if len(paths) == 1:
        return paths[0]
    if paths:
        raise ValueError(
            f"{_label(kind)} name {name!r} is ambiguous; use one of: {', '.join(paths)}"
        )
    known = sorted(p.split(":")[1] for ps in discover(kind).values() for p in ps)
    why = [reason for path in _files(kind) for cls, reason in _classes(path)[1] if cls == "" or cls.lower() == name.lower()]
    raise ValueError(
        f"Unknown {_label(kind)}: {name!r}. Available: {', '.join(known) or 'none'}"
        + (f". Left out: {'; '.join(why)}" if why else "")
    )


def canonical_name(kind: str, name: str) -> str:
    """The class name as written in its file (``fcfspsus`` -> ``FCFSPSUS``); a path, or an unknown name, as given."""
    if ":" in name:
        return name
    try:
        return resolve(kind, name).split(":")[1]
    except ValueError:
        return name


def options(kind: str) -> list[tuple[str, str, str]]:
    """``(value, title, path)`` for every choosable policy.

    ``value`` is what goes into the definition: the class name, or the full
    path when the class name is ambiguous.
    """
    rows = []
    for paths in discover(kind).values():
        for path in paths:
            title = path.split(":")[1]
            rows.append((title if len(paths) == 1 else path, title, path))
    return sorted(rows, key=lambda row: row[1].lower())
