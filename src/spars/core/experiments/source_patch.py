"""Change the values of a definition file in place, keeping its comments.

The definition is parsed with ``ast``, which knows where every value starts and
ends in the text. Only the values that differ from the new data are replaced,
added or removed; every other character of the file (comments, blank lines,
layout) stays as it was. Anything the patcher cannot place safely raises
``ValueError`` and the caller regenerates the file instead.
"""

from __future__ import annotations

import ast
import pprint
from typing import Any

WIDTH = 88


class _Text:
    def __init__(self, text: str) -> None:
        self.data = text.encode("utf-8")
        self.starts = [0]
        for line in self.data.split(b"\n")[:-1]:
            self.starts.append(self.starts[-1] + len(line) + 1)
        self.edits: list[tuple[int, int, bytes]] = []

    def start(self, node: ast.AST) -> int:
        return self.starts[node.lineno - 1] + node.col_offset

    def end(self, node: ast.AST) -> int:
        return self.starts[node.end_lineno - 1] + node.end_col_offset

    def column(self, offset: int) -> int:
        return offset - (self.data.rfind(b"\n", 0, offset) + 1)

    def replace(self, node: ast.AST, value: Any) -> None:
        start = self.start(node)
        self.edits.append((start, self.end(node), self.format(value, self.column(start))))

    def format(self, value: Any, column: int) -> bytes:
        text = pprint.pformat(value, sort_dicts=False, width=max(40, WIDTH - column))
        return text.replace("\n", "\n" + " " * column).encode("utf-8")

    def line_indent(self, node: ast.AST) -> bytes | None:
        """The leading whitespace of the node's line, if the node starts it."""
        start = self.start(node)
        line_start = start - self.column(start)
        before = self.data[line_start:start]
        return before if not before.strip() else None

    def delete_entry(self, first: ast.AST, last: ast.AST) -> None:
        start, end = self.start(first), self.end(last)
        indent = self.line_indent(first)
        if indent is not None:
            start -= len(indent)
        after = self.data[end:]
        rest = after.lstrip(b" \t")
        end += len(after) - len(rest)
        if rest.startswith(b","):
            end += 1
            rest = rest[1:]
        tail = rest.split(b"\n", 1)[0].strip()
        if indent is not None and (not tail or tail.startswith(b"#")):
            newline = self.data.find(b"\n", end)  # the whole line goes, comment included
            end = len(self.data) if newline < 0 else newline + 1
        elif rest[:1] == b" ":
            end += len(rest) - len(rest.lstrip(b" "))
        self.edits.append((start, end, b""))

    def append_entry(self, container: ast.AST, first: ast.AST, last: ast.AST, entry: bytes) -> bool:
        """Add an entry after the one spanning ``first``..``last``, on its own line.

        False if the layout is too tight.
        """
        indent = self.line_indent(first)
        if indent is None or container.end_lineno <= last.end_lineno:
            return False
        end = self.end(last)
        if not self.data[end:].lstrip(b" \t").startswith(b","):
            self.edits.append((end, end, b","))
        line_end = self.data.find(b"\n", end)
        entry = entry.replace(b"\n", b"\n" + indent)
        self.edits.append((line_end, line_end, b"\n" + indent + entry + b","))
        return True

    def apply(self) -> str:
        data = self.data
        for start, end, text in sorted(self.edits, reverse=True):
            data = data[:start] + text + data[end:]
        return data.decode("utf-8")


def _same(node: ast.AST, new: Any) -> bool:
    old = ast.literal_eval(node)
    return type(old) is type(new) and old == new


def _string_keys(node: ast.Dict) -> list[str] | None:
    keys = [k.value if isinstance(k, ast.Constant) and isinstance(k.value, str) else None for k in node.keys]
    return None if None in keys or len(set(keys)) != len(keys) else keys


def _diff(text: _Text, node: ast.AST, new: Any) -> None:
    if isinstance(node, ast.Dict) and isinstance(new, dict) and (keys := _string_keys(node)) is not None:
        _diff_dict(text, node, keys, new)
    elif isinstance(node, ast.List) and isinstance(new, list):
        _diff_list(text, node, new)
    elif not _same(node, new):
        text.replace(node, new)


def _diff_dict(text: _Text, node: ast.Dict, keys: list[str], new: dict) -> None:
    for key, key_node, value_node in zip(keys, node.keys, node.values):
        if key not in new:
            text.delete_entry(key_node, value_node)
        else:
            _diff(text, value_node, new[key])
    added = [k for k in new if k not in keys]
    kept = [i for i, k in enumerate(keys) if k in new]
    if not added:
        return
    entries = b",\n".join(f"{k!r}: ".encode() + text.format(new[k], 0) for k in added)
    if not kept or not text.append_entry(
        node, node.keys[kept[-1]], node.values[kept[-1]], entries
    ):
        # Cannot add without disturbing the layout: rewrite this dict, keep the rest.
        text.edits = [e for e in text.edits if not (text.start(node) <= e[0] < text.end(node))]
        text.replace(node, new)


def _names(items: list[Any]) -> list[str] | None:
    if not all(isinstance(i, dict) and isinstance(i.get("name"), str) for i in items):
        return None
    names = [i["name"] for i in items]
    return names if len(set(names)) == len(names) else None


def _diff_list(text: _Text, node: ast.List, new: list) -> None:
    old = [ast.literal_eval(e) for e in node.elts]
    old_names, new_names = _names(old), _names(new)
    if old_names is None or new_names is None:
        if len(old) == len(new):
            for element, value in zip(node.elts, new):
                _diff(text, element, value)
        elif not _same(node, new):
            text.replace(node, new)
        return
    pairs = {new_names.index(n): i for i, n in enumerate(old_names) if n in new_names}
    gone = [i for i in range(len(old)) if i not in pairs.values()]
    fresh = [j for j in range(len(new)) if j not in pairs]
    if len(gone) == len(fresh) == 1:  # one renamed experiment
        pairs[fresh.pop()] = gone.pop()
    for j, i in pairs.items():
        _diff(text, node.elts[i], new[j])
    for i in gone:
        text.delete_entry(node.elts[i], node.elts[i])
    if not fresh:
        return
    survivors = [node.elts[i] for i in sorted(pairs.values())]
    entries = b",\n".join(text.format(new[j], 0) for j in fresh)
    if not survivors or not text.append_entry(node, survivors[-1], survivors[-1], entries):
        text.edits = [e for e in text.edits if not (text.start(node) <= e[0] < text.end(node))]
        text.replace(node, new)


def patch_definition(source: str, experiments: list, order: int | None, docstring: str | None) -> str:
    """``source`` with EXPERIMENTS and EXPERIMENT_SET_ORDER set to the new values."""
    tree = ast.parse(source)
    text = _Text(source)
    targets = {"EXPERIMENTS": experiments, "EXPERIMENT_SET_ORDER": order}
    seen = set()
    for index, node in enumerate(tree.body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if node.value.value != docstring:
                raise ValueError("docstring changed")
            continue
        name = node.targets[0].id if isinstance(node, ast.Assign) else None
        if name not in targets or name in seen:
            raise ValueError(f"cannot place {name or type(node).__name__}")
        seen.add(name)
        _diff(text, node.value, targets[name])
    if "EXPERIMENTS" not in seen or (order is not None) != ("EXPERIMENT_SET_ORDER" in seen):
        raise ValueError("EXPERIMENTS or EXPERIMENT_SET_ORDER is missing or new")
    patched = text.apply()
    check = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(patched).body if isinstance(n, ast.Assign)}
    if check["EXPERIMENTS"] != experiments or check.get("EXPERIMENT_SET_ORDER") != order:
        raise ValueError("patched file does not hold the new data")
    return patched
