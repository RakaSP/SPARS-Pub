"""The input fields every dashboard screen uses, so they all behave the same.

Rules (the same on every screen):

- White text in a box is the value that will be used. Grey placeholder text
  only appears where an empty box is allowed, and says what empty means
  ("empty = all jobs"). A required field is marked ``*`` and starts empty
  when there is no sensible default.
- Entering a box, by keyboard or by a first mouse click, selects its text, so
  typing replaces it. A second click places the cursor.
- Up/Down and Tab/Shift+Tab move to the previous or next field. Left/Right
  change a Yes/No or option field.
- A pick-list field (``Picker``) opens its list when you type (filtered by the
  text) or press Enter (the whole list). While it is open, Up/Down move in the
  list, Enter takes the highlighted entry and Esc closes it. Tab never changes
  the value. Search matches the value and the description shown next to it.
  Any text is allowed, also text that is not in the list.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.input import Selection
from textual.widgets.option_list import Option

from spars.core.experiments.defaults import PROJECT_ROOT, RESULTS_ROOT

MAX_LISTED = 50
DATA_ROOT = PROJECT_ROOT / "data"


def filter_options(
    options: list[str], query: str, details: dict[str, tuple[str, str]] | None = None
) -> list[str]:
    """Options whose value or description contains ``query`` (any case), then those with its letters in order."""
    query = query.lower().strip()
    if not query:
        return list(options)
    details = details or {}

    def text(option: str) -> str:
        title, detail = details.get(option, ("", ""))
        return f"{option} {title} {detail}".lower()

    def subsequence(value: str) -> bool:
        letters = iter(value)
        return all(char in letters for char in query)

    contains = [o for o in options if query in text(o)]
    loose = [o for o in options if o not in contains and subsequence(o.lower())]
    return contains + loose


def data_files(subdirectory: str, patterns: tuple[str, ...] = ("*.json",)) -> list[str]:
    """Paths under data/<subdirectory> as experiments write them, e.g. workloads/json/x.json."""
    folder = DATA_ROOT / subdirectory
    if not folder.is_dir():
        return []
    return sorted(
        {path.relative_to(DATA_ROOT).as_posix() for pattern in patterns for path in folder.rglob(pattern)
         if not path.name.startswith(".")}
    )


_WORKLOAD_SNIFF: dict[tuple[str, int], bool] = {}


def _looks_like_workload(path) -> bool:
    """A SPARS workload JSON starts with its nb_res or jobs key (summaries and parameter files do not)."""
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _WORKLOAD_SNIFF:
        try:
            with open(path, encoding="utf-8", errors="replace") as stream:
                head = stream.read(4096)
        except OSError:
            head = ""
        _WORKLOAD_SNIFF[key] = re.match(r'\s*\{\s*"(nb_res|jobs)"', head) is not None
    return _WORKLOAD_SNIFF[key]


def workload_files() -> list[str]:
    """Workload JSON files under data/workloads/, as data paths."""
    return [value for value in data_files("workloads") if _looks_like_workload(DATA_ROOT / value)]


def checkpoint_files() -> list[str]:
    if not RESULTS_ROOT.is_dir():
        return []
    paths = sorted(RESULTS_ROOT.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [path.relative_to(PROJECT_ROOT).as_posix() for path in paths[:200]]


def row_label(label: str, classes: str = "row-label") -> Static:
    """A field label; leading spaces become indentation (Static drops them)."""
    widget = Static(label.strip(), classes=classes)
    widget.styles.padding = (0, 0, 0, 3 + len(label) - len(label.lstrip()))
    return widget


class TextInput(Input):
    """A text box whose text is selected when you enter it, by key or by a first click."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._focused_at = 0.0

    # Textual calls the handler of every class in the MRO (Input's after these)
    # unless prevent_default() stops it.
    def _on_focus(self, event: events.Focus) -> None:
        self._focused_at = time.monotonic()  # Input._on_focus then selects the text

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        # The click that focused the box keeps the text selected; later clicks place the cursor.
        if time.monotonic() - self._focused_at < 0.3:
            self.selection = Selection(0, len(self.value))
            event.prevent_default()


def field_row(
    label: str,
    value: object = "",
    *,
    id: str,
    placeholder: str = "",
    classes: str = "",
) -> Horizontal:
    """A label and a text box on one line."""
    return Horizontal(
        row_label(label),
        TextInput("" if value is None else str(value), placeholder=placeholder, id=id, classes=classes),
        classes="row",
    )


class FormScroll(VerticalScroll):
    """A scrolling form where Up/Down move between fields instead of scrolling."""

    BINDINGS = [
        Binding("up", "app.focus_previous", "Previous field", show=False),
        Binding("down", "app.focus_next", "Next field", show=False),
    ]


FIELD_KEYS = [
    Binding("up", "app.focus_previous", "Previous field", show=False),
    Binding("down", "app.focus_next", "Next field", show=False),
]
"""Bindings for a screen whose fields are not inside a FormScroll."""


class _Entry(TextInput):
    """The text box of a Picker; gives the list keys to its Picker first."""

    async def _on_key(self, event: events.Key) -> None:
        picker = next(a for a in self.ancestors if isinstance(a, Picker))
        if picker.use_key(event.key):
            event.stop()
            event.prevent_default()  # Input._on_key must not see it


class Picker(Vertical):
    DEFAULT_CSS = """
    Picker { height: auto; }
    Picker > .picker-row { height: 1; }
    Picker .picker-label { width: 34; text-wrap: nowrap; text-overflow: ellipsis; }
    Picker Input { width: 64; }
    Picker OptionList { display: none; height: auto; max-height: 10; margin: 0 0 0 34; width: 1fr; }
    Picker.-open OptionList { display: block; }
    """

    class Changed(Message):
        def __init__(self, picker: "Picker") -> None:
            super().__init__()
            self.picker = picker

    def __init__(
        self,
        label: str,
        suggest: Callable[[], list[str]],
        value: str = "",
        *,
        id: str,
        placeholder: str = "",
        describe: Callable[[], dict[str, tuple[str, str]]] | None = None,
    ) -> None:
        super().__init__(id=id)
        self.label = label
        self.suggest = suggest
        self.describe = describe  # value -> (title, dim detail); shown next to the value
        self.initial = value
        self.placeholder = placeholder
        self._shown: list[str] = []
        self._set_by_code: str | None = None  # a value set_value wrote; its Changed must not reopen the list

    def compose(self) -> ComposeResult:
        with Horizontal(classes="picker-row"):
            yield row_label(self.label, "picker-label")
            yield _Entry(self.initial, placeholder=self.placeholder)
        yield OptionList()

    @property
    def entry(self) -> Input:
        return self.query_one(_Entry)

    @property
    def value(self) -> str:
        return self.entry.value.strip()

    @property
    def is_open(self) -> bool:
        return self.has_class("-open")

    def set_value(self, value: str) -> None:
        if value != self.entry.value:
            self._set_by_code = value  # Input.Changed arrives later, as a message
        self.entry.value = value
        self.close()

    def open(self, query: str = "") -> None:
        """Show the options matching ``query`` (all of them for ''), current value highlighted."""
        details = self.describe() if self.describe else {}
        matches = filter_options(self.suggest(), query, details)
        self._shown = matches[:MAX_LISTED]
        listing = self.query_one(OptionList)
        listing.clear_options()
        rows = [Option(self._row(value, details), id=value) for value in self._shown]
        if not matches:
            rows.append(Option(Text("no match: any text is still allowed", style="dim"), disabled=True))
        elif len(matches) > MAX_LISTED:
            rows.append(Option(Text(f"+{len(matches) - MAX_LISTED} more, type to narrow", style="dim"), disabled=True))
        listing.add_options(rows)
        if self._shown:
            listing.highlighted = self._shown.index(self.value) if self.value in self._shown else 0
        self.add_class("-open")

    def close(self) -> None:
        self.remove_class("-open")

    @staticmethod
    def _row(value: str, details: dict[str, tuple[str, str]]) -> "Text | str":
        if value not in details:
            return value
        title, detail = details[value]
        if title == value:
            return Text.assemble((value, "bold"), "  ", (detail, "dim"))
        return Text.assemble((value, "bold"), "  ", (title, ""), "  ", (detail, "dim"))

    def _take(self, value: str) -> None:
        self.set_value(value)
        # Selected, so typing again replaces it, like entering the box.
        self.entry.selection = Selection(0, len(self.entry.value))
        self.post_message(self.Changed(self))

    def use_key(self, key: str) -> bool:
        """True when the key was used by the list (the box must not see it)."""
        listing = self.query_one(OptionList)
        if not self.is_open:
            if key == "enter":
                self.open()
                return True
            return False
        if key in ("down", "up"):
            listing.action_cursor_down() if key == "down" else listing.action_cursor_up()
            return True
        if key == "enter":
            if listing.highlighted is not None and listing.highlighted < len(self._shown):
                self._take(self._shown[listing.highlighted])
            else:
                self.close()
            return True
        if key == "escape":
            self.close()
            return True
        if key in ("tab", "shift+tab"):
            self.close()  # Tab moves on and keeps the typed text
        return False

    def on_input_changed(self, message: Input.Changed) -> None:
        message.stop()
        if self._set_by_code is not None and message.value == self._set_by_code:
            self._set_by_code = None
            return
        if self.entry.has_focus:
            self.open(message.value)
        self.post_message(self.Changed(self))

    def on_input_submitted(self, message: Input.Submitted) -> None:
        message.stop()  # Enter belongs to the list

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if isinstance(event.widget, _Entry):
            # Wait a moment: a mouse click on the list blurs the box first.
            self.set_timer(0.2, self._close_unless_focused)

    def _close_unless_focused(self) -> None:
        if not (self.entry.has_focus or self.query_one(OptionList).has_focus):
            self.close()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        message.stop()
        if message.option.id is not None:
            self._take(message.option.id)
        self.entry.focus()
