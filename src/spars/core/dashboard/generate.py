"""Screens that generate a synthetic workload or a platform file.

Opened from the experiment form, a screen returns the path of the file it
wrote (``workloads/json/x.json``), so the form can use it straight away.
"""

from __future__ import annotations

import re
from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Static

from spars.core.experiments.defaults import PROJECT_ROOT
from spars.core.platforms.generate import (
    DVFSProfile,
    PlatformConfig,
    generate_platform,
    warnings as platform_warnings,
    write_platform,
)
from spars.core.simulator.simulator import validate_workload
from spars.core.workloads.generate import (
    GeneratorConfig,
    generate_workload,
    mean_interarrival,
    summarize,
    warnings as workload_warnings,
    write_workload,
)

from .fields import FormScroll, Picker, TextInput, field_row, row_label

WORKLOAD_DIR = PROJECT_ROOT / "data" / "workloads" / "json"
PLATFORM_DIR = PROJECT_ROOT / "data" / "platforms"
NAME = re.compile(r"[A-Za-z0-9_.-]+")


class FormError(ValueError):
    """A field the user must correct; the message names it."""


def _number(text: str, label: str, kind: type = float) -> Any:
    try:
        return kind(text.strip())
    except ValueError:
        raise FormError(f"{label} must be {'a whole number' if kind is int else 'a number'}.") from None


def _file_name(text: str) -> str:
    name = text.strip().removesuffix(".json")
    if not NAME.fullmatch(name):
        raise FormError("File name: use letters, digits, '_', '-' and '.' only.")
    return name


class GenerateScreen(Screen[str]):
    """Label + box rows, a help line for the focused field, Ctrl+S to generate.

    ``pick=True`` (opened from the experiment form): after a successful
    generate the screen closes and returns the new file's data path.
    """

    BINDINGS = [
        Binding("ctrl+s", "generate", "Generate", priority=True),
        Binding("escape", "back", "Back"),
    ]
    TITLE_TEXT = ""
    HELP: dict[str, str] = {}

    def __init__(self, pick: bool = False) -> None:
        super().__init__()
        self.pick = pick

    def compose(self) -> ComposeResult:
        yield Static(self.TITLE_TEXT, id="title")
        with FormScroll(id="form"):
            yield from self.rows()
        yield Static(id="preview")
        yield Static(id="help", classes="hint")
        yield Static(id="status")
        yield Footer()

    def rows(self) -> ComposeResult:
        raise NotImplementedError

    def value(self, key: str) -> str:
        return self.query_one(f"#{key}", Input).value

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def help_for(self, key: str) -> str:
        return self.HELP.get(key, "")

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        # A Picker's box has no id of its own: use the nearest id that has help.
        for node in (event.widget, *event.widget.ancestors):
            text = self.help_for(node.id or "")
            if text:
                break
        self.query_one("#help", Static).update(text)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        self.focus_next()

    def status(self, message: str, error: bool = False) -> None:
        self.query_one("#status", Static).update(Text(message, style="red" if error else "green"))

    def action_back(self) -> None:
        self.dismiss(None)

    def action_generate(self) -> None:
        try:
            path, message = self.generate()
        except (FormError, ValueError, FileExistsError, OSError) as error:
            self.status(str(error), error=True)
            return
        if self.pick:
            self.app.notify(message)
            self.dismiss(path)
        else:
            self.status(message)

    def generate(self) -> tuple[str, str]:
        """Write the file; return its data path and a status line."""
        raise NotImplementedError


class GenerateWorkloadScreen(GenerateScreen):
    TITLE_TEXT = "Generate workload   ·   Ctrl+S writes data/workloads/json/<file name>.json"
    HELP = {
        "name": "File name without .json. An existing file is never overwritten.",
        "num_jobs": "Number of jobs.",
        "nb_res": "Nodes of the platform the workload is made for; no job asks for more.",
        "target_utilization": "Share of the platform's node-time the jobs ask for. It sets the arrival rate; "
        "above 1 more work arrives than the platform can do.",
        "mean_runtime": "Mean actual runtime of a job in seconds (exponential).",
        "resource_geometric_p": "Nodes per job are geometric with this p, in (0, 1], cut at nb_res. A larger p gives smaller jobs.",
        "reqtime_bias": "Mean requested time = mean runtime x this (exponential, drawn independently).",
        "seed": "Random seed: the same settings and seed give the same workload.",
        "profile_name": "Key every job uses in the workload's profiles.",
    }

    def rows(self) -> ComposeResult:
        d = GeneratorConfig()
        yield field_row("File name *", "synthetic-3000", id="name")
        yield field_row("Jobs", d.num_jobs, id="num_jobs", classes="kind-int")
        yield field_row("Platform nodes (nb_res)", d.nb_res, id="nb_res", classes="kind-int")
        yield field_row("Target utilization", d.target_utilization, id="target_utilization", classes="kind-float")
        yield field_row("Mean runtime (s)", d.mean_runtime, id="mean_runtime", classes="kind-float")
        yield field_row("Resource geometric p", d.resource_geometric_p, id="resource_geometric_p", classes="kind-float")
        yield field_row("Requested time bias", d.reqtime_bias, id="reqtime_bias", classes="kind-float")
        yield field_row("Seed", d.seed, id="seed", classes="kind-int")
        yield field_row("Profile name", d.profile_name, id="profile_name")

    def config(self) -> GeneratorConfig:
        return GeneratorConfig(
            num_jobs=_number(self.value("num_jobs"), "Jobs", int),
            nb_res=_number(self.value("nb_res"), "Platform nodes", int),
            target_utilization=_number(self.value("target_utilization"), "Target utilization"),
            mean_runtime=_number(self.value("mean_runtime"), "Mean runtime"),
            resource_geometric_p=_number(self.value("resource_geometric_p"), "Resource geometric p"),
            reqtime_bias=_number(self.value("reqtime_bias"), "Requested time bias"),
            seed=_number(self.value("seed"), "Seed", int),
            profile_name=self.value("profile_name").strip(),
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            config = self.config()
            text = Text(f"Mean interarrival time: {mean_interarrival(config):.0f} s", style="dim")
            for warning in workload_warnings(config):
                text.append(f"\nNote: {warning}", style="yellow")
        except (FormError, ValueError, ZeroDivisionError):
            text = Text("")
        self.query_one("#preview", Static).update(text)

    def generate(self) -> tuple[str, str]:
        name = _file_name(self.value("name"))
        config = self.config()
        workload = generate_workload(config)
        validate_workload(workload, config.nb_res)
        write_workload(workload, WORKLOAD_DIR / f"{name}.json")
        return f"workloads/json/{name}.json", f"Wrote workloads/json/{name}.json: {summarize(config, workload)}"


# Fields of one node type: (key, label, PlatformConfig attribute, kind).
NODE_FIELDS = [
    ("num_nodes", "Nodes", "num_nodes", int),
    ("node_name", "Node type name", "node_name", str),
    ("base_compute_power", "Compute power (W)", "base_compute_power", float),
    ("base_idle_power", "Idle power (W)", "base_idle_power", float),
    ("switching_on_power", "Switching-on power (W)", "switching_on_power", float),
    ("switching_off_power", "Switching-off power (W)", "switching_off_power", float),
    ("sleeping_power", "Sleeping power (W)", "sleeping_power", float),
    ("switching_on_time", "Switching-on time (s)", "switching_on_time", float),
    ("switching_on_std", "Switching-on std (s)", "switching_on_std", float),
    ("switching_off_time", "Switching-off time (s)", "switching_off_time", float),
    ("switching_off_std", "Switching-off std (s)", "switching_off_std", float),
    ("compute_speed", "Compute speed", "compute_speed_variation", float),
]
NODE_HELP = {
    "num_nodes": "Number of identical nodes of this type.",
    "node_name": "Name of this node type; every type needs its own name.",
    "base_compute_power": "Power in W while computing, at the base DVFS profile.",
    "base_idle_power": "Power in W while idle, at the base DVFS profile.",
    "switching_on_power": "Power in W while a node switches on.",
    "switching_off_power": "Power in W while a node switches off.",
    "sleeping_power": "Power in W while a node sleeps.",
    "switching_on_time": "Seconds to switch a node on.",
    "switching_on_std": "Standard deviation of that time in seconds (0 = always the same).",
    "switching_off_time": "Seconds to switch a node off.",
    "switching_off_std": "Standard deviation of that time in seconds.",
    "compute_speed": "Relative compute speed of these nodes at the base profile (1 = reference speed).",
    "dvfs_mode": "The DVFS profile these nodes start in: one of this type's profiles. Type or Enter to pick.",
    "add-dvfs": "Add a DVFS profile: factors that scale the base compute power, idle power and compute speed.",
    "remove-type": "Remove this node type (one must remain).",
    "dvfs-name": "Name of this DVFS profile.",
    "dvfs-compute": "Factor on the base compute power (1 = unchanged).",
    "dvfs-idle": "Factor on the base idle power.",
    "dvfs-speed": "Factor on the compute speed; a higher speed finishes jobs sooner.",
    "dvfs-remove": "Remove this DVFS profile (one must remain).",
}


class NodeTypeGroup(Vertical):
    """The fields of one node type; ids end in ``-<n>`` so several groups can coexist."""

    def __init__(self, number: int, name: str) -> None:
        super().__init__(classes="group node-type", id=f"type-{number}")
        self.number = number
        self.default_name = name
        self._rows = 0

    def compose(self) -> ComposeResult:
        n = self.number
        d = PlatformConfig()
        yield Static(f"Node type {n}", classes="subsection")
        for key, label, attribute, kind in NODE_FIELDS:
            value = self.default_name if key == "node_name" else getattr(d, attribute)
            yield field_row(f"  {label}", value, id=f"{key}-{n}", classes="" if kind is str else "kind-float")
        yield Static("  DVFS profiles   (name, then factors of compute power, idle power, compute speed)", classes="hint")
        yield Vertical(self.dvfs_row(DVFSProfile("base")), id=f"dvfs-rows-{n}", classes="group")
        yield Picker("  Initial DVFS mode", self.profile_names, d.dvfs_mode, id=f"dvfs_mode-{n}")
        with Horizontal(classes="buttons"):
            yield Button("Add DVFS profile", id=f"add-dvfs-{n}")
            yield Button("Remove node type", id=f"remove-type-{n}")

    def dvfs_row(self, profile: DVFSProfile) -> Horizontal:
        self._rows += 1
        tag = f"{self.number}-{self._rows}"
        return Horizontal(
            row_label("    profile"),
            TextInput(profile.name, placeholder="name *", classes="dvfs-name", id=f"dvfs-name-{tag}"),
            TextInput(str(profile.compute_power), classes="dvfs-factor", id=f"dvfs-compute-{tag}"),
            TextInput(str(profile.idle_power), classes="dvfs-factor", id=f"dvfs-idle-{tag}"),
            TextInput(str(profile.compute_speed), classes="dvfs-factor", id=f"dvfs-speed-{tag}"),
            Button("Remove", id=f"dvfs-remove-{tag}", classes="dvfs-remove"),
            classes="dvfs-row",
        )

    def profile_names(self) -> list[str]:
        return [row.query(Input).first().value.strip() for row in self.query(".dvfs-row")]

    def value(self, key: str) -> str:
        return self.query_one(f"#{key}-{self.number}", Input).value

    def config(self) -> PlatformConfig:
        type_name = self.value("node_name").strip() or f"type {self.number}"
        values: dict[str, Any] = {}
        for key, label, attribute, kind in NODE_FIELDS:
            text = self.value(key)
            values[attribute] = text.strip() if kind is str else _number(text, f"{type_name}: {label}", kind)
        profiles = []
        for row in self.query(".dvfs-row"):
            name, compute, idle, speed = (box.value for box in row.query(Input))
            label = f"{type_name}, DVFS {name.strip() or '?'}"
            profiles.append(DVFSProfile(
                name.strip(),
                _number(compute, f"{label}: compute power factor"),
                _number(idle, f"{label}: idle power factor"),
                _number(speed, f"{label}: compute speed factor"),
            ))
        mode = self.query_one(f"#dvfs_mode-{self.number}", Picker).value
        return PlatformConfig(**values, dvfs_profiles=tuple(profiles), dvfs_mode=mode)


class GeneratePlatformScreen(GenerateScreen):
    TITLE_TEXT = "Generate platform   ·   Ctrl+S writes data/platforms/<file name>.json"
    HELP = {
        "name": "File name without .json. An existing file is never overwritten.",
        "add-type": "Add another node type: a group of identical nodes with its own power and speed.",
    }

    def rows(self) -> ComposeResult:
        yield field_row("File name *", "my-platform", id="name")
        yield Vertical(id="node-types", classes="group")
        with Horizontal(classes="buttons"):
            yield Button("Add node type", id="add-type")

    async def on_mount(self) -> None:
        self._types = 0
        await self.add_type()
        self.query_one(Input).focus()

    async def add_type(self) -> None:
        self._types += 1
        await self.query_one("#node-types").mount(NodeTypeGroup(self._types, f"type{self._types}" if self._types > 1 else PlatformConfig().node_name))

    def help_for(self, key: str) -> str:
        if key in self.HELP:
            return self.HELP[key]
        for prefix in sorted(NODE_HELP, key=len, reverse=True):
            if key.startswith(prefix):
                return NODE_HELP[prefix]
        return ""

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id or ""
        group = next((a for a in event.button.ancestors if isinstance(a, NodeTypeGroup)), None)
        if button == "add-type":
            await self.add_type()
        elif button.startswith("remove-type") and group is not None:
            if len(self.query(NodeTypeGroup)) > 1:
                await group.remove()
            else:
                self.status("A platform needs at least one node type.", error=True)
        elif button.startswith("add-dvfs") and group is not None:
            await self.query_one(f"#dvfs-rows-{group.number}").mount(group.dvfs_row(DVFSProfile("", 1.0, 1.0, 1.0)))
        elif "dvfs-remove" in event.button.classes and group is not None:
            if len(group.query(".dvfs-row")) > 1:
                await event.button.parent.remove()
            else:
                self.status("A node type needs at least one DVFS profile.", error=True)

    def configs(self) -> list[PlatformConfig]:
        return [group.config() for group in self.query(NodeTypeGroup)]

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            configs = self.configs()
            text = Text(f"{sum(c.num_nodes for c in configs)} nodes in {len(configs)} node type(s)", style="dim")
            for config in configs:
                for warning in platform_warnings(config):
                    text.append(f"\nNote: {warning}", style="yellow")
        except (FormError, ValueError):
            text = Text("")
        self.query_one("#preview", Static).update(text)

    def generate(self) -> tuple[str, str]:
        name = _file_name(self.value("name"))
        configs = self.configs()
        write_platform(generate_platform(configs), PLATFORM_DIR / f"{name}.json")
        nodes = sum(config.num_nodes for config in configs)
        return f"platforms/{name}.json", f"Wrote platforms/{name}.json: {nodes} nodes in {len(configs)} node type(s)"
