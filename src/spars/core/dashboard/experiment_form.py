"""Dashboard screens to list, create, edit and delete experiments.

A set is a folder under src/spars/experiments/; each experiment the dashboard
makes is one file in it (``<name>.py``) with one experiment. The Experiments
screen lists the files of the current set; Enter opens the form. A file
written by hand with several experiments opens a list of them first. All
writes go through spars.core.experiments.store, which only touches definition
files that are plain data.

An experiment can be saved before it is complete (a platform not made yet,
say); it is marked incomplete and cannot run until the problems are fixed.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Input, Static, TextArea

from spars.core.experiments.configuration import merge_overrides, set_nested
from spars.core.experiments.defaults import BASE_CONFIG
from spars.core.registry import resolve_run_components
from spars.core.simulator.simulator import DEFAULT_STALL_TIMEOUT
from spars.core.simulator.platform_control import VALID_OVERRUN_POLICIES
from spars.policies.discovery import options as policy_options, resolve
from spars.core.experiments import store

from .app import Choice, ConfirmScreen
from .fields import FormScroll, Picker, TextInput, checkpoint_files, data_files, row_label, workload_files
from .generate import GeneratePlatformScreen, GenerateWorkloadScreen
from .settings import LOG_LEVELS

_MISSING = object()
_YES_NO = ["Yes", "No"]


def _get(tree: dict, path: tuple[str, ...], default: Any = _MISSING) -> Any:
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return default
        tree = tree[key]
    return tree


def _kind_for(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "yaml"


def _show(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else json.dumps(value))


class FormError(Exception):
    pass


HELP = {
    "name": "Required. Short unique name; it labels the result folders.",
    "display_name": "Optional. Friendlier name shown in the dashboard and result tables.",
    "enabled": "No = the experiment stays in the file but is skipped when the definition runs.",
    "paths.workload": "Required to run. Job trace to simulate. Type to search, Enter for the list; "
    "'New workload' makes one without leaving this form.",
    "paths.platform": "Required to run. Machine description (nodes, power states, transition times). "
    "'New platform' makes one without leaving this form.",
    "scheduler": "Decides which waiting job starts next and on which nodes. Type to search names and "
    "descriptions, Enter for the list. Your own class under src/spars/policies/scheduler/ is listed too.",
    "power_management": "Decides when idle nodes are switched off or on. Type to search, Enter for the list.",
    "new-workload": "Generate a workload and use it here; the form stays as it is.",
    "new-platform": "Generate a platform and use it here; the form stays as it is.",
    "run.start_time": "Simulated clock at the start: seconds, a date string, or 'now'.",
    "run.overrun_policy": "continue = a job runs for its real runtime; "
    "terminate = a job is cut at its requested time.",
    "run.stall_timeout": "Stop the run (and warn) when jobs have waited this many simulated seconds "
    "with nothing running and none started or finished. Empty = never stop. In RL training only that epoch ends.",
    "run.force_wakeup": "Safety net for RL runs: if jobs are unfinished but only RL calls remain, "
    "nodes are woken so the simulation cannot stall.",
    "rl.enabled": "Yes = a reinforcement-learning policy takes decisions. No = the other RL settings are not needed.",
    "rl.learn": "Yes = train the agent. No = only run (evaluate) it, usually with a checkpoint.",
    "rl.type": "When the policy is called: discrete = every dt seconds, continuous = on job events.",
    "rl.dt": "Seconds between policy calls (discrete; must be > 0).",
    "rl.epochs": "Number of training passes over the workload.",
    "rl.episode_batch_size": "Decisions collected before each model update.",
    "rl.seed": "Random seed, for repeatable training.",
    "rl.device": "gpu or cpu for the model.",
    "rl.checkpoint": "Optional saved model (agent_checkpoint.pt) to start from or to evaluate.",
    "rl.assign": "Which agent from rl.agents is used.",
    "has_validation": "Yes = pick the best epoch by its score on a separate workload instead of the training score.",
    "rl.validation.workload": "Held-out workload; must differ from the training workload.",
    "rl.validation.patience": "Stop after this many epochs without improvement (empty = never).",
    "rl.validation.window": "Odd number: epochs averaged around each epoch when ranking them.",
    "gym.feature_extractor": "How the simulator state becomes model input.",
    "gym.translator": "How the model output becomes node actions.",
    "gym.learner": "Training algorithm.",
    "gym.reward.name": "Reward function.",
    "logging.level": "How much the simulator logs. TRACE levels can write very large files.",
    "logging.file": "Log file path (empty = no file).",
    "add-sweep": "Add a parameter to sweep: the experiment runs once per value (or combination).",
    "edit-test": "Settings that replace the experiment's own when the trained agent is tested (YAML).",
    "edit-curriculum": "Training stages, each with its own overrides (YAML).",
    "edit-scheduler-config": "Edit the scheduler's settings as YAML: use it for a policy that lists no settings.",
    "edit-power_management-config": "Edit the power manager's settings as YAML: use it for a policy that lists no settings.",
    "edit-agents": "Edit all agents as YAML: add a model or any optimizer argument.",
    "save": "Write the file. An incomplete experiment is saved too; it cannot run until it is complete.",
    "cancel": "Close without saving.",
}
POLICY_KEY_HELP = {
    "timeout": "Seconds a node may stay idle before it is switched off.",
    "alpha": "ICON: weight of the expected power-induced waiting time.",
    "beta": "ICON: weight of the expected energy.",
    "min_spare_nodes": "ICON: fewest nodes kept ready as spare capacity.",
    "max_spare_nodes": "ICON: most nodes kept ready as spare capacity (empty = no limit).",
}


def policy_doc(kind: str, name: str) -> str:
    """First docstring line of a policy class, or '' (read from its source)."""
    return store.policy_summary(kind, name)


def _policy_values(kind: str) -> list[str]:
    return [value for value, _, _ in policy_options(kind)]


def _policy_details(kind: str) -> dict[str, tuple[str, str]]:
    """value -> (title, description), for the pick-list; search also matches the description."""
    return {value: (title, store.policy_summary(kind, path)) for value, title, path in policy_options(kind)}


class FormChoice(Choice):
    LABEL_WIDTH = 31  # the form's label column (34) minus the 3-column margin


@dataclass
class Field:
    label: str
    path: tuple[str, ...]
    kind: str  # str int float num bool choice pick yaml
    widget_id: str
    group: str
    nullable: bool = False
    always: bool = False  # write even when equal to the default
    manual: bool = False  # collected by the form, not by path
    default: Any = _MISSING  # value that needs no override; BASE_CONFIG's when unset


class NameScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, question: str, value: str = "") -> None:
        super().__init__()
        self.question = question
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.question)
            yield TextInput(self.value, id="name")

    def on_input_submitted(self, message: Input.Submitted) -> None:
        self.dismiss(message.value.strip() or None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class YamlScreen(ModalScreen[Any]):
    BINDINGS = [
        Binding("ctrl+s", "apply", "Apply"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, title: str, value: Any, empty: Any) -> None:
        super().__init__()
        self.title_text = title
        self.empty = empty
        self.initial = yaml.safe_dump(value, sort_keys=False) if value else ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Static(f"{self.title_text}   (YAML, Ctrl+S apply, Esc cancel)")
            yield TextArea(self.initial, id="yaml")
            yield Static(id="yaml-error")

    def action_apply(self) -> None:
        try:
            value = yaml.safe_load(self.query_one("#yaml", TextArea).text)
        except yaml.YAMLError as error:
            self.query_one("#yaml-error", Static).update(Text(str(error), style="red"))
            return
        self.dismiss(self.empty if value is None else value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ExperimentFormScreen(Screen[bool]):
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "app.focus_previous", "Up", show=False),
        Binding("down", "app.focus_next", "Down", show=False),
    ]

    def __init__(
        self,
        definition: store.DefinitionFile | None,
        index: int | None,
        experiment: dict[str, Any],
        other_names: set[str],
        research: str | None = None,
    ) -> None:
        """``definition`` None: a new experiment, saved as its own file in set ``research``."""
        super().__init__()
        self.definition = definition
        self.research = research or (definition.path.parent.name if definition else "")
        self.index = index
        self.experiment = copy.deepcopy(experiment)
        self.other_names = other_names
        self.overrides = copy.deepcopy(self.experiment.get("overrides", {}))
        self.test_overrides = copy.deepcopy(self.experiment.get("test_overrides") or {})
        self.curriculum = copy.deepcopy(self.experiment.get("curriculum") or [])
        self.fields: dict[str, Field] = {}
        self._counter = 0
        self._built_policy: dict[str, str] = {}
        self._saved_state: str | None = None

    # -- field helpers -------------------------------------------------------

    def effective(self) -> dict[str, Any]:
        return merge_overrides(BASE_CONFIG, self.overrides)

    def _register(self, label, path, kind, group, **options) -> str:
        self._counter += 1
        widget_id = f"fld{self._counter}"
        self.fields[widget_id] = Field(label, tuple(path), kind, widget_id, group, **options)
        return widget_id

    def _drop(self, *groups: str) -> None:
        self.fields = {k: f for k, f in self.fields.items() if f.group not in groups}

    def f_text(self, label, path, kind, value, group, **options) -> Widget:
        widget_id = self._register(label, path, kind, group, **options)
        return Horizontal(
            row_label(label),
            TextInput(_show(value), id=widget_id, classes=f"kind-{kind}"),
            classes="row",
        )

    def f_choice(self, label, path, choices, value, group, **options) -> Widget:
        widget_id = self._register(label, path, "choice", group, **options)
        return FormChoice(label, choices, str(value), id=widget_id)

    def f_bool(self, label, path, value, group, **options) -> Widget:
        widget_id = self._register(label, path, "bool", group, **options)
        return FormChoice(label, _YES_NO, "Yes" if value else "No", id=widget_id)

    def f_pick(self, label, path, suggest, value, group, **options) -> Widget:
        widget_id = self._register(label, path, "pick", group, **options)
        return Picker(label, suggest, _show(value), id=widget_id)

    def f_auto(self, label, path, value, group, **options) -> Widget:
        kind = _kind_for(value)
        if kind == "bool":
            return self.f_bool(label, path, value, group, **options)
        return self.f_text(label, path, kind, value, group, **options)

    def parse(self, field: Field) -> Any:
        widget = self.query_one(f"#{field.widget_id}")
        if field.kind == "bool":
            return widget.value == "Yes"
        if field.kind == "choice":
            return widget.value
        raw = widget.value.strip()
        if raw == "" and field.kind not in ("str", "pick"):  # an empty pick (no platform yet) is a draft
            if field.nullable or field.kind == "yaml":
                return None
            raise FormError(f"{field.label.strip()} must not be empty")
        try:
            if field.kind == "int":
                return int(raw)
            if field.kind == "float":
                return float(raw)
            if field.kind in ("num", "numstr"):
                try:
                    number = float(raw)
                except ValueError:
                    if field.kind == "numstr":
                        return raw
                    raise
                return int(number) if number.is_integer() else number
            if field.kind == "yaml":
                return yaml.safe_load(raw)
        except (ValueError, yaml.YAMLError) as error:
            raise FormError(f"{field.label.strip()}: {error}") from error
        return raw

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        eff = self.effective()
        title = "New experiment" if self.index is None else f"Edit {self.experiment['name']}"
        where = self.definition.path.name if self.definition else f"set {self.research}"
        yield Static(f"{title}   ·   {where}   ·   Ctrl+S saves", id="title")
        with FormScroll(id="form"):
            yield Static("Experiment   (* required)", classes="section")
            yield self._plain("Name *", "name", self.experiment.get("name", ""))
            display = self.experiment.get("display_name", "")
            yield self._plain(
                "Display name", "display_name",
                "" if display == self.experiment.get("name") else display, "empty = the name",
            )
            yield FormChoice(
                "Enabled", _YES_NO, "Yes" if self.experiment.get("enabled", True) else "No", id="enabled"
            )
            yield Static("Paths", classes="section")
            yield self.f_pick(
                "Workload *", ("paths", "workload"), workload_files,
                eff["paths"]["workload"], "top", always=True,
            )
            yield self.f_pick(
                "Platform *", ("paths", "platform"), lambda: data_files("platforms"),
                eff["paths"]["platform"], "top", always=True,
            )
            with Horizontal(classes="buttons"):
                yield Button("New workload", id="new-workload")
                yield Button("New platform", id="new-platform")
            yield Static("Run", classes="section")
            run = eff["run"]
            scheduler, power = resolve_run_components(run)
            yield Picker("Scheduler", lambda: _policy_values("scheduler"), scheduler["name"], id="scheduler", describe=lambda: _policy_details("scheduler"))
            yield Vertical(id="sched-rows", classes="group")
            yield Picker("Power management", lambda: _policy_values("power_management"), power["name"], id="power_management", describe=lambda: _policy_details("power_management"))
            yield Vertical(id="power-rows", classes="group")
            yield self.f_text("Start time", ("run", "start_time"), "numstr", run["start_time"], "top")
            yield self.f_choice(
                "Overrun policy", ("run", "overrun_policy"), sorted(VALID_OVERRUN_POLICIES),
                run["overrun_policy"], "top",
            )
            yield self.f_bool("Force wakeup", ("run", "force_wakeup"), run["force_wakeup"], "top")
            yield self.f_text(
                "Stall timeout (s)", ("run", "stall_timeout"), "num",
                run.get("stall_timeout", DEFAULT_STALL_TIMEOUT), "top", nullable=True,
                default=DEFAULT_STALL_TIMEOUT,
            )
            yield Static("Reinforcement learning", classes="section")
            yield self.f_bool("Enabled", ("rl", "enabled"), eff["rl"]["enabled"], "top", always=True)
            yield Vertical(id="rl-rows", classes="group")
            yield Static("Logging", classes="section")
            yield self.f_choice("Level", ("logging", "level"), list(LOG_LEVELS), eff["logging"]["level"], "top")
            yield self.f_text("File", ("logging", "file"), "str", eff["logging"]["file"], "top")
            yield Static("Sweep   (parameter path = values, comma separated)", classes="section")
            yield Vertical(id="sweep-rows", classes="group")
            with Horizontal(classes="buttons"):
                yield Button("Add sweep", id="add-sweep")
            yield Static("Advanced", classes="section")
            with Horizontal(classes="buttons"):
                yield Button("test_overrides (YAML)", id="edit-test")
                yield Button("curriculum (YAML)", id="edit-curriculum")
        yield Static(id="help", classes="hint")
        yield Static(id="errors")
        with Horizontal(classes="buttons"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Cancel", id="cancel")
        yield Footer()

    def _plain(self, label: str, widget_id: str, value: str, placeholder: str = "") -> Widget:
        return Horizontal(
            row_label(label), TextInput(value, id=widget_id, placeholder=placeholder), classes="row"
        )

    async def on_mount(self) -> None:
        eff = self.effective()
        scheduler, power = resolve_run_components(eff["run"])
        await self._rebuild_policy("scheduler", scheduler["name"], scheduler["config"])
        await self._rebuild_policy("power_management", power["name"], power["config"])
        if eff["rl"]["enabled"]:
            await self._rebuild_rl()
        sweep = self.experiment.get("sweep") or {}
        await self.query_one("#sweep-rows").mount(
            *(self._sweep_row(path, values) for path, values in sweep.items())
        )
        self.query_one("#name", Input).focus()
        self._saved_state = self._snapshot()

    def _snapshot(self) -> str | None:
        try:
            return repr(self.collect())
        except FormError:
            return None  # a half-typed value counts as a change

    # -- policy components ---------------------------------------------------

    async def _rebuild_policy(self, kind: str, name: str, existing: dict[str, Any]) -> None:
        container = self.query_one("#sched-rows" if kind == "scheduler" else "#power-rows")
        await container.remove_children()
        self._drop(kind)
        config = {**store.policy_defaults(kind, name), **existing}
        rows = [
            self.f_auto(f"  {key}", ("run", kind, "config", key), value, kind, manual=True)
            for key, value in config.items()
        ] or [Static("(no settings)", classes="hint")]
        rows.append(Button("Edit settings (YAML)", id=f"edit-{kind}-config"))
        self._built_policy[kind] = name
        await container.mount(*rows)

    def _policy_values(self, kind: str) -> dict[str, Any]:
        values = {}
        for field in self.fields.values():
            if field.group == kind:
                try:
                    values[field.path[-1]] = self.parse(field)
                except FormError:
                    pass
        return values

    # -- RL ------------------------------------------------------------------

    def agent_names(self) -> list[str]:
        return list(self.effective()["rl"]["agents"])

    async def _rebuild_rl(self) -> None:
        container = self.query_one("#rl-rows")
        await container.remove_children()
        self._drop("rl", "val", "agent")
        eff = self.effective()
        rl, gym = eff["rl"], eff["gym"]
        g = "rl"
        learner = gym["learner"]
        rows: list[Widget] = [
            self.f_bool("Learn", ("rl", "learn"), rl["learn"], g),
            self.f_choice("Type", ("rl", "type"), ["discrete", "continuous"], rl["type"], g),
            self.f_text("dt (s)", ("rl", "dt"), "num", rl["dt"], g),
            self.f_text("Epochs", ("rl", "epochs"), "int", rl["epochs"], g),
            self.f_text("Episode batch size", ("rl", "episode_batch_size"), "int", rl["episode_batch_size"], g),
            self.f_text("Seed", ("rl", "seed"), "int", rl["seed"], g),
            self.f_choice("Device", ("rl", "device"), ["gpu", "cpu"], rl["device"], g),
            self.f_pick("Checkpoint", ("rl", "checkpoint"), checkpoint_files, rl["checkpoint"], g, nullable=True),
            self.f_pick("Agent", ("rl", "assign"), self.agent_names, rl["assign"], g, always=True),
            FormChoice("Validation", _YES_NO, "Yes" if rl["validation"] else "No", id="has_validation"),
            Vertical(id="val-rows", classes="group"),
            Static("Agent", classes="subsection"),
            Vertical(id="agent-rows", classes="group"),
            Static("Gym", classes="subsection"),
            self.f_pick("Feature extractor", ("gym", "feature_extractor"), lambda: store.rl_objects("feature_extraction"), gym["feature_extractor"], g),
            self.f_pick("Translator", ("gym", "translator"), lambda: store.rl_objects("action_translator"), gym["translator"], g),
            self.f_pick("Learner", ("gym", "learner"), lambda: store.rl_objects("learn"), learner, g)
            if isinstance(learner, str)
            else self.f_text("Learner", ("gym", "learner"), "yaml", learner, g),
            self.f_pick("Reward", ("gym", "reward", "name"), lambda: store.rl_objects("Reward"), gym["reward"]["name"], g),
        ]
        rows += [
            self.f_auto(f"  reward.{key}", ("gym", "reward", "params", key), value, g)
            for key, value in gym["reward"].get("params", {}).items()
        ]
        await container.mount(*rows)
        await self._rebuild_validation()
        await self._rebuild_agent()

    async def _rebuild_validation(self) -> None:
        container = self.query_one("#val-rows")
        await container.remove_children()
        self._drop("val")
        if self.query_one("#has_validation", Choice).value != "Yes":
            return
        v = self.effective()["rl"]["validation"] or {"workload": "", "patience": None, "window": 1}
        await container.mount(
            self.f_pick("workload", ("rl", "validation", "workload"), workload_files, v["workload"], "val", always=True),
            self.f_text("patience", ("rl", "validation", "patience"), "int", v.get("patience"), "val", nullable=True, always=True),
            self.f_text("window", ("rl", "validation", "window"), "int", v.get("window", 1), "val", always=True),
        )

    async def _rebuild_agent(self) -> None:
        container = self.query_one("#agent-rows")
        await container.remove_children()
        self._drop("agent")
        name = self._assigned_agent()
        agent = self.effective()["rl"]["agents"].get(name)
        if agent is None:
            await container.mount(Static(f"No settings for agent {name!r}", classes="hint"))
            return
        base = ("rl", "agents", name)
        # The class is fixed: overrides merge into the base agent, so a new model
        # is added as another agent through the YAML button.
        rows: list[Widget] = [
            Horizontal(
                row_label("  class"),
                TextInput(agent["class"], disabled=True),
                classes="row",
            )
        ]
        parameters = {**store.model_defaults(agent["class"]), **agent.get("params", {})}
        for key, value in parameters.items():
            default = _get(BASE_CONFIG, (*base, "params", key), store.model_defaults(agent["class"]).get(key))
            rows.append(
                self.f_auto(f"params.{key}", (*base, "params", key), value, "agent", default=default)
            )
        optimizer = agent.get("optimizer")
        if optimizer:
            rows.append(self.f_pick(
                "optimizer", (*base, "optimizer", "class"),
                lambda: ["torch.optim:Adam", "torch.optim:AdamW", "torch.optim:SGD"], optimizer["class"], "agent",
            ))
            rows += [
                self.f_auto(f"optimizer.{k}", (*base, "optimizer", "params", k), v, "agent")
                for k, v in optimizer.get("params", {}).items()
            ]
        rows.append(Button("Edit agents (YAML)", id="edit-agents"))
        await container.mount(*rows)

    def _assigned_agent(self) -> str:
        for field in self.fields.values():
            if field.path == ("rl", "assign"):
                return self.query_one(f"#{field.widget_id}", Picker).value
        return str(self.effective()["rl"]["assign"])

    # -- sweep ---------------------------------------------------------------

    def _sweep_row(self, path: str = "", values: Any = ()) -> Widget:
        self._counter += 1
        return Horizontal(
            Picker("parameter", self._sweep_paths, path, id=f"sw{self._counter}"),
            TextInput(
                ", ".join(json.dumps(v) for v in values),
                classes="sweep-values",
            ),
            Button("Remove", id=f"sd{self._counter}", classes="sweep-remove"),
            classes="sweep-row",
        )

    def _sweep_paths(self) -> list[str]:
        paths: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict) and node:
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}" if prefix else key)
            else:
                paths.append(prefix)

        walk(self.effective(), "")
        return sorted(paths)

    # -- events --------------------------------------------------------------

    def on_descendant_focus(self, event) -> None:
        widget = event.widget
        key = None
        for node in (widget, *widget.ancestors):
            node_id = node.id or ""
            if node_id in self.fields:
                field = self.fields[node_id]
                key = ".".join(field.path)
                if key in ("paths.workload", "paths.platform"):
                    self._help(self._paths_help(key))
                    return
                if field.group in ("scheduler", "power_management"):
                    default = store.policy_defaults(field.group, self._built_policy.get(field.group, ""))
                    text = POLICY_KEY_HELP.get(field.path[-1]) or (
                        f"{field.path[-1]}: setting of the selected {field.group.replace('_', ' ')}"
                        + (f" (default {default[field.path[-1]]!r})." if field.path[-1] in default else ".")
                    )
                    self._help(text)
                    return
                if field.group == "agent":
                    self._help(HELP.get(key) or f"Argument of the model or optimizer ({field.path[-1]}); the default is the class's own.")
                    return
                break
            if node_id in ("scheduler", "power_management"):
                name = self._built_policy.get(node_id) or ""
                doc = policy_doc(node_id, name)
                self._help(HELP[node_id] + (f"  [{name}: {doc}]" if doc else ""))
                return
            if node_id in HELP:
                key = node_id
                break
            if "sweep-values" in node.classes:
                self._help("Values to try, comma separated, e.g. 3600, 7200, 10800")
                return
            if "sweep-remove" in node.classes:
                self._help("Remove this sweep row.")
                return
            if node_id.startswith("sw"):
                self._help("Config path to sweep, e.g. run.power_management.config.timeout")
                return
        self._help(HELP.get(key or "", ""))

    def _field_value(self, path: tuple[str, ...]) -> str:
        field = next((f for f in self.fields.values() if f.path == path), None)
        return self.query_one(f"#{field.widget_id}", Picker).value if field else ""

    def _paths_help(self, key: str) -> str:
        summary = store.workload_summary(
            self._field_value(("paths", "workload")), self._field_value(("paths", "platform"))
        )
        return f"{HELP[key]}\n{summary}"

    def _help(self, text: str) -> None:
        self.query_one("#help", Static).update(text)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        message.stop()
        self.focus_next()

    async def on_picker_changed(self, message: Picker.Changed) -> None:
        picker = message.picker
        field = self.fields.get(picker.id or "")
        if field and field.path in (("paths", "workload"), ("paths", "platform")):
            self._help(self._paths_help(".".join(field.path)))
        if picker.id in ("scheduler", "power_management"):
            name = picker.value
            try:
                resolve(picker.id, name)
                valid = True
            except ValueError:
                valid = False
            if valid and name != self._built_policy.get(picker.id):
                values = self._policy_values(picker.id)
                allowed = store.policy_defaults(picker.id, name)
                if store.declared_defaults(picker.id, name) is not None:  # it lists its settings
                    values = {k: v for k, v in values.items() if k in allowed}
                await self._rebuild_policy(picker.id, name, values)
        elif picker.id in {f.widget_id for f in self.fields.values() if f.path == ("rl", "assign")}:
            if picker.value in self.agent_names():
                await self._rebuild_agent()

    async def on_choice_changed(self, message: Choice.Changed) -> None:
        choice = message.choice
        field = self.fields.get(choice.id or "")
        if field and field.path == ("rl", "enabled"):
            container = self.query_one("#rl-rows")
            if choice.value == "Yes":
                await self._rebuild_rl()
            else:
                await container.remove_children()
                self._drop("rl", "val", "agent")
        elif choice.id == "has_validation":
            await self._rebuild_validation()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        rows = self.query_one("#sweep-rows")
        if event.button.id in ("new-workload", "new-platform"):
            path = ("paths", "workload") if event.button.id == "new-workload" else ("paths", "platform")
            screen = GenerateWorkloadScreen(pick=True) if path[1] == "workload" else GeneratePlatformScreen(pick=True)
            self.app.push_screen(screen, lambda made, path=path: self._use_new_file(path, made))
        elif event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "add-sweep":
            await rows.mount(self._sweep_row())
        elif "sweep-remove" in event.button.classes:
            await event.button.parent.remove()
        elif event.button.id == "edit-test":
            self.app.push_screen(YamlScreen("test_overrides", self.test_overrides, {}), self._set("test_overrides"))
        elif event.button.id in ("edit-scheduler-config", "edit-power_management-config"):
            kind = event.button.id[len("edit-"):-len("-config")]
            self.app.push_screen(
                YamlScreen(f"run.{kind}.config", self._policy_values(kind), {}),
                lambda value, kind=kind: self._set_policy_config(kind, value),
            )
        elif event.button.id == "edit-agents":
            agents = self.effective()["rl"]["agents"]
            self.app.push_screen(YamlScreen("rl.agents", agents, agents), self._set_agents)
        elif event.button.id == "edit-curriculum":
            self.app.push_screen(YamlScreen("curriculum", self.curriculum, []), self._set("curriculum"))

    def _use_new_file(self, path: tuple[str, ...], made: str | None) -> None:
        """A generator screen opened from this form wrote ``made``: use it."""
        if not made:
            return
        field = next(f for f in self.fields.values() if f.path == path)
        picker = self.query_one(f"#{field.widget_id}", Picker)
        picker.set_value(made)
        picker.entry.focus()
        self._help(self._paths_help(".".join(path)))

    async def _set_policy_config(self, kind: str, value: Any) -> None:
        if isinstance(value, dict):
            await self._rebuild_policy(kind, self._built_policy[kind], value)
        elif value is not None:
            self.show_errors([f"run.{kind}.config must be a mapping of setting names to values"])

    async def _set_agents(self, value: Any) -> None:
        if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
            self.overrides.setdefault("rl", {})["agents"] = value
            await self._rebuild_agent()
        elif value is not None:
            self.show_errors(["rl.agents must map agent names to {class, params, optimizer}"])

    def _set(self, attribute: str) -> Callable[[Any], None]:
        def apply(value: Any) -> None:
            if value is not None:
                setattr(self, attribute, value)
        return apply

    # -- save ----------------------------------------------------------------

    def collect(self) -> dict[str, Any]:
        overrides = copy.deepcopy(self.overrides)
        for field in self.fields.values():
            if field.manual:
                continue
            value = self.parse(field)
            baseline = _get(BASE_CONFIG, field.path) if field.default is _MISSING else field.default
            if field.always or _get(overrides, field.path) is not _MISSING or value != baseline:
                set_nested(overrides, ".".join(field.path), value)

        for kind in ("scheduler", "power_management"):
            config = {
                f.path[-1]: self.parse(f)
                for f in self.fields.values()
                if f.group == kind
            }
            name = self.query_one(f"#{kind}", Picker).value
            if not name:
                raise FormError(f"{kind.replace('_', ' ')} must not be empty")
            overrides.setdefault("run", {})[kind] = {"name": name, "config": config}

        rl_on = self._rl_enabled()
        if not rl_on:
            # Keep what the file already says about RL; it is unused while RL is off.
            overrides["rl"] = {
                **copy.deepcopy(self.overrides.get("rl", {})),
                **{k: v for k, v in overrides.get("rl", {}).items() if k in ("enabled", "learn", "checkpoint")},
            }
        else:
            # An explicit None only when the file had a block that is now off.
            if self.query_one("#has_validation", Choice).value != "Yes":
                if _get(self.overrides, ("rl", "validation"), None) is not None:
                    overrides["rl"]["validation"] = None

        sweep = {}
        for row in self.query("#sweep-rows > .sweep-row"):
            path = row.query_one(Picker).value
            text = row.query_one(".sweep-values", Input).value.strip()
            if not path and not text:
                continue
            try:
                values = yaml.safe_load(f"[{text}]")
            except yaml.YAMLError as error:
                raise FormError(f"Sweep {path}: {error}") from error
            if not path or not values:
                raise FormError("Each sweep needs a parameter and at least one value")
            sweep[path] = values

        experiment = {k: v for k, v in self.experiment.items() if k not in ("overrides", "sweep", "test_overrides", "curriculum")}
        experiment.update(
            name=self.query_one("#name", Input).value.strip(),
            display_name=self.query_one("#display_name", Input).value.strip()
            or self.query_one("#name", Input).value.strip(),
            enabled=self.query_one("#enabled", Choice).value == "Yes",
            overrides=overrides,
        )
        if sweep:
            experiment["sweep"] = sweep
        if self.test_overrides:
            experiment["test_overrides"] = self.test_overrides
        if self.curriculum:
            experiment["curriculum"] = self.curriculum
        return experiment

    def _rl_enabled(self) -> bool:
        return any(
            self.parse(f) for f in self.fields.values() if f.path == ("rl", "enabled")
        )

    def show_errors(self, errors: list[str]) -> None:
        self.query_one("#errors", Static).update(Text("\n".join(errors), style="red"))

    def action_save(self) -> None:
        """Save, also when the experiment is not complete yet; only a bad name or value stops it."""
        try:
            experiment = self.collect()
        except FormError as error:
            self.show_errors([str(error)])
            return
        original = self.experiment.get("name")
        others = self.other_names - ({original} if original else set())
        blocking = store.name_problems(experiment["name"], others)
        if blocking:
            self.show_errors(blocking)
            return
        problems = store.validate_experiment(experiment, others) + store.experiment_notes(experiment)
        self._write(experiment, problems)

    def _write(self, experiment: dict[str, Any], problems: list[str]) -> None:
        try:
            if self.definition is None:
                path = store.DEFINITIONS_ROOT / self.research / f"{store.file_stem(experiment['name'])}.py"
                definition = store.DefinitionFile(path=path, order=1000, experiments=[experiment])
                store.write_definition(definition, create=True)
                kept = True
            else:
                definition = self.definition
                if self.index is None:
                    definition.experiments.append(experiment)
                else:
                    definition.experiments[self.index] = experiment
                kept = store.write_definition(definition)
        except FileExistsError:
            self.show_errors([
                f"An experiment file {store.file_stem(experiment['name'])}.py already exists in set "
                f"{self.research}; choose another name."
            ])
            return
        except (OSError, ValueError) as error:
            self.show_errors([str(error)])
            return
        if problems:
            self.app.notify(
                f"Saved {experiment['name']}, not ready to run yet: " + "; ".join(problems[:3]),
                severity="warning",
                timeout=12,
            )
        else:
            self.app.notify(f"Saved {experiment['name']} to {definition.path.name}.")
        if not kept:
            self.app.notify(f"Comments in {definition.path.name} could not be kept.", severity="warning")
        self.dismiss(True)

    def action_cancel(self) -> None:
        if self._snapshot() == self._saved_state:
            self.dismiss(False)
            return
        self.app.push_screen(
            ConfirmScreen("Discard unsaved changes?"),
            lambda yes: self.dismiss(False) if yes else None,
        )


def _summary_row(experiment: dict[str, Any]) -> tuple[str, str, str, str]:
    """Scheduler, power manager, workload and platform of an experiment, for a table row."""
    try:
        cfg = merge_overrides(BASE_CONFIG, experiment.get("overrides", {}))
        scheduler, power = resolve_run_components(cfg["run"])
        return scheduler["name"], power["name"], cfg["paths"]["workload"] or "-", cfg["paths"]["platform"] or "-"
    except Exception:  # a hand-edited file can hold anything
        return "?", "?", "?", "?"


def _status(experiment: dict[str, Any]) -> Text:
    if not experiment.get("enabled", True):
        return Text("disabled", style="dim")
    problems = store.quick_problems(experiment)
    if problems:
        return Text(f"incomplete: {problems[0]}", style="yellow")
    return Text("ready", style="green")


def _try(app, action: Callable[[], Any], what: str) -> bool:
    """Run a file action; show its error instead of closing the dashboard."""
    try:
        action()
    except (OSError, ValueError) as error:
        app.notify(f"Cannot {what}: {error}", severity="error")
        return False
    return True


class ExperimentsListScreen(Screen):
    """The experiments inside one hand-written definition file that holds several."""

    BINDINGS = [
        Binding("n", "new", "New"),
        Binding("enter", "edit", "Edit", priority=True),
        Binding("u", "duplicate", "Duplicate"),
        Binding("x", "delete", "Delete"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, path_key: str) -> None:
        super().__init__()
        self.path_key = path_key

    def definition(self) -> store.DefinitionFile:
        research, stem = self.path_key.split("/")
        return store.read_definition(store.DEFINITIONS_ROOT / research / f"{stem}.py")  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        yield DataTable(id="experiments", cursor_type="row", zebra_stripes=True)
        yield Static(id="empty", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Name", "Scheduler", "Power manager", "Workload", "Platform", "Status")
        self.refresh_table()
        table.focus()

    def refresh_table(self) -> None:
        definition = self.definition()
        status = "" if definition.editable else f"   ·   read-only: {definition.reason}"
        self.query_one("#title", Static).update(f"{definition.path.name}{status}")
        table = self.query_one(DataTable)
        table.clear()
        for experiment in definition.experiments:
            table.add_row(str(experiment.get("name", "?")), *_summary_row(experiment), _status(experiment))
        self.query_one("#empty", Static).update(
            "" if definition.experiments else "No experiments in this file yet. Press n to create one."
        )

    def _editable(self) -> store.DefinitionFile | None:
        definition = self.definition()
        if not definition.editable:
            self.notify(f"Read-only: {definition.reason}", severity="warning")
            return None
        return definition

    def _other_names(self) -> set[str]:
        return store.experiment_names(self.path_key.split("/")[0])

    def _open_form(self, definition, index, experiment) -> None:
        def done(saved: bool | None) -> None:
            if saved:
                self.app.reload_definitions()  # type: ignore[attr-defined]
            self.refresh_table()

        self.app.push_screen(
            ExperimentFormScreen(definition, index, experiment, self._other_names()), done
        )

    def action_new(self) -> None:
        definition = self._editable()
        if definition is not None:
            self._open_form(definition, None, store.new_experiment("", "", ""))

    def action_edit(self) -> None:
        definition = self._editable()
        table = self.query_one(DataTable)
        if definition is None or not definition.experiments:
            return
        index = table.cursor_row
        self._open_form(definition, index, definition.experiments[index])

    def action_duplicate(self) -> None:
        definition = self._editable()
        if definition is None or not definition.experiments:
            return
        copy_ = copy.deepcopy(definition.experiments[self.query_one(DataTable).cursor_row])
        copy_["name"] = store.unused_name(f"{copy_['name']}-copy", self._other_names())
        copy_["display_name"] = copy_["name"]
        definition.experiments.append(copy_)
        if _try(self.app, lambda: store.write_definition(definition), "duplicate"):
            self.app.reload_definitions()  # type: ignore[attr-defined]
        self.refresh_table()

    def action_delete(self) -> None:
        definition = self._editable()
        table = self.query_one(DataTable)
        if definition is None or not definition.experiments:
            return
        index = table.cursor_row
        name = definition.experiments[index]["name"]

        def confirmed(yes: bool | None) -> None:
            if yes:
                definition.experiments.pop(index)
                if _try(self.app, lambda: store.write_definition(definition), "delete"):
                    self.app.reload_definitions()  # type: ignore[attr-defined]
                self.refresh_table()

        self.app.push_screen(ConfirmScreen(f"Delete experiment {name}?"), confirmed)


class ExperimentsScreen(Screen):
    """The experiments of the current set: one file each; Enter edits, n makes a new one."""

    BINDINGS = [
        Binding("n", "new", "New experiment"),
        Binding("enter", "open", "Edit", priority=True),
        Binding("u", "duplicate", "Duplicate"),
        Binding("x", "delete", "Delete"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        yield DataTable(id="definitions-table", cursor_type="row", zebra_stripes=True)
        yield Static(id="empty", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Experiment", "Scheduler", "Power manager", "Workload", "Platform", "Status")
        self.refresh_table()
        table.focus()

    @property
    def research(self) -> str:
        return self.app.research  # type: ignore[attr-defined]

    def refresh_table(self) -> None:
        self.query_one("#title", Static).update(
            f"Experiments   ·   set {self.research}   ·   n new   Enter edit   u duplicate   x delete"
        )
        table = self.query_one(DataTable)
        row = table.cursor_row
        table.clear()
        self.files = store.list_definition_files(self.research)
        for definition in self.files:
            if len(definition.experiments) == 1:
                experiment = definition.experiments[0]
                status = _status(experiment) if definition.editable else Text(f"read-only: {definition.reason}", style="dim")
                table.add_row(str(experiment.get("name", definition.path.stem)), *_summary_row(experiment), status)
            else:
                count = len(definition.experiments)
                note = f"{count} experiments: Enter lists them" if count else "empty file: Enter adds one"
                if not definition.editable:
                    note = f"read-only: {definition.reason}"
                table.add_row(definition.path.name, "", "", "", "", Text(note, style="dim"))
        if table.row_count:
            table.move_cursor(row=min(row, table.row_count - 1))
        self.query_one("#empty", Static).update(
            "" if self.files else f"No experiments in set {self.research} yet. Press n to make one."
        )

    def _current(self) -> store.DefinitionFile | None:
        table = self.query_one(DataTable)
        return self.files[table.cursor_row] if self.files else None

    def _form_closed(self, saved: bool | None) -> None:
        if saved:
            self.app.reload_definitions()  # type: ignore[attr-defined]
        self.refresh_table()

    def action_new(self) -> None:
        self.app.push_screen(
            ExperimentFormScreen(
                None, None, store.new_experiment("", "", ""), store.experiment_names(self.research),
                research=self.research,
            ),
            self._form_closed,
        )

    def action_open(self) -> None:
        definition = self._current()
        if definition is None:
            return
        if len(definition.experiments) == 1 and definition.editable:
            self.app.push_screen(
                ExperimentFormScreen(
                    definition, 0, definition.experiments[0], store.experiment_names(self.research)
                ),
                self._form_closed,
            )
        elif not definition.experiments and definition.editable:
            self.app.push_screen(
                ExperimentFormScreen(
                    definition, None, store.new_experiment("", "", ""), store.experiment_names(self.research)
                ),
                self._form_closed,
            )
        else:
            self.app.push_screen(ExperimentsListScreen(definition.key), lambda _: self.refresh_table())

    def action_duplicate(self) -> None:
        definition = self._current()
        if definition is None or not definition.experiments:
            return
        if not definition.editable:
            self.notify(f"Read-only: {definition.reason}", severity="warning")
            return

        def named(name: str | None) -> None:
            if not name:
                return
            problems = store.name_problems(name, store.experiment_names(self.research))
            if problems:
                self.notify(problems[0], severity="error")
                return
            if _try(self.app, lambda: store.duplicate_experiment(definition, name), "duplicate"):
                self.app.reload_definitions()  # type: ignore[attr-defined]
                self.refresh_table()

        first = str(definition.experiments[0].get("name", definition.path.stem))
        suggestion = store.unused_name(f"{first}-copy", store.experiment_names(self.research))
        self.app.push_screen(NameScreen("Name of the copy", suggestion), named)

    def action_delete(self) -> None:
        definition = self._current()
        if definition is None:
            return
        if not definition.editable:
            self.notify(f"Read-only: {definition.reason}", severity="warning")
            return

        def confirmed(yes: bool | None) -> None:
            if not yes:
                return
            moved: list[Path] = []
            if _try(self.app, lambda: moved.append(store.delete_definition(definition)), "delete"):
                where = moved[0].relative_to(store.PROJECT_ROOT) if moved[0].is_relative_to(store.PROJECT_ROOT) else moved[0]
                self.app.notify(f"Moved to {where} (you can restore it from there)")
                self.app.reload_definitions()  # type: ignore[attr-defined]
            self.refresh_table()

        what = (
            str(definition.experiments[0].get("name")) if len(definition.experiments) == 1
            else f"{definition.path.name} and its {len(definition.experiments)} experiment(s)"
        )
        self.app.push_screen(ConfirmScreen(f"Delete {what}? It is moved to .spars/trash."), confirmed)
