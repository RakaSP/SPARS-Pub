# SPARS Simulator

SPARS is an event-driven simulator for job scheduling and node power management in high-performance computing (HPC) systems.

SPARS can:

- convert real SWF traces or generate synthetic workloads, and generate platforms with one or more node types;
- run schedulers such as FCFS, EASY backfilling and SNF;
- model active, sleeping, switching-on and switching-off nodes, and the power managers that switch them;
- train or evaluate a reinforcement-learning (RL) power manager; and
- produce job, node, energy and waiting-time logs, metrics and Gantt charts.

## Requirements

- Python 3.10 or newer (3.11 or 3.12 recommended)
- Linux, macOS or Windows

## Installation

```bash
git clone https://github.com/RakaSP/SPARS-Pub.git
cd SPARS-Pub
python install.py
```

`install.py` creates the virtual environment `SPARS-venv` and installs SPARS in it. It asks whether to install PyTorch, which only the RL power manager needs; the default is no, and `python install.py --torch` adds it later. With PyTorch it picks the build that fits the machine (NVIDIA, AMD, Intel or CPU). It ends with `Setup complete.` Activate the environment in every new terminal:

```bash
source SPARS-venv/bin/activate        # Linux, macOS
.\SPARS-venv\Scripts\Activate.ps1     # Windows PowerShell
```

`setup_guide.pdf` walks through every step, from installing Python to starting the dashboard.

SPARS can also be installed into any environment you already use: `pip install -e .` in the repository folder. That does not install PyTorch; add it with `pip install torch`, or `pip install -e ".[rl]"`.

## Start

```bash
spars                 # or: python -m spars.cli
```

This opens the dashboard in the terminal. `?` shows every key.

## Main workflow

A SPARS run uses three inputs:

1. **Workload:** the jobs, with submission time, requested nodes, requested time and actual runtime.
2. **Platform:** the nodes, with their power in each state, transition times and DVFS profiles.
3. **Experiment:** which workload, platform, scheduler and power manager to run.

Experiments are grouped in **sets**. A set is a folder `src/spars/experiments/<Set>/`, and every experiment the dashboard makes is one file in it.

```text
workload JSON + platform JSON -> experiment (in a set) -> dashboard: run -> results/<Set>-NNNNNN/ -> metrics and Gantt plots
```

## 1. Get a workload

Workloads are JSON files in `data/workloads/json/`. A workload is one object:

```json
{
  "nb_res": 8,
  "jobs": [
    {
      "job_id": 1,
      "res": 2,
      "subtime": 10.5,
      "reqtime": 400.0,
      "runtime": 280.0,
      "user_id": 0
    }
  ]
}
```

`res` is the number of nodes requested, `subtime`, `reqtime` and `runtime` are simulator seconds, and `nb_res` is the number of nodes the workload is made for.

- **Convert a real trace:** the SWF traces in `data/workloads/cshuji/` come from the Parallel Workloads Archive. On the Workloads screen (`w`), pick a trace from the list (type part of its name, or press Enter), choose the first and last job, and press Ctrl+S. The list shows how many nodes each trace needs. Jobs 2001 to 2500 of a trace become `<trace>-2001-2500.json`, with times counted from just before job 2001. Without the dashboard: `python -m spars.core.workloads.swf SOURCE.swf OUTPUT.json [LAST_JOB] [FIRST_JOB]`.
- **Generate:** `g`, from a target utilization. A utilization above 1 is allowed; the queue then keeps growing.

## 2. Get a platform

Platforms are JSON files in `data/platforms/`. Generate one with `p`.

- A platform has one or more **node types** (Add node type). Each type is a group of identical nodes with its own power in each state, switching times and compute speed.
- Each node type has one or more **DVFS profiles**, factors that scale its compute power, idle power and speed.
- Each node has four states: `active`, `sleeping`, `switching_on` and `switching_off`. A node runs jobs only while `active`.
- **Size:** a platform needs at least as many nodes as the workload's `nb_res`. The Workloads screen lists it. For the cshuji traces: DAS2-fs0 needs 144, the other DAS2 traces 64, SDSC-BLUE 1152.

## 3. Make an experiment

Press `e` for the experiments of the current set. The first time, it asks for a set name. Then:

1. Press `n`, give the experiment a name, and choose its workload, platform, scheduler and power manager.
2. Press Ctrl+S to save it as `src/spars/experiments/<Set>/<name>.py`.

An experiment can be saved before it is complete, for example before its platform exists. The list then marks it incomplete, and it cannot run until it is fixed. The New workload and New platform buttons open the generators without leaving the form; the new file is filled in when you come back.

Every field works the same way:

- Up/Down or Tab move between fields, and typing replaces the value.
- Pick-list fields open when you type (filtered) or press Enter (everything). In the list, Up/Down choose and Enter takes the entry.
- The line under the form explains the field you are on.

An experiment file is plain Python, so it can also be written by hand:

```python
EXPERIMENT_SET_ORDER = 1

EXPERIMENTS = [
    {
        "name": "FCFS",
        "enabled": True,
        "overrides": {
            "paths": {"workload": "workloads/json/DAS2-fs1-1-3000.json", "platform": "platforms/my-platform.json"},
            "run": {
                "scheduler": {"name": "FCFSPSUS", "config": {}},
                "power_management": {"name": "PSUSTimeoutPowerManager", "config": {"timeout": 1800}},
            },
        },
    },
]
```

`overrides` changes only the listed values of the shared base configuration (`src/spars/core/experiments/defaults.py`). A `sweep`, for example `{"run.power_management.config.timeout": [1800, 3600]}`, runs one experiment per value. A hand-written file may hold several experiments; the dashboard lists them when you open it.

### Schedulers and power managers

A **scheduler** decides which job runs on which nodes. A **power manager** decides which nodes are switched on or off. Both are named by their class name.

| Scheduler                                     | What it does                                                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FCFSPSUS`, `EASYPSUS`                        | First come first served, or EASY backfilling; power-state unaware: they wake sleeping nodes when a job needs them. Pair with `PSUSTimeoutPowerManager`. |
| `FCFSPSAS`, `EASYPSAS`                        | The same orders, power-state aware: they plan with the node states. Pair with `TimeoutPowerManager`.                                                    |
| `SNFPSAS`                                     | Smallest node request first, power-state aware.                                                                                                         |
| `SNFICON`                                     | SNF for the ICON spare-capacity policy. Pair with `ICONPowerManager`.                                                                                   |
| `SNFOraclePSAS`                               | SNF over the complete workload, known in advance. Pair with `OraclePowerManager`.                                                                       |
| `FCFSBaseline`, `EASYBaseline`, `SNFBaseline` | Use only nodes that are on and never switch any. Pair with `NoPowerManager`.                                                                            |

| Power manager                                    | What it does                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `TimeoutPowerManager`, `PSUSTimeoutPowerManager` | Switch a node off after it has been idle for `timeout` seconds.    |
| `ICONPowerManager`                               | ICON: keeps spare nodes ready from a workload model.               |
| `OraclePowerManager`                             | Knows the whole workload. It is an upper bound, not a real policy. |
| `RLPowerManager`                                 | The RL agent (needs PyTorch).                                      |
| `NoPowerManager`                                 | Never switches a node.                                             |

A class you write under `src/spars/policies/` appears in these lists after Ctrl+R. A `"module:Class"` path works too.

## 4. Run SPARS

On the main screen:

1. Space selects experiments.
2. Enter goes to the options: plots, log level, log file, and Simulation and Support cores.
3. Enter again starts the batch.

An experiment that would fail (no platform yet, an unknown scheduler or setting) is left out: the dashboard lists it and asks whether to run the others, and the log lists it again when the batch ends. `spars --set` does the same without asking.

Each running simulation uses one Simulation core. The dashboard and plotting use the Support cores. Other keys:

- `l` shows the log;
- `r` shows the results;
- `c` cancels a batch;
- `s` switches the set or makes a new one;
- Ctrl+R restarts the dashboard on changed code;
- `?` shows the help.

Without a terminal (a server, a script, CI), run sets directly:

```bash
spars --set NAME             # one set; --all for every set
spars --set NAME --cores 8   # cores and plots default to the dashboard's last choice
```

In Git Bash on Windows, open the dashboard with `winpty python -m spars.cli`, or use PowerShell.

## 5. Inspect the results

Each batch writes to `results/<Set>-NNNNNN/`. It contains:

- one folder per run, with the configuration used, per-job and per-node CSV logs and `metrics.csv` (mean waiting time, wasted energy, energy per state);
- the batch's `simulation.log`.

The Results screen (`r`) shows the mean waiting time and wasted energy of every run.

A run in which jobs wait and nothing has started or finished for a long time (`run.stall_timeout`, default 100000 simulated seconds) is stopped and marked STALLED. `stalled.json` then says how many jobs finished, waited, ran and were never submitted, and why.

## 6. Plot the results

Metrics and Gantt plots are made after a batch when they are switched on in the options. The Results screen makes them for a finished run with `p`, or:

```bash
spars --plot-only results/<Set>-NNNNNN --metrics --gantt
```

## How one simulation step works

The simulator jumps from event to event: a job arrives, a job ends, a node finishes switching. After processing the events of one time, the Planner asks whether a new plan is needed. The general triggers, the scheduler and the power manager can each say yes. If one does, the Planner calls `scheduler.schedule()` and then `power_manager.decide(now, decision)`. Both answer with events (`execution_start`, `switch_on`, ...) that the simulator queues and executes at their time.

`spars-overview.html` is an interactive diagram of SPARS's architecture — the experiment, core and policies. `spars-flow.html` diagrams one run, from start to results, including the step above. Open either in a browser and click a component to see its code.

## Adding a scheduler or power manager

Put a class anywhere under `src/spars/policies/scheduler/` or `src/spars/policies/power_management/` and name it in an experiment. It is found without registration. The simplest start is to extend an existing policy, for example `FCFSPSUS` in `policies/scheduler/fcfs/psus.py`.

- A scheduler takes `(machines, jobs_manager, start_time, config)` and implements `set_time(t)` and `schedule()`.
- A power manager takes `(machines, start_time, config)` and implements `prepare(t)`, `decide(t, decision)` and `remaining_idle_timeout(node_id, t)`. `policies/power_management/timeout/policy.py` is the reference.
- A class attribute `DEFAULTS = {"setting": value}` lists a policy's settings, and the experiment form shows them as fields.

Policies read the requested time of a job, never its actual runtime; only the Oracle may.

## Structure

```
src/spars/
  cli.py             the spars command: dashboard, or runs without a terminal
  core/              the simulator and its tooling; no need to read it to write a policy
    simulator/       event loop, JobsManager, PlatformControl, Machine, Monitor
    planner.py       when to plan; calls the scheduler, then the power manager
    trigger.py       the general plan triggers
    experiments/     finds, expands and runs the experiments; defaults.py holds the base configuration
    runner/          one run: run_with_config, CPU cores, training and evaluation
    gym/             the RL environment
    dashboard/       the terminal dashboard
    analysis/        metrics, Gantt charts, comparison plots
    workloads/       workload generator, SWF converter
    platforms/       platform generator
    output.py        writes the logs and metrics.csv
  policies/          schedulers and power managers, including the RL agents
    scheduler/       fcfs/, easy/, snf/ and their shared base/
    power_management/  timeout/, icon.py, oracle.py, none.py, rl/
    shared/          helpers both kinds use
    discovery.py     finds a policy class by its name
  experiments/       the experiment sets, one folder each
data/                platforms, workloads, SWF traces
tests/               pytest suite: python -m pytest tests
spars-overview.html  architecture: experiment, core and policies
spars-flow.html      one run, from start to results
```

## Data sources

The SWF traces in `data/workloads/cshuji/` come from the Parallel Workloads Archive (https://www.cs.huji.ac.il/labs/parallel/workload/).

## License

SPARS is released under the MIT License. Please credit the project when you use or modify it in your own work.
