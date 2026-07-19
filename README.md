# SPARS Simulator

SPARS is an event-driven simulator for high-performance computing (HPC) scheduling, resource allocation, node power management, and reinforcement-learning experiments.

SPARS can:

- generate synthetic workloads and platform descriptions;
- run FCFS and EASY-backfilling;
- model active, sleeping, switching-on, and switching-off nodes;
- train or evaluate an RL power-management agent; and
- produce job, node, energy, waiting-time, metric, and Gantt outputs.

## Requirements

- Python 3.11
- Linux or Windows
- Jupyter Notebook for `WorkloadGenerator.ipynb` and `PlatformGenerator.ipynb`
- PyTorch, installed separately after the other requirements

Run commands from the project root. `RunAll.py` expects the virtual environment to be named `SPARS-venv` and located in the project directory.

## Installation

### Linux

```bash
python3.11 -m venv SPARS-venv
source SPARS-venv/bin/activate
python -m pip install -r requirements-linux.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv SPARS-venv
.\SPARS-venv\Scripts\Activate.ps1
python -m pip install -r requirements-windows.txt
```

### Install PyTorch

PyTorch is not included in `requirements-linux.txt` or `requirements-windows.txt` because CPU and GPU systems require different PyTorch builds.

#### CPU installation

Use the CPU build when GPU acceleration is not needed or when the machine does not have a supported GPU. This is sufficient for normal non-RL simulations and can also run RL experiments on the CPU.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### GPU installation

Use the installer provided by PyTorch for the machine's supported compute platform:

https://pytorch.org/get-started/locally/

Select the operating system, **Pip**, **Python**, and the appropriate compute platform, then run the generated command inside `SPARS-venv`.

For an NVIDIA GPU, `nvidia-smi` can be used to check the GPU and driver information before choosing one of the CUDA builds offered by the PyTorch installer:

```bash
nvidia-smi
```

For another GPU, AMD systems require a supported ROCm build, macOS uses the macOS PyTorch build, and systems without a supported GPU should use the CPU installation above.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA/ROCm available:', torch.cuda.is_available()); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('XPU available:', hasattr(torch, 'xpu') and torch.xpu.is_available())"
```

## Main workflow

A SPARS run uses three inputs:

1. **Workload:** the jobs, including submission time, requested nodes, requested wall time, and actual runtime.
2. **Platform:** the simulated nodes, including power values, transition times, and DVFS profiles.
3. **Simulator configuration:** the workload and platform paths, scheduler, run behavior, RL settings, logging, experiments, and output settings.

```text
WorkloadGenerator.ipynb  -> workload JSON
PlatformGenerator.ipynb  -> platform JSON
RunSPARSOuter.py          -> experiment configurations
RunAll.py                 -> execute the configured experiments
PlotMetrics.py            -> metrics comparison plots
PlotGantt.py              -> Gantt plots
```

## 1. Generate a workload

Open `WorkloadGenerator.ipynb` and edit `GeneratorConfig`.

```python
CONFIG = GeneratorConfig(
    num_jobs=20,
    nb_res=8,
    target_utilization=0.40,
    mean_runtime=300.0,
    resource_geometric_p=0.35,
    reqtime_bias=1.20,
    seed=4,
)

OUTPUT_PATH = Path("workloads/demo-workload.json")
PARAMETERS_PATH = Path("workloads/demo-workload-parameters.json")
```

| Parameter              | Meaning                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `num_jobs`             | Number of jobs to generate.                                                                                             |
| `nb_res`               | Maximum resource capacity represented by the workload. It must not exceed the number of nodes in the selected platform. |
| `target_utilization`   | Intended average workload pressure. Larger values create shorter arrival gaps.                                          |
| `mean_runtime`         | Mean actual job runtime in simulator seconds.                                                                           |
| `resource_geometric_p` | Controls the generated job sizes. Larger values produce smaller jobs on average.                                        |
| `reqtime_bias`         | Multiplies `mean_runtime` to set the mean requested wall time.                                                          |
| `seed`                 | NumPy random seed used to reproduce the generated workload.                                                             |
| `user_id`              | User identifier assigned to the generated jobs.                                                                         |

The generator writes:

- the workload JSON; and
- a parameter JSON containing the values used to generate it.

Each generated job contains:

```json
{
  "job_id": 1,
  "res": 2,
  "subtime": 10.5,
  "reqtime": 400.0,
  "runtime": 280.0,
  "user_id": 0
}
```

`subtime`, `reqtime`, and `runtime` use simulator seconds.

## 2. Generate a platform

Open `PlatformGenerator.ipynb` and edit the `generate_machine` call.

```python
machines = [
    generate_machine(
        machine_ids=range(0, 8),
        node_name="demo-node",
        base_compute_power=150.0,
        base_idle_power=90.0,
        switching_on_power=120.0,
        switching_off_power=30.0,
        sleeping_power=10.0,
        switching_on_time=20.0,
        switching_off_time=10.0,
        switching_on_std=0.0,
        switching_off_std=0.0,
        compute_speed_variation=1.0,
    ),
]

platform = {"machines": machines}
output_path = Path("platforms/demo-platform.json")
```

| Parameter                 | Meaning                                                 |
| ------------------------- | ------------------------------------------------------- |
| `machine_ids`             | Node IDs represented by this machine type.              |
| `node_name`               | Name of the node type.                                  |
| `base_compute_power`      | Power used while computing in the base DVFS mode.       |
| `base_idle_power`         | Power used while active and idle in the base DVFS mode. |
| `switching_on_power`      | Power used while switching on.                          |
| `switching_off_power`     | Power used while switching off.                         |
| `sleeping_power`          | Power used while sleeping.                              |
| `switching_on_time`       | Mean switch-on duration in simulator seconds.           |
| `switching_off_time`      | Mean switch-off duration in simulator seconds.          |
| `switching_on_std`        | Standard deviation of switch-on duration.               |
| `switching_off_std`       | Standard deviation of switch-off duration.              |
| `compute_speed_variation` | Multiplier applied to the node's compute speed.         |

The generated platform contains five DVFS profiles. Each profile defines compute power, idle power, and compute speed.

## 3. Configure the simulation

`RunSPARSOuter.py` contains the shared simulator configuration, the experiment list, and the automatic plotting controls.

The example below configures one small non-RL simulation. The explanation follows the same order as the configuration.

```python
BASE_CONFIG = {
    "paths": {
        "workload": "workloads/demo-workload.json",
        "platform": "platforms/demo-platform.json",
        "output": "",
    },
    "run": {
        "algorithm": "fcfs_psus",
        "algo_config": {
            "timeout": None,
        },
        "overrun_policy": "continue",
        "start_time": 0,
        "force_wakeup": False,
    },
    "rl": {
        "enabled": False,
        "learn": False,
        "type": "discrete",
        "dt": 300,
        "device": "cpu",
        "epochs": 1,
        "checkpoint": None,
        "episode_batch_size": 32,
        "agents": {},
        "assign": "Budiarjo",
    },
    "gym": {
        "feature_extractor": "Budiarjo",
        "translator": "Budiarjo",
        "reward": {
            "name": "Budiarjo",
            "params": {
                "weight1": 0.5,
                "weight2": 0.5,
                "device": "cpu",
            },
        },
        "learner": "Budiarjo",
    },
    "logging": {
        "level": "INFO",
        "file": "results/simulation.log",
    },
}
```

The repository's original `BASE_CONFIG` already contains the full Budiarjo agent definition. Keep that definition when enabling RL.

### 3.1 `paths`: select the input files

```python
"paths": {
    "workload": "workloads/demo-workload.json",
    "platform": "platforms/demo-platform.json",
    "output": "",
},
```

- `workload` selects the workload JSON.
- `platform` selects the platform JSON.
- `output` is left empty because `RunSPARSOuter.py` creates the result path for each experiment.

### 3.2 `run`: select the scheduler and simulator behavior

After selecting the input files, the `run` section defines how SPARS executes them.

#### `algorithm`

The public scheduler names are:

```text
fcfs
fcfs_baseline
fcfs_psus
fcfs_psas
fcfs_baseline_psas
easy
easy_baseline
easy_psus
easy_psas
easy_baseline_psas
```

The main abbreviations are:

- **FCFS:** first-come, first-served.
- **EASY:** FCFS with EASY backfilling.
- **PSUS:** power-state-unaware scheduling.
- **PSAS:** power-state-aware scheduling.
- **baseline:** comparison implementation.
- **oracle:** comparison method that uses information unavailable to an online scheduler.

#### `algo_config.timeout`

Most schedulers require:

```python
"algo_config": {
    "timeout": None,
}
```

`timeout` is the number of simulator seconds an active-idle node may remain unused before a switch-off request. `None` disables the timeout. For example, `7200` means two simulator hours.

```python
"algo_config": {}
```

#### `overrun_policy`

`overrun_policy` controls a job whose actual runtime exceeds its requested wall time.

- `"continue"`: let the job finish at its actual completion time.
- `"terminate"`: stop the job when its requested wall time expires.

A job whose actual runtime is shorter than its request finishes at its actual completion time under either policy.

#### `start_time`

`start_time` sets the simulation time origin. Workload submission times are added to it.

Accepted values are:

- `0` for a relative timeline;
- a Unix timestamp;
- `"now"`; or
- `"YYYY-MM-DD HH:MM:SS"`.

#### `force_wakeup`

`force_wakeup` is a simulator fallback for a stalled run. When enabled under its trigger conditions, SPARS can wake sleeping nodes while unfinished jobs remain and no useful progress event is available.

The current implementation calculates the target from the total resources requested by jobs in the waiting queue, capped by the platform size. It is not part of the selected scheduler. Keep it `False` unless the experiment is specifically testing recovery behavior.

### 3.3 `rl` and `gym`: configure reinforcement learning

When `rl.enabled` is `False`, SPARS uses the selected heuristic scheduler and ignores the training settings.

When `rl.enabled` is `True`:

| Field                | Meaning                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learn`              | Train the agent when `True`; evaluate a checkpoint when `False`.                                                                                        |
| `type`               | `"discrete"` uses fixed decision intervals. `"continuous"` requests decisions at selected events and is experimental.                                   |
| `dt`                 | Decision interval in simulator seconds for discrete mode.                                                                                               |
| `device`             | `"cpu"`, `"cuda"`, or `"auto"`.                                                                                                                         |
| `epochs`             | Number of complete training runs. In the current runner, one epoch creates one fresh simulator and runs one full workload, so one epoch is one episode. |
| `checkpoint`         | Saved model path, or `None`.                                                                                                                            |
| `episode_batch_size` | Maximum number of RL transitions collected before one learning update. It is not a number of episodes; the current name is misleading.                  |
| `agents`             | Agent class, parameters, optimizer class, and optimizer parameters.                                                                                     |
| `assign`             | Agent entry selected from `agents`.                                                                                                                     |

The code calls each transition batch a rollout internally, but it is only a partial batch from the current episode. In the documentation and video, call it a **transition batch** to avoid mixing it with the complete episode.

The `gym` section selects the four components used during each RL decision:

1. `feature_extractor`: simulator state to observation.
2. `translator`: agent action to simulator events.
3. `reward`: reward calculation.
4. `learner`: model update.

### 3.4 `logging`: select the log level

- `INFO` prints normal progress information.
- `TRACE` prints detailed event-level information.

The `logging.file` value is saved in the configuration, but the current logger writes to the console.

### 3.5 `EXPERIMENTS`: select what will actually run

`BASE_CONFIG` provides the shared values. Each enabled entry in `EXPERIMENTS` copies that configuration and changes only the values listed in `overrides`.

```python
EXPERIMENTS = [
    {
        "name": "Demo",
        "enabled": True,
        "overrides": {
            "run": {
                "algorithm": "fcfs_psus",
                "algo_config": {
                    "timeout": None,
                },
                "force_wakeup": False,
            },
            "rl": {
                "enabled": False,
                "learn": False,
                "device": "cpu",
            },
        },
    }
]
```

A sweep is a list of values to test. This example runs the same workload with three schedulers:

```python
"sweep": {
    "run.algorithm": [
        "fcfs_psus",
        "easy_psus",
    ],
}
```

A curriculum is only for RL training. SPARS runs the first stage, takes its best checkpoint, and continues training in the next stage.

### 3.6 Automatic plotting controls

At the end of `RunSPARSOuter.py`, set the plotting behavior before starting the simulation:

```python
AUTO_PLOT = True
GENERATE_GANTT_PLOTS = False
GENERATE_METRICS_PLOT = False
```

- `AUTO_PLOT` enables the post-run plotting stage.
- `GENERATE_GANTT_PLOTS` runs `PlotGantt.py` after the experiments finish.
- `GENERATE_METRICS_PLOT` runs `PlotMetrics.py` after the experiments finish.

The plotting scripts can also be run manually after the simulation, as shown in Step 6.

## 4. Run SPARS

Use `RunAll.py` from the project root:

```bash
python RunAll.py 1
```

The number controls both the CPU-core limit and the maximum number of concurrent experiment threads. For example:

```bash
python RunAll.py 4
```

allows up to four non-RL experiment threads. RL experiments are still run one at a time.

`RunAll.py` starts `RunSPARSOuter.py`, which creates the concrete YAML configurations and runs them through `RunSPARSConfig.py`.

One generated YAML configuration can also be run directly:

```bash
python RunSPARSConfig.py path/to/config.yaml
```

## 5. Inspect the results

The result structure is:

```text
results/<UID>/<experiment>/<platform>/<workload>/<algorithm-and-parameters>/
```

A non-RL result directory contains files such as:

- `simulator_config_used.yaml`
- `raw_job_log.csv`
- `unfinished_jobs_log.csv`
- `node_log.csv`
- `waiting_time_log.csv`
- `energy_log.csv`
- `metrics.csv`
- `state_switch.csv`
- `runtime_seconds.txt`
- `profile.csv`

RL training also creates epoch directories, step logs, checkpoints, Gym metadata, and a `best_epoch_<n>` directory.

## 6. Visualize the results

Both plotting scripts take the path to one result UID directory.

```bash
python PlotMetrics.py <path_to_uid>
python PlotGantt.py <path_to_uid>
```

For example:

```bash
python PlotMetrics.py results/2026-07-19_12-00-00
python PlotGantt.py results/2026-07-19_12-00-00
```

`PlotMetrics.py` reads `metrics.csv` from the runs under that UID and creates comparison plots inside each workload's `metrics_comparison/` directory.

`PlotGantt.py` reads `node_log.csv` and creates Gantt images inside each run's `plots/` directory.

## Event-driven execution

At each simulation timestamp, SPARS:

1. processes all events at that timestamp in priority order;
2. processes any additional same-time events created by those events;
3. records the new state;
4. calls the scheduler;
5. inserts the scheduler's events; and
6. advances after no same-time work remains.

Events include job arrivals, execution starts and finishes, switch requests, transition completions, and RL decision calls.

## Adding a scheduler

1. Create a scheduler class under `SPARS/Simulator/Algo/`.
2. Give its constructor explicit required parameters.
3. Add the class to `ALGO_MAP` in `SPARS/Simulator/Scheduler.py`.
4. Add each configurable constructor parameter to `run.algo_config`.

The scheduler wrapper validates `algo_config` against the constructor signature.

## Adding RL components

The main extension directories are:

- `SPARS/Gym/features/`
- `SPARS/Gym/translators/`
- `SPARS/Gym/rewards/`
- `SPARS/Gym/learners/`
- `RL_Agent/`

Register a short name in `SPARS/Gym/config.py`, or provide a dotted import path in the configuration.

Continuous RL uses variable-duration decision intervals and is currently experimental.

## Reproducibility

Keep these files and values with an experiment:

- the generated workload JSON and its parameter JSON, including the workload seed;
- the generated platform JSON and the platform-generator values used to create it; and
- `simulator_config_used.yaml` from the result directory.

The current platform-transition sampler uses seed `42` by default in `PlatformControl`.

## License

SPARS is released under the MIT License. See `LICENSE` for details.
