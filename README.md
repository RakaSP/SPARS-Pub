# SPARS Simulator 
SPARS is a lightweight simulator for studying **job scheduling** and **resource allocation** in High-Performance Computing (HPC) systems.  
It allows you to **generate workloads**, **define platforms**, **run scheduling simulations** with different algorithms, and **visualize results**.

## 📥 Installation

### Prerequisites

- Python 3.11.13 or newer
- Recommended: create a virtual environment to isolate dependencies

### Clone and Setup Environment

```bash
git clone https://github.com/RakaSP/SPARS-Pub.git
cd SPARS-Pub
python setup.py
```

## ⚙️ Workflow Overview

Workload Generation → Define synthetic job traces.

Platform Generation → Define HPC platform topology and machine states.

Scheduling Simulation → Run the simulation with different schedulers.

Results Visualization → Analyze and visualize scheduling outcomes.

Each stage is provided as a Jupyter Notebook for ease of experimentation.

## 📂 Workload Generation

**Notebook:** `WorkloadGenerator.ipynb`

This stage procedurally builds synthetic HPC job traces used as input to the simulator.

### Parameters

- Job arrival rate (`lambda_arrival`)  
  Controls the exponential inter-arrival process for jobs.

- Requested execution time distribution (`mu_execution`, `sigma_execution`)  
  Mean and standard deviation of the (normal) distribution used to draw requested runtimes.

- Runtime noise (`mu_noise`, `sigma_noise`)  
  Optional perturbation added to requested runtimes to obtain actual runtimes; can be disabled by setting `runtime_equals_reqtime=True`.

- Workload size (`num_jobs`)  
  Number of jobs to generate.

- Resource demand (`max_node`)  
  Maximum number of nodes in the system; also used as an upper bound when sampling job sizes (`res` per job).

- Minimum time unit (`min_time`)  
  Lower bound enforced on requested/actual runtimes to avoid zero-length jobs.

### Result

Given these parameters, the notebook generates a workload description as a JSON file  
(e.g., `workloads/generated_workload.json`) that defines the jobs to be scheduled.

### Workload format

The generated JSON file has a single top-level object with:

- **Number of resources** (`nb_res`)  
  Total number of compute nodes available in the platform (matches `max_node`).

- **Job list** (`jobs`)  
  An array of job entries, each with:
  - `job_id` – unique job identifier  
  - `res` – number of requested nodes  
  - `subtime` – job submission time  
  - `reqtime` – requested execution time  
  - `runtime` – actual execution time used by the simulator  
  - `profile` – identifier of the job’s resource-usage profile  
  - `user_id` – user identifier (e.g., for multi-user scenarios)

- **Profiles** (`profiles`)  
  Mapping from profile IDs (e.g., `"100"`) to a simple resource model:
  - `cpu` – CPU work volume  
  - `com` – communication volume  
  - `type` – job type (e.g., `"parallel_homogeneous"`)

Together, these fields describe the synthetic workload that the schedulers will execute on the generated platform.

## 🏗️ Platform Generation

**Notebook:** `PlatformGenerator.ipynb`

This stage procedurally builds the HPC platform model used by the simulator.

### Parameters

- Number of compute nodes (`num_nodes`)
- Baseline active power and compute speed per node (`base_power`, normalized speed)
- Power draw in non-active states (`switching_off_power`, `sleeping_power`, `switching_on_power`)
- State-transition latencies (`switching_off_time`, `switching_on_time`)
- DVFS configuration patterns (e.g., 2-, 3-, or 5-level DVFS profiles per node)

### Result

Given these parameters, the notebook generates a JSON platform description (e.g., `Generated_16.json`) that defines the environment where jobs are scheduled and energy usage is computed.

### Platform format

The generated JSON file has a single top-level object with a `machines` array.  
Each element of `machines` is a node entry with the following fields:

- **Node ID** (`id`)  
  Unique identifier of the machine.

- **DVFS profiles** (`dvfs_profiles`)  
  Mapping from DVFS modes to pairs of nominal power and normalized compute speed.

- **DVFS mode** (`dvfs_mode`)  
  Default DVFS profile used by the node (e.g., `"base"`).

- **Power-state model** (`states`)  
  Set of power states (`active`, `sleeping`, `switching_on`, `switching_off`), where each state defines:
  - its power/energy consumption rate,
  - its compute speed (possibly inherited from the DVFS profile), and
  - allowed transitions to other states with associated transition times, and whether the state can run jobs.

Together, these fields describe the platform on which jobs are executed and power management policies are applied.

### Result

Given these parameters, the notebook generates a workload description as a JSON file  
(e.g., `workloads/generated_500_8_ws.json`) that defines the jobs to be scheduled.

## 💻 Scheduling Simulation

Notebook: `Runner.ipynb`

Available schedulers: **FCFS** and **EASY**, with 3 variants:

1. **PSUS** → power-state-unaware scheduling.
2. **PSAS+IPM** → power-state-aware scheduling combined with an intelligent power manager that proactively switches nodes between active and low-power states.
3. **Auto On** → baseline always-on configuration where nodes remain powered up at all times (no power saving).


The simulation produces CSV logs containing job start/finish times, node allocations, and system events.

## 📊 Results Visualization

Notebook: create_ganttchart.ipynb

This stage transforms raw CSV logs into visual insights.
Outputs include:

- Node log → per-interval node state (DVFS mode, state, submission/start/finish times, allocated nodes, job ID, terminated flag)
- Job log → per-job lifecycle summary (job ID, event type, requested resources and nodes, submission/start/finish times, requested vs. actual runtime and finish time, terminated flag)
- Summary metrics → aggregate waiting-time, energy, and time-in-state statistics (e.g., total_waiting_time, mean_waiting_time, total_energy_waste, total_energy_consumption, total_time_all_states)
- Gantt chart → shows job execution timelines and node allocations

These visualizations help you evaluate the effectiveness of different scheduling policies and platform configurations.

## 🚀 Example End-to-End Run

1. Generate a workload → `WorkloadGenerator.ipynb`
2. Generate a platform → `PlatformGenerator.ipynb`
3. Run simulation → `Runner.ipynb`
4. Visualize results → `create_ganttchart.ipynb`

## 📝 License

MIT License – feel free to use and extend for research and teaching.  
Please provide appropriate credit when using or modifying this project in your own work.
