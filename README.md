# HPCv3 Simulator 🖥️⚡

HPCv3 is a lightweight simulator for studying **job scheduling** and **resource allocation** in High-Performance Computing (HPC) systems.  
It allows you to **generate workloads**, **define platforms**, **convert swf formats to JSON**, **run scheduling simulations** with different algorithms, and **visualize results**.

## 📥 Installation

### Prerequisites

- Python 3.9 or newer
- Recommended: create a virtual environment to isolate dependencies

### Clone and Setup Environment

```bash
git clone https://github.com/algoritmakomputasi-ugm/HPCv3-AIBT-2025.git
cd HPCv3-AIBT-2025
python setup.py
```

## ⚙️ Workflow Overview

Workload Generation → Define synthetic job traces.

Platform Generation → Define HPC platform topology and machine states.

Scheduling Simulation → Run the simulation with different schedulers.

Results Visualization → Analyze and visualize scheduling outcomes.

Each stage is provided as a Jupyter Notebook for ease of experimentation.

## 📂 Workload Generation

Notebook: `workloads_generator.ipynb`

This stage generates job traces that describe:

- Number of jobs
- Arrival times
- Job duration
- Resource requirements

You can configure parameters (e.g., inter-arrival rate, job length distribution) to model different HPC workloads.
The output is stored as a .json or .csv file, which will be used as input for the simulation.

## 🏗️ Platform Generation

Notebook: `platforms_generator.ipynb`

This stage defines the HPC platform, including:

- Number of compute nodes
- Available DVFS (Dynamic Voltage and Frequency Scaling) profiles
- Power consumption and compute speed
- The generated platform description file is also saved as .json, providing the environment where jobs will be scheduled.

## 💻 Scheduling Simulation

Notebook: `SPARS_runner.ipynb`

Here you run the simulation by combining:

- A workload file (jobs to run)
- A platform file (system configuration)
- A scheduler of your choice

Available schedulers:

1. **Easy Backfilling** – Improves utilization by allowing smaller jobs to jump ahead if they don’t delay larger jobs.
2. **FCFS (First-Come, First-Served)** – Jobs are executed in the order they arrive.
3. **Smart FCFS** – An enhanced FCFS variant with an early switch-on policy.

The simulation produces CSV logs containing job start/finish times, node allocations, and system events.

## 📊 Results Visualization

Notebook: create_ganttchart.ipynb

This stage transforms raw CSV logs into visual insights.
Outputs include:

- Gantt chart → shows job execution timelines and node allocations
- Job statistics → execution time, waiting time, utilization
- Energy consumption analysis (if DVFS/platform states are enabled)

These visualizations help you evaluate the effectiveness of different scheduling policies and platform configurations.

## 🚀 Example End-to-End Run

1. Generate a workload → `workloads_generator.ipynb`
2. Generate a platform → `platforms_generator.ipynb`
3. Run simulation → `HPCv3_runner.ipynb`
4. Visualize results → `create_ganttchart.ipynb`

## Additional SWF to JSON converter

The notebook `swf_to_json.ipynb` lets you convert SWF format files into JSON files.

## 📝 License

MIT License – feel free to use and extend for research and teaching.  
Please provide appropriate credit when using or modifying this project in your own work.
