import os
from pathlib import Path


# The dashboard runs batches from a snapshot of src/ and points SPARS_ROOT back
# at the real project, so data and results stay in one place.
PROJECT_ROOT = Path(os.environ.get("SPARS_ROOT") or Path(__file__).resolve().parents[4])

DEFAULT_CORES = 1
WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "runner" / "worker.py"
RESULTS_ROOT = PROJECT_ROOT / "results"

GENERATE_GANTT_PLOTS = False
GENERATE_METRICS_PLOT = False


# ============================================================
# Shared configuration
# ============================================================

BASE_CONFIG = {
    "paths": {
        "workload": "workloads/generated-markovian-3000.json",
        "platform": "platforms/AOBA-64.json",
        "output": "",
    },
    "run": {
        # A definition names a class under spars/policies/ (see spars/policies/discovery.py).
        "scheduler": {"name": "FCFSPSAS", "config": {}},
        "power_management": {"name": "TimeoutPowerManager", "config": {"timeout": 3600}},
        "overrun_policy": "continue",
        "start_time": 0,
        "force_wakeup": False,
    },
    "rl": {
        "enabled": False,
        "learn": True,
        "type": "discrete",
        "dt": 3600,
        "device": "gpu",
        "epochs": 10,
        "checkpoint": None,
        "seed": 42,
        "episode_batch_size": 32,
        # {"workload": path, "patience": None | int, "window": odd int}: pick
        # the best epoch by its argmax return on this workload (averaged over
        # the window of epochs centred on it) instead of the training return.
        "validation": None,
        "agents": {
            "Budiarjo": {
                "class": "spars.policies.power_management.rl.budiarjo.model:Budiarjo",
                "params": {
                    "n_heads": 8,
                    "n_gae_layers": 3,
                    "input_dim": 11,
                    "embed_dim": 128,
                    "gae_ff_hidden": 512,
                    "tanh_clip": 10,
                    "device": "gpu",
                },
                "optimizer": {
                    "class": "torch.optim:Adam",
                    "params": {
                        "lr": 0.0001,
                    },
                },
            },
        },
        "assign": "Budiarjo",
    },
    "gym": {
        "feature_extractor": "spars.policies.power_management.rl.budiarjo.features:feature_extraction",
        "translator": "spars.policies.power_management.rl.budiarjo.translator:action_translator",
        "reward": {
            "name": "spars.policies.power_management.rl.reward:Reward",
            "params": {
                "weight1": 0.5,
                "weight2": 0.5,
                "device": "gpu",
            },
        },
        "learner": "spars.policies.power_management.rl.budiarjo.learner:learn",
    },
    "logging": {
        "level": "QUIET",
        "file": "simulation.log",  # relative paths are inside the batch folder results/<Set>-NNNNNN/
    },
}
