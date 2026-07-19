from __future__ import annotations

import asyncio
import copy
import itertools
import re
import sys
import uuid
from pathlib import Path

import yaml

from RunnerUtils import runner_main
from AutoPlotResults import generate_all_plots


def deep_merge(base, override):
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def set_nested(cfg, path, value):
    keys = path.split(".")
    current = cfg

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}

        if not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot set {path}: {key} is not a dictionary"
            )

        current = current[key]

    current[keys[-1]] = copy.deepcopy(value)


def safe_name(value):
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)

    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-")


def file_stem(path):
    return safe_name(Path(path).stem)


def timeout_tag(timeout):
    if timeout is None:
        return "no-timeout"

    return f"timeout-{safe_name(timeout)}"


def build_reward_weight_tags(cfg):
    reward_params = (
        cfg.get("gym", {})
        .get("reward", {})
        .get("params", {})
    )

    if not isinstance(reward_params, dict):
        raise TypeError(
            "gym.reward.params must be a dictionary"
        )

    weights = []

    for parameter_name, value in reward_params.items():
        match = re.fullmatch(
            r"weight(\d+)",
            str(parameter_name),
        )

        if match is None:
            continue

        weight_number = int(match.group(1))
        weights.append(
            (
                weight_number,
                f"w{weight_number}{safe_name(value)}",
            )
        )

    weights.sort(key=lambda item: item[0])

    return [tag for _, tag in weights]


def build_run_folder_name(cfg):
    algorithm = safe_name(cfg["run"]["algorithm"])

    if cfg["rl"]["enabled"]:
        rl_name = safe_name(cfg["rl"]["assign"])
        weight_tags = build_reward_weight_tags(cfg)

        if weight_tags:
            return "_".join([algorithm, rl_name, *weight_tags])

        return f"{algorithm}_{rl_name}"

    timeout = timeout_tag(cfg["run"]["algo_config"].get("timeout"))
    return f"{algorithm}_{timeout}"


def build_workload_root(cfg, run_uid, experiment_name):
    platform_name = file_stem(cfg["paths"]["platform"])
    workload_name = file_stem(cfg["paths"]["workload"])

    return (
        RESULTS_ROOT
        / safe_name(run_uid)
        / safe_name(experiment_name)
        / platform_name
        / workload_name
    )


def build_output_path(cfg, run_uid, experiment_name):
    return (
        build_workload_root(cfg, run_uid, experiment_name)
        / build_run_folder_name(cfg)
    )


def build_generated_config_path(
    cfg,
    run_uid,
    experiment_name,
    suffix=None,
):
    file_name = build_run_folder_name(cfg)

    if suffix is not None:
        file_name = f"{file_name}_{safe_name(suffix)}"

    return (
        build_workload_root(
            cfg,
            run_uid,
            experiment_name,
        )
        / "generated_configs"
        / f"{file_name}.yaml"
    )


def expand_sweep(sweep):
    if not sweep:
        yield {}
        return

    parameter_names = list(sweep.keys())
    parameter_values = []

    for parameter_name in parameter_names:
        values = sweep[parameter_name]

        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"Sweep parameter {parameter_name} "
                "must contain a list or tuple"
            )

        if not values:
            raise ValueError(
                f"Sweep parameter {parameter_name} is empty"
            )

        parameter_values.append(values)

    for combination in itertools.product(*parameter_values):
        yield dict(zip(parameter_names, combination))


def validate_config(cfg):
    required_fields = [
        ("paths", "workload"),
        ("paths", "platform"),
        ("paths", "output"),
        ("run", "algorithm"),
        ("run", "overrun_policy"),
        ("rl", "enabled"),
        ("rl", "learn"),
        ("logging", "level"),
    ]

    for section, field in required_fields:
        if section not in cfg or field not in cfg[section]:
            raise KeyError(
                f"Missing configuration field: {section}.{field}"
            )

        if cfg[section][field] is None:
            raise ValueError(
                f"Configuration field is None: {section}.{field}"
            )

    if cfg["rl"]["enabled"]:
        agent_name = cfg["rl"]["assign"]

        if agent_name not in cfg["rl"]["agents"]:
            raise KeyError(
                f"RL agent not found: {agent_name}"
            )


def write_generated_config(cfg, config_path):
    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False,
        )


def verify_generated_config(run_spec):
    config_path = Path(
        run_spec["config_path"]
    ).resolve()

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        written_cfg = yaml.safe_load(file)

    expected_cfg = run_spec["cfg"]
    checks = [
        (
            "paths.workload",
            written_cfg["paths"]["workload"],
            expected_cfg["paths"]["workload"],
        ),
        (
            "paths.platform",
            written_cfg["paths"]["platform"],
            expected_cfg["paths"]["platform"],
        ),
        (
            "paths.output",
            written_cfg["paths"]["output"],
            expected_cfg["paths"]["output"],
        ),
        (
            "rl.learn",
            written_cfg["rl"]["learn"],
            expected_cfg["rl"]["learn"],
        ),
        (
            "rl.checkpoint",
            written_cfg["rl"].get("checkpoint"),
            expected_cfg["rl"].get("checkpoint"),
        ),
    ]

    mismatches = [
        f"{name}: written={written!r}, expected={expected!r}"
        for name, written, expected in checks
        if written != expected
    ]

    if mismatches:
        raise RuntimeError(
            "Generated config does not match the run specification:\n"
            + "\n".join(mismatches)
        )

    print("\nVerified worker config:")
    print("config:", config_path)
    print("workload:", written_cfg["paths"]["workload"])
    print("output:", written_cfg["paths"]["output"])
    print("rl.enabled:", written_cfg["rl"]["enabled"])
    print("rl.learn:", written_cfg["rl"]["learn"])
    print("checkpoint:", written_cfg["rl"].get("checkpoint"))
    print("timeout:", written_cfg["run"].get("timeout"))

    if written_cfg["rl"]["enabled"]:
        print(
            "reward weights:",
            build_reward_weight_tags(written_cfg),
        )

    return config_path


def reserve_output_path(output_path, used_outputs):
    resolved_output = str(output_path.resolve())

    if resolved_output in used_outputs:
        raise ValueError(
            "Duplicate result path generated:\n"
            f"{output_path}\n\n"
            "Two experiments generated the same output path."
        )

    used_outputs.add(resolved_output)


def print_generated_run(run_spec):
    cfg = run_spec["cfg"]

    print("\nGenerated run:")
    print("experiment:", run_spec["name"])
    print("phase:", run_spec["phase"])
    print("uid:", run_spec.get("uid"))
    print(
        "root uid:",
        run_spec.get(
            "root_uid",
            run_spec.get("uid"),
        ),
    )
    print("config:", run_spec["config_path"])
    print("platform:", cfg["paths"]["platform"])
    print("workload:", cfg["paths"]["workload"])
    print("algorithm:", cfg["run"]["algorithm"])
    print("timeout:", cfg["run"].get("timeout"))
    print("output:", run_spec["output_path"])
    print("run parameters:", cfg["run"])

    if cfg["rl"]["enabled"]:
        print(
            "reward weights:",
            build_reward_weight_tags(cfg),
        )

    if run_spec["sweep"]:
        print("sweep:", run_spec["sweep"])

    if cfg["rl"].get("checkpoint") is not None:
        print("checkpoint:", cfg["rl"]["checkpoint"])


def make_run_record(run_spec, output_path=None):
    cfg = run_spec["cfg"]
    return {
        "name": run_spec["name"],
        "phase": run_spec["phase"],
        "uid": run_spec["uid"],
        "root_uid": run_spec.get("root_uid", run_spec["uid"]),
        "output": str(output_path if output_path is not None else run_spec["output_path"]),
        "platform": cfg["paths"]["platform"],
        "workload": cfg["paths"]["workload"],
        "algorithm": cfg["run"]["algorithm"],
        "timeout": cfg["run"].get("timeout"),
        "run_parameters": copy.deepcopy(cfg["run"]),
        "sweep": copy.deepcopy(run_spec["sweep"]),
        "config_path": str(run_spec["config_path"]),
    }

def build_curriculum_stage_spec(curriculum_spec,stage,stage_index,checkpoint_path):
    stage_name=safe_name(stage.get("name",f"stage-{stage_index+1}"))
    experiment_name=curriculum_spec["name"]
    root_uid=curriculum_spec["root_uid"]
    cfg=deep_merge(curriculum_spec["base_cfg"],stage.get("overrides",{}))
    cfg["paths"]["workload"]=stage["workload"]

    for key in ("workload","platform"):
        path=Path(cfg["paths"][key])
        if not path.is_absolute():
            path=PROJECT_ROOT/path
        cfg["paths"][key]=str(path.resolve())

    cfg["rl"]["enabled"]=True
    cfg["rl"]["learn"]=True
    cfg["rl"]["epochs"]=int(stage["epochs"])
    cfg["rl"]["checkpoint"]=str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None

    run_folder=f"{build_run_folder_name(cfg)}_{stage_index+1:02d}-{stage_name}"
    output_path=(build_workload_root(cfg,root_uid,experiment_name)/run_folder).resolve()
    cfg["paths"]["output"]=str(output_path)

    config_path=build_generated_config_path(
        cfg,
        root_uid,
        experiment_name,
        suffix=f"{stage_index+1:02d}-{stage_name}",
    ).resolve()

    validate_config(cfg)
    write_generated_config(cfg,config_path)

    stage_spec={
        "name":f"{experiment_name} - {stage_name}",
        "experiment_name":experiment_name,
        "phase":f"train-{stage_name}",
        "is_rl":True,
        "is_rl_training":True,
        "is_curriculum":False,
        "cfg":cfg,
        "config_path":config_path,
        "output_path":output_path,
        "sweep":copy.deepcopy(curriculum_spec["sweep"]),
        "uid":f"{root_uid}-stage-{stage_index+1}",
        "root_uid":root_uid,
        "test_overrides":copy.deepcopy(curriculum_spec.get("test_overrides") or {}),
    }

    print_generated_run(stage_spec)
    return stage_spec

def build_run_specs():
    run_specs = []
    used_outputs = set()
    run_uid = uuid.uuid4().hex[:8]

    for experiment in EXPERIMENTS:
        if not experiment.get("enabled", True):
            continue

        experiment_name = experiment["name"]
        experiment_cfg = deep_merge(
            BASE_CONFIG,
            experiment.get("overrides", {}),
        )

        for sweep_values in expand_sweep(experiment.get("sweep")):
            cfg = copy.deepcopy(experiment_cfg)

            for path, value in sweep_values.items():
                set_nested(cfg, path, value)

            curriculum = experiment.get("curriculum")

            if curriculum is not None:
                if not curriculum:
                    raise ValueError(
                        f"Curriculum is empty: {experiment_name}"
                    )

                if not cfg["rl"]["enabled"]:
                    raise ValueError(
                        f"Curriculum requires rl.enabled=True: {experiment_name}"
                    )

                curriculum_spec = {
                    "name": experiment_name,
                    "phase": "curriculum",
                    "is_rl": True,
                    "is_rl_training": True,
                    "is_curriculum": True,
                    "base_cfg": cfg,
                    "curriculum": copy.deepcopy(curriculum),
                    "test_overrides": copy.deepcopy(
                        experiment.get("test_overrides") or {}
                    ),
                    "sweep": copy.deepcopy(sweep_values),
                    "uid": run_uid,
                    "root_uid": run_uid,
                }

                print("\nGenerated curriculum:")
                print("experiment:", experiment_name)
                print("uid:", run_uid)
                print("stages:", len(curriculum))

                for index, stage in enumerate(curriculum, start=1):
                    print(
                        f" {index}. {stage.get('name', f'stage-{index}')}: "
                        f"{stage['workload']} ({stage['epochs']} epochs)"
                    )

                run_specs.append(curriculum_spec)
                continue

            output_path = build_output_path(cfg, run_uid, experiment_name)
            cfg["paths"]["output"] = str(output_path)

            validate_config(cfg)
            reserve_output_path(output_path, used_outputs)

            is_rl = bool(cfg["rl"]["enabled"])
            is_rl_training = is_rl and bool(cfg["rl"]["learn"])

            if is_rl_training:
                phase = "train"
            elif is_rl:
                phase = "test"
            else:
                phase = "run"

            config_path = build_generated_config_path(
                cfg,
                run_uid,
                experiment_name,
            )
            write_generated_config(cfg, config_path)

            run_spec = {
                "name": experiment_name,
                "phase": phase,
                "is_rl": is_rl,
                "is_rl_training": is_rl_training,
                "is_curriculum": False,
                "cfg": cfg,
                "config_path": config_path,
                "output_path": output_path,
                "sweep": copy.deepcopy(sweep_values),
                "uid": run_uid,
                "root_uid": run_uid,
            }

            if is_rl_training:
                run_spec["test_overrides"] = copy.deepcopy(
                    experiment.get("test_overrides") or {}
                )

            print_generated_run(run_spec)
            run_specs.append(run_spec)

    if not run_specs:
        raise RuntimeError("No enabled experiments were generated")

    return run_specs


async def run_worker_batch(run_specs, concurrency):
    if not run_specs:
        return

    config_paths = [
        verify_generated_config(run_spec)
        for run_spec in run_specs
    ]

    await runner_main(
        script_path=str(
            Path(WORKER_SCRIPT).resolve()
        ),
        args_=[
            [str(config_path)]
            for config_path in config_paths
        ],
        venv_path=str(
            Path(VENV_PATH).resolve()
        ),
        THREAD_LIMIT=concurrency,
    )


def find_best_epoch_checkpoint(training_output_path):
    training_output_path = Path(training_output_path)
    candidates = []

    for checkpoint_path in training_output_path.glob(
        "best_epoch_*/agent_checkpoint.pt"
    ):
        match = re.fullmatch(
            r"best_epoch_(\d+)",
            checkpoint_path.parent.name,
        )

        if match is None or not checkpoint_path.is_file():
            continue

        candidates.append(checkpoint_path)

    if not candidates:
        expected_path = (
            training_output_path
            / "best_epoch_<epoch>"
            / "agent_checkpoint.pt"
        )
        raise FileNotFoundError(
            "RL training finished, but the best epoch checkpoint "
            "was not found. Expected:\n"
            f"{expected_path}"
        )

    checkpoint_path = max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )

    if len(candidates) > 1:
        print(
            "Found multiple best_epoch directories; using the "
            "most recently modified checkpoint:",
            checkpoint_path,
        )

    return checkpoint_path.resolve()


def build_automatic_test_spec(training_spec,checkpoint_path):
    experiment_name=training_spec.get("experiment_name",training_spec["name"])
    root_uid=training_spec["root_uid"]
    test_overrides=training_spec.get("test_overrides") or {}
    test_cfg=deep_merge(copy.deepcopy(training_spec["cfg"]),test_overrides)

    for key in ("workload","platform"):
        path=Path(test_cfg["paths"][key])
        if not path.is_absolute():
            path=PROJECT_ROOT/path
        test_cfg["paths"][key]=str(path.resolve())

    test_cfg["rl"]["enabled"]=True
    test_cfg["rl"]["learn"]=False
    test_cfg["rl"]["checkpoint"]=str(Path(checkpoint_path).resolve())

    test_uid=f"{root_uid}-test"
    test_output_path=build_output_path(test_cfg,root_uid,experiment_name).resolve()
    test_config_path=build_generated_config_path(
        test_cfg,
        root_uid,
        experiment_name,
        suffix="test",
    ).resolve()

    test_cfg["paths"]["output"]=str(test_output_path)

    expected_workload=test_overrides.get("paths",{}).get("workload")
    if expected_workload is not None:
        expected_path=Path(expected_workload)
        if not expected_path.is_absolute():
            expected_path=PROJECT_ROOT/expected_path
        expected_workload=str(expected_path.resolve())
        if test_cfg["paths"]["workload"]!=expected_workload:
            raise RuntimeError(
                "Test workload override was not applied: "
                f"expected {expected_workload!r}, got {test_cfg['paths']['workload']!r}"
            )

    validate_config(test_cfg)
    write_generated_config(test_cfg,test_config_path)

    test_spec={
        "name":experiment_name,
        "phase":"test",
        "is_rl":True,
        "is_rl_training":False,
        "cfg":test_cfg,
        "config_path":test_config_path,
        "output_path":test_output_path,
        "sweep":copy.deepcopy(training_spec["sweep"]),
        "uid":test_uid,
        "root_uid":root_uid,
    }

    verify_generated_config(test_spec)
    print_generated_run(test_spec)
    return test_spec


async def execute_run_specs(run_specs, cores):
    plot_records = []
    pending_non_rl = []

    async def run_pending_non_rl():
        if not pending_non_rl:
            return

        print("\nRunning non-RL experiments:")
        print("runs:", len(pending_non_rl))
        print("concurrency:", cores)

        await run_worker_batch(
            pending_non_rl,
            concurrency=cores,
        )

        plot_records.extend(
            make_run_record(run_spec)
            for run_spec in pending_non_rl
        )
        pending_non_rl.clear()

    async def run_curriculum(curriculum_spec):
        configured_checkpoint = (
            curriculum_spec["base_cfg"]
            .get("rl", {})
            .get("checkpoint")
        )

        checkpoint_path = None

        if configured_checkpoint is not None:
            checkpoint_path = Path(configured_checkpoint)

            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path

            checkpoint_path = checkpoint_path.resolve()

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Curriculum starting checkpoint was not found:\n"
                    f"{checkpoint_path}"
                )

            print(
                "\nCurriculum starting checkpoint:",
                checkpoint_path,
            )

        final_stage_spec = None

        for stage_index, stage in enumerate(
            curriculum_spec["curriculum"]
        ):
            stage_spec = build_curriculum_stage_spec(
                curriculum_spec,
                stage,
                stage_index,
                checkpoint_path,
            )

            print("\nRunning curriculum stage:")
            print("stage:", stage_index + 1)
            print("name:", stage_spec["name"])
            print(
                "workload:",
                stage_spec["cfg"]["paths"]["workload"],
            )
            print(
                "epochs:",
                stage_spec["cfg"]["rl"]["epochs"],
            )
            print(
                "checkpoint:",
                stage_spec["cfg"]["rl"]["checkpoint"],
            )
            print("concurrency: 1")

            await run_worker_batch(
                [stage_spec],
                concurrency=1,
            )

            checkpoint_path = find_best_epoch_checkpoint(
                stage_spec["output_path"]
            )

            print("saved checkpoint:", checkpoint_path)

            plot_records.append(
                make_run_record(
                    stage_spec,
                    output_path=checkpoint_path.parent,
                )
            )

            final_stage_spec = stage_spec

        if final_stage_spec is None or checkpoint_path is None:
            raise RuntimeError(
                "Curriculum produced no checkpoint: "
                f"{curriculum_spec['name']}"
            )

        test_spec = build_automatic_test_spec(
            final_stage_spec,
            checkpoint_path,
        )

        print("\nRunning curriculum test:")
        print(
            "workload:",
            test_spec["cfg"]["paths"]["workload"],
        )
        print("checkpoint:", checkpoint_path)
        print("concurrency: 1")

        await run_worker_batch(
            [test_spec],
            concurrency=1,
        )

        plot_records.append(
            make_run_record(test_spec)
        )

    for run_spec in run_specs:
        if run_spec.get("is_curriculum", False):
            await run_pending_non_rl()
            await run_curriculum(run_spec)
            continue

        if not run_spec["is_rl"]:
            pending_non_rl.append(run_spec)
            continue

        await run_pending_non_rl()

        print("\nRunning RL experiment exclusively:")
        print("experiment:", run_spec["name"])
        print("phase:", run_spec["phase"])
        print("uid:", run_spec.get("uid"))
        print("concurrency: 1")

        await run_worker_batch(
            [run_spec],
            concurrency=1,
        )

        if not run_spec["is_rl_training"]:
            plot_records.append(
                make_run_record(run_spec)
            )
            continue

        checkpoint_path = find_best_epoch_checkpoint(
            run_spec["output_path"]
        )

        plot_records.append(
            make_run_record(
                run_spec,
                output_path=checkpoint_path.parent,
            )
        )

        print("\nBest epoch checkpoint:", checkpoint_path)

        test_spec = build_automatic_test_spec(
            run_spec,
            checkpoint_path,
        )

        print("\nRunning RL test immediately after training:")
        print("experiment:", test_spec["name"])
        print("concurrency: 1")

        await run_worker_batch(
            [test_spec],
            concurrency=1,
        )

        plot_records.append(
            make_run_record(test_spec)
        )

    await run_pending_non_rl()
    return plot_records


def main():
    cores = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_CORES
    )

    if len(sys.argv) > 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} [cores]"
        )

    if cores < 1:
        raise ValueError(
            "cores must be at least 1"
        )

    run_specs = build_run_specs()

    non_rl_count = sum(
        not run_spec["is_rl"]
        for run_spec in run_specs
    )

    rl_count = (
        len(run_specs) - non_rl_count
    )

    print("\nSPARS outer runner:")
    print("runs:", len(run_specs))
    print("non-RL runs:", non_rl_count)
    print("RL runs:", rl_count)
    print("non-RL concurrency:", cores)
    print("RL concurrency: 1")
    print("venv:", VENV_PATH)

    run_records = asyncio.run(
        execute_run_specs(
            run_specs,
            cores,
        )
    )

    if AUTO_PLOT:
        generate_all_plots(
            runs=run_records,
            generate_gantt=GENERATE_GANTT_PLOTS,
            generate_metrics=GENERATE_METRICS_PLOT,
        )


# ============================================================
# Parameters
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CORES = 1
WORKER_SCRIPT = PROJECT_ROOT / "RunSPARSConfig.py"
VENV_PATH = PROJECT_ROOT / "SPARS-venv"
RESULTS_ROOT = PROJECT_ROOT / "results"

AUTO_PLOT = True
GENERATE_GANTT_PLOTS = False
GENERATE_METRICS_PLOT = False


# ============================================================
# Shared configuration
# ============================================================

BASE_CONFIG = {
    "paths": {
        "workload": "workloads/workload-1.json",
        "platform": "platforms/platform.json",
        "output": "",  # Leave it empty! SPARS will generate the result path automatically based on your experiment setup.
    },
    "run": {
        "algorithm": "easy_psas",  # The scheduling algorithm. Check the Scheduler class to see what is available.
        "algo_config": {
            "timeout": None,  # Idle time before switching off an unused active node. None disables it.
        },
        "overrun_policy": "continue",  # "continue" lets overrunning jobs finish; "terminate" stops them at their requested time.
        "start_time": 0,
        "force_wakeup": False,  # Simulator fallback that wakes nodes when the system cannot make progress.
    },
    "rl": {
        "enabled": False,
        "learn": True,
        "type": "discrete",  # "discrete" requests a decision every dt; "continuous" requests decisions on job arrivals and completions.
        "dt": 3600,  # Required for discrete mode.
        "device": "cuda",  # "cpu" or "cuda".
        "epochs": 10,  # One epoch runs one complete simulation over the workload.
        "checkpoint": None,  # Path to an agent checkpoint. None creates a new agent.
        "episode_batch_size": 32,  # Number of episodes collected before each learning update.
        "agents": {
            # Implementation from "Improving the Efficiency of a Deep Reinforcement Learning-Based
            # Power Management System for HPC Clusters Using Curriculum Learning."
            "Budiarjo": {
                "class": "RL_Agent.SPARS.Budiarjo:Budiarjo",
                "params": {
                    "n_heads": 8,
                    "n_gae_layers": 3,
                    "input_dim": 11,
                    "embed_dim": 128,
                    "gae_ff_hidden": 512,
                    "tanh_clip": 10,
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
        "feature_extractor": "Budiarjo",
        "translator": "Budiarjo",
        "reward": {
            "name": "Budiarjo",
            "params": {
                "weight1": 0.5,
                "weight2": 0.5,
                "device": "cuda",
            },
        },
        "learner": "Budiarjo",
    },
    "logging": {
        "level": "INFO",  # "TRACE" or "INFO".
        "file": "results/simulation.log",
    },
}
# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    {
        "name": "Heuristic",
        "enabled": True,
        "overrides": {
            "run": {
                "force_wakeup": False,
                "algorithm": "easy_psus",
                "algo_config": {
                    "timeout": None,
                },
            },
        },
        "sweep": {
            "run.algorithm": ['easy_psus', 'easy_psas']
        }
    },
    {
        "name": "RL with Curriculum Training",
        "enabled": True,
        "overrides": {
            "run": {
                "force_wakeup": False,
                "algorithm": "easy_psus",
                "algo_config": {
                    "timeout": None,
                },
            },
            "rl": {
                "enabled": True,
                "assign": "Budiarjo",
            },
        },
        "curriculum": [
            {
                "name": "easy",
                "workload": "workloads\workload-2.json",
                "epochs": 1,
            },
            {
                "name": "medium",
                "workload": "workloads\workload-3.json",
                "epochs": 1,
            },
            {
                "name": "hard",
                "workload": "workloads\workload-4.json",
                "epochs": 1,
            },
        ],
        "test_overrides": {
            "paths": {
                "workload": "workloads\workload-1.json",
            },
        },
    }
]

if __name__ == "__main__":
    main()