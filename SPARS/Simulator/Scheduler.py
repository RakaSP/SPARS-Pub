from inspect import Parameter, signature

from SPARS.Simulator.Algo.algo_config import require_algo_config
from SPARS.Simulator.Algo.easy import EASY
from SPARS.Simulator.Algo.easy_baseline import EASYNormal
from SPARS.Simulator.Algo.easy_baseline_psas import EASYBaselinePSAS
from SPARS.Simulator.Algo.easy_psas import EASYPSAS
from SPARS.Simulator.Algo.easy_psus import EASYPSUS
from SPARS.Simulator.Algo.fcfs import FCFS
from SPARS.Simulator.Algo.fcfs_baseline import FCFSNormal
from SPARS.Simulator.Algo.fcfs_baseline_psas import FCFSBaselinePSAS
from SPARS.Simulator.Algo.fcfs_psas import FCFSPSAS
from SPARS.Simulator.Algo.fcfs_psus import FCFSPSUS


ALGO_MAP = {
    "fcfs": FCFS,
    "fcfs_baseline": FCFSNormal,
    "easy": EASY,
    "easy_baseline": EASYNormal,
    "fcfs_psus": FCFSPSUS,
    "easy_psus": EASYPSUS,
    "fcfs_psas": FCFSPSAS,
    "fcfs_baseline_psas": FCFSBaselinePSAS,
    "easy_baseline_psas": EASYBaselinePSAS,
    "easy_psas": EASYPSAS,
}


class Scheduler:
    def __init__(
        self,
        machines,
        jobs_manager,
        algorithm,
        start_time,
        algo_config,
        workload=None,
        platform=None,
        monitor=None,
        platform_control=None,
    ):
        algorithm_name = algorithm.lower()

        if algorithm_name not in ALGO_MAP:
            raise ValueError(
                f"Unknown scheduling algorithm: {algorithm!r}. "
                f"Available algorithms: {sorted(ALGO_MAP)}"
            )

        algorithm_class = ALGO_MAP[algorithm_name]

        runtime_values = {
            "machines": machines,
            "jobs_manager": jobs_manager,
            "start_time": start_time,
            "workload": workload,
            "platform": platform,
            "monitor": monitor,
            "platform_control": platform_control,
        }

        constructor_parameters = signature(
            algorithm_class.__init__
        ).parameters

        runtime_arguments = {}
        configurable_parameters = []

        for parameter_name, parameter in constructor_parameters.items():
            if parameter_name == "self":
                continue

            if parameter.kind in (
                Parameter.VAR_POSITIONAL,
                Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"{algorithm_class.__name__}.__init__ cannot use "
                    f"*args or **kwargs."
                )

            if parameter_name in runtime_values:
                runtime_arguments[parameter_name] = runtime_values[
                    parameter_name
                ]
                continue

            if parameter.default is not Parameter.empty:
                raise TypeError(
                    f"Configurable parameter {parameter_name!r} in "
                    f"{algorithm_class.__name__} has a default value."
                )

            configurable_parameters.append(parameter_name)

        validated_algo_config = require_algo_config(
            algo_config=algo_config,
            configurable_parameters=configurable_parameters,
            algorithm_name=algorithm_class.__name__,
        )

        self.algorithm = algorithm_class(
            **runtime_arguments,
            **validated_algo_config,
        )

    def schedule(self, current_time):
        self.algorithm.set_time(current_time)
        return self.algorithm.schedule()
