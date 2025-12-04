from SPARS.Simulator.Algo.easy_auto_switch_on import EASY
from SPARS.Simulator.Algo.easy_normal import EASYNormal
from SPARS.Simulator.Algo.fcfs_auto_switch_on import FCFS
from SPARS.Simulator.Algo.fcfs_normal import FCFSNormal
from SPARS.Simulator.Algo.fcfs_psus import FCFSPSUS
from SPARS.Simulator.Algo.easy_psus import EASYPSUS
from SPARS.Simulator.Algo.fcfs_psas import FCFSPSAS
from SPARS.Simulator.Algo.easy_psas import EASYPSAS

ALGO_MAP = {
    'fcfs': FCFS,
    'fcfs_normal': FCFSNormal,
    'easy_normal': EASYNormal,
    'easy': EASY,
    'fcfs_psus': FCFSPSUS,
    'easy_psus': EASYPSUS,
    'fcfs_psas': FCFSPSAS,
    'easy_psas': EASYPSAS,
}


class Scheduler:
    def __init__(self, machines, jobs_manager, algorithm, start_time, timeout=None):
        AlgorithmClass = ALGO_MAP[algorithm.lower()]
        self.algorithm = AlgorithmClass(
            machines,
            jobs_manager,
            start_time,
            timeout
        )

    def schedule(self, current_time):
        self.algorithm.set_time(current_time)
        events = self.algorithm.schedule()
        return events
