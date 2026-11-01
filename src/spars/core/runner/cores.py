"""Split the CPUs into Simulation cores and Support cores.

A Simulation core runs exactly one simulation worker and nothing else. Support
cores run everything else: the CLI, the executor, the progress dashboard, log
forwarding and plotting.

A core is a physical core: hyperthread siblings stay together, so no other
process shares a simulation's core. Cores are ordered from the fastest maximum
frequency to the slowest, which puts performance cores before efficiency cores
on hybrid CPUs. Simulations take the fastest cores and Support takes the
slowest. On CPUs without frequency information the CPU ID order is used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import psutil


SIMULATION_CPUS_ENV = "SPARS_SIMULATION_CPUS"
SUPPORT_CPUS_ENV = "SPARS_SUPPORT_CPUS"
DEFAULT_SUPPORT_CORES = 1
_SYSFS_CPU = Path("/sys/devices/system/cpu")


@dataclass(frozen=True)
class CoreAllocation:
    # One CPU group (a physical core) per simulation slot, fastest first.
    simulation: tuple[tuple[int, ...], ...]
    support: tuple[int, ...]

    def to_environment(self) -> dict[str, str]:
        return {
            SIMULATION_CPUS_ENV: json.dumps([list(core) for core in self.simulation]),
            SUPPORT_CPUS_ENV: json.dumps(list(self.support)),
        }


def _parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, ValueError):
        return None


def available_cpus() -> set[int]:
    """The CPUs this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    try:
        return set(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):  # macOS has no CPU affinity
        return set(range(os.cpu_count() or 1))


def set_affinity(cpus, pid: int = 0) -> None:
    """Pin a process to the CPUs; does nothing where the OS has no affinity (macOS)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(pid, set(cpus))
        return
    try:
        psutil.Process(pid or None).cpu_affinity(sorted(cpus))
    except (AttributeError, psutil.Error):
        pass


def _threads_per_core() -> int:
    """Hyperthreads per core when there is no sysfs (Windows numbers siblings together)."""
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False) or logical
    return logical // physical if logical % physical == 0 else 1


def physical_cores(available: set[int]) -> list[tuple[int, ...]]:
    """Group the available CPUs by physical core, fastest core first."""
    remaining = set(available)
    threads = _threads_per_core()
    cores: list[tuple[int, int, tuple[int, ...]]] = []

    for cpu in sorted(available):
        if cpu not in remaining:
            continue
        siblings = _read(_SYSFS_CPU / f"cpu{cpu}/topology/thread_siblings_list")
        if siblings:
            group = _parse_cpu_list(siblings) & remaining
        else:
            group = {c for c in available if c // threads == cpu // threads} & remaining
        group.add(cpu)
        remaining -= group
        frequency = 0
        for member in group:
            value = _read(_SYSFS_CPU / f"cpu{member}/cpufreq/cpuinfo_max_freq")
            if value is not None and value.strip().isdigit():
                frequency = max(frequency, int(value))
        cores.append((-frequency, min(group), tuple(sorted(group))))

    return [group for *_, group in sorted(cores)]


def allocate_cores(
    simulation_cores: int,
    support_cores: int,
    available: set[int] | None = None,
) -> CoreAllocation:
    """Pick the Simulation and Support cores from the given CPUs.

    By default these are the CPUs this process may run on.
    """
    if simulation_cores < 1:
        raise ValueError("simulation cores must be at least 1")
    if support_cores < 1:
        raise ValueError("support cores must be at least 1")

    if available is None:
        available = available_cpus()
    cores = physical_cores(available)
    if len(cores) == 1 and simulation_cores == support_cores == 1:
        # A one-core machine (a small VM): simulation and support share the core.
        return CoreAllocation(simulation=(cores[0],), support=tuple(sorted(cores[0])))
    if simulation_cores + support_cores > len(cores):
        raise ValueError(
            f"Requested {simulation_cores} Simulation + {support_cores} Support "
            f"cores, but only {len(cores)} physical cores are available"
        )

    support = cores[len(cores) - support_cores :]
    return CoreAllocation(
        simulation=tuple(cores[:simulation_cores]),
        support=tuple(sorted(cpu for core in support for cpu in core)),
    )


def inherited_allocation() -> CoreAllocation | None:
    """Return the allocation handed down by the CLI, if any."""
    simulation = os.environ.get(SIMULATION_CPUS_ENV)
    support = os.environ.get(SUPPORT_CPUS_ENV)
    if simulation is None or support is None:
        return None
    return CoreAllocation(
        simulation=tuple(tuple(int(cpu) for cpu in core) for core in json.loads(simulation)),
        support=tuple(int(cpu) for cpu in json.loads(support)),
    )


def resolve_core_allocation(
    simulation_cores: int, support_cores: int
) -> CoreAllocation:
    """Reuse the CLI's allocation or make one, then pin this process to Support."""
    allocation = inherited_allocation()
    if allocation is None:
        allocation = allocate_cores(simulation_cores, support_cores)
    elif len(allocation.simulation) < simulation_cores:
        raise ValueError(
            f"{simulation_cores} workers requested, but only "
            f"{len(allocation.simulation)} Simulation cores were allocated"
        )
    set_affinity(allocation.support)
    return allocation
