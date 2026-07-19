from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import psutil

from RunnerUtils import RUNNER_THREAD_LIMIT_ENV, find_venv_python


DEFAULT_CORES = 2
DEFAULT_USE_GPU = False


def parse_bool(value: str) -> bool:
    """Convert a command-line string into a boolean."""
    normalized = value.strip().lower()

    if normalized not in {"true", "false"}:
        raise ValueError(
            "use_gpu must be either 'true' or 'false'"
        )

    return normalized == "true"


async def terminate_process(
    process: asyncio.subprocess.Process,
) -> None:
    """Terminate a subprocess cleanly, killing it if necessary."""
    if process.returncode is not None:
        return

    try:
        process.terminate()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return

        await process.wait()


async def run_script(
    script_path: str | Path,
    python_path: Path,
    args: Sequence[object] | None = None,
    thread_limit: int = 1,
) -> None:
    """
    Run one outer Python script and wait for it to finish.

    thread_limit is passed to the outer process through the
    RUNNER_THREAD_LIMIT environment variable. Any runner_main() call
    made by that outer process will use this value.
    """
    script = Path(script_path)

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script.resolve()}"
        )

    if thread_limit < 1:
        raise ValueError("thread_limit must be at least 1")

    command = [
        str(python_path),
        str(script),
        *(str(argument) for argument in (args or [])),
    ]

    child_environment = os.environ.copy()
    child_environment[RUNNER_THREAD_LIMIT_ENV] = str(thread_limit)

    print("\n" + "=" * 70, flush=True)
    print(f"Starting: {script}", flush=True)
    print(f"Worker concurrency: {thread_limit}", flush=True)
    print(f"Command: {' '.join(command)}", flush=True)
    print("=" * 70, flush=True)

    # stdout and stderr are inherited so child output appears live.
    process = await asyncio.create_subprocess_exec(
        *command,
        env=child_environment,
    )

    try:
        return_code = await process.wait()
    except asyncio.CancelledError:
        await terminate_process(process)
        raise

    if return_code != 0:
        raise RuntimeError(
            f"{script} failed with exit code {return_code}"
        )

    print(f"\nFinished: {script}", flush=True)


def configure_cpu_affinity(cores: int) -> list[int]:
    """
    Restrict the master process and its children to the requested CPUs.

    Child processes inherit the master's CPU affinity.
    """
    current_process = psutil.Process()

    if not hasattr(current_process, "cpu_affinity"):
        raise RuntimeError(
            "CPU affinity is not supported on this operating system"
        )

    available_cpus = current_process.cpu_affinity()

    if cores > len(available_cpus):
        raise ValueError(
            f"Requested {cores} cores, but only "
            f"{len(available_cpus)} CPU cores are available"
        )

    selected_cpus = available_cpus[:cores]
    current_process.cpu_affinity(selected_cpus)

    return selected_cpus


async def main() -> None:
    # Command-line format:
    #
    #   python master.py [cores] [use_gpu]
    #
    # Examples:
    #
    #   python master.py
    #   python master.py 4
    #   python master.py 4 true

    cores = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_CORES
    )

    if len(sys.argv) > 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} [cores] [use_gpu]\n"
            "\n"
            "Examples:\n"
            f"  {sys.argv[0]}\n"
            f"  {sys.argv[0]} 4\n"
            f"  {sys.argv[0]} 4 true"
        )

    if cores < 1:
        raise ValueError("cores must be at least 1")

    selected_cpus = configure_cpu_affinity(cores)

    python_path = find_venv_python()

    print("Master script configuration:", flush=True)
    print(f"  Cores: {cores}", flush=True)
    print(f"  CPU IDs: {selected_cpus}", flush=True)
    print(f"  Python: {python_path.resolve()}", flush=True)

    await run_script(
        script_path="RunSPARSOuter.py",
        python_path=python_path,
        args=[
            cores,
        ],
        thread_limit=cores,
    )

    print(
        "\nAll scripts completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
    ) as error:
        print(
            f"\nError: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    except KeyboardInterrupt:
        print(
            "\nExecution cancelled.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)