from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path


# The master script sets this variable for each child process.
# It overrides any THREAD_LIMIT supplied by an outer script.
RUNNER_THREAD_LIMIT_ENV = "RUNNER_THREAD_LIMIT"
VENV_PATH = "./SPARS-venv"

def find_venv_python() -> Path:
    """Return the Python executable belonging to a virtual environment."""
    venv = Path(VENV_PATH)

    # Linux/macOS
    unix_python = venv / "bin" / "python"

    # Windows
    windows_python = venv / "Scripts" / "python.exe"

    if unix_python.is_file():
        return unix_python

    if windows_python.is_file():
        return windows_python

    raise FileNotFoundError(
        "Virtual-environment Python was not found.\n"
        f"Checked: {unix_python.resolve()}\n"
        f"Checked: {windows_python.resolve()}"
    )


def resolve_thread_limit(requested_limit: int | None) -> int:
    """
    Determine the number of scripts allowed to run concurrently.

    RUNNER_THREAD_LIMIT is checked first. This lets the master script
    enforce the concurrency limit, even when an outer script passes its
    own THREAD_LIMIT value.
    """
    environment_value = os.environ.get(RUNNER_THREAD_LIMIT_ENV)

    if environment_value is not None:
        try:
            thread_limit = int(environment_value)
        except ValueError as error:
            raise ValueError(
                f"{RUNNER_THREAD_LIMIT_ENV} must be an integer, "
                f"not {environment_value!r}"
            ) from error
    elif requested_limit is not None:
        thread_limit = requested_limit
    else:
        # Safe default: only one child script at a time.
        thread_limit = 1

    if thread_limit < 1:
        raise ValueError("THREAD_LIMIT must be at least 1")

    return thread_limit


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


async def run_script_async(
    semaphore: asyncio.Semaphore,
    script_path: str | Path,
    venv_path: str | Path,
    args: Sequence[object] | None,
    iterator: int,
) -> None:
    """Run one Python script instance and wait for it to finish."""
    script = Path(script_path)

    if not script.is_file():
        raise FileNotFoundError(
            f"Script not found: {script.resolve()}"
        )

    python_executable = find_venv_python()

    command = [
        str(python_executable),
        str(script),
        *(str(argument) for argument in (args or [])),
    ]

    async with semaphore:
        print(
            f"\nStarting instance {iterator}: {script}",
            flush=True,
        )
        print(
            f"Command: {' '.join(command)}",
            flush=True,
        )

        # stdout and stderr are inherited so output appears live.
        process = await asyncio.create_subprocess_exec(*command)

        try:
            return_code = await process.wait()
        except asyncio.CancelledError:
            await terminate_process(process)
            raise

        if return_code != 0:
            raise RuntimeError(
                f"{script} instance {iterator} failed "
                f"with exit code {return_code}"
            )

        print(
            f"Finished instance {iterator}: {script}",
            flush=True,
        )


async def runner_main(
    script_path: str | Path,
    args_: Sequence[Sequence[object]],
    venv_path: str | Path = "../ermc-py-venv-cpu",
    THREAD_LIMIT: int | None = None,
) -> None:
    """
    Run multiple instances of the same script.

    The effective concurrency is selected in this order:

    1. RUNNER_THREAD_LIMIT environment variable set by the master.
    2. THREAD_LIMIT passed directly to this function.
    3. Default value of 1.

    The master therefore controls whether an outer script runs workers
    sequentially or concurrently.
    """
    thread_limit = resolve_thread_limit(THREAD_LIMIT)

    print(
        f"\nRunner configuration for {script_path}:",
        flush=True,
    )
    print(
        f"  Instances: {len(args_)}",
        flush=True,
    )
    print(
        f"  Concurrent instances: {thread_limit}",
        flush=True,
    )

    semaphore = asyncio.Semaphore(thread_limit)

    tasks = [
        asyncio.create_task(
            run_script_async(
                semaphore=semaphore,
                script_path=script_path,
                venv_path=venv_path,
                args=args,
                iterator=iterator,
            )
        )
        for iterator, args in enumerate(args_, start=1)
    ]

    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # Cancel any workers that have not finished if one fails.
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise