# setup.py
from venv import create as create_venv
from os import getcwd, makedirs, name as os_name
from os.path import join
from subprocess import run
from pathlib import Path
import sys


def main():
    venv_dir = join(getcwd(), "SPARS-venv")
    print(f"Creating virtual environment at: {venv_dir}")
    create_venv(venv_dir, with_pip=True)

    # Scripts dir name differs by platform
    scripts_dir = "Scripts" if os_name == "nt" else "bin"
    python_exe = join(venv_dir, scripts_dir,
                      "python.exe" if os_name == "nt" else "python")

    # Pick requirements file based on OS (Windows vs. Linux/Unix-like)
    base_dir = Path(__file__).resolve().parent
    req_file = base_dir / (
        "requirements-windows.txt" if os_name == "nt" else "requirements-linux.txt"
    )

    if not req_file.exists():
        raise FileNotFoundError(f"{req_file.name} not found at {req_file}")

    print(f"Detected OS: {'Windows' if os_name == 'nt' else 'Linux/Unix-like'}")
    print(f"Using requirements file: {req_file}")

    # Upgrade pip (optional but helpful), then install
    print("Upgrading pip inside venv…")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print(f"Installing dependencies from {req_file} ...")
    run([python_exe, "-m", "pip", "install", "-r", str(req_file)], check=True)

    print("\nDone.")
    print("To activate the venv:")
    if os_name == "nt":
        print(rf"  PowerShell: .\SPARS-venv\Scripts\Activate.ps1")
        print(rf"  CMD      : .\SPARS-venv\Scripts\activate.bat")
    else:
        print(rf"  Bash/Zsh : source ./SPARS-venv/bin/activate")

    makedirs(join(getcwd(), "workloads"), exist_ok=True)
    makedirs(join(getcwd(), "platforms"), exist_ok=True)
    makedirs(join(getcwd(), "plt"), exist_ok=True)
    makedirs(join(getcwd(), "results"), exist_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        sys.exit(1)
