#!/usr/bin/env python3
"""Cross-platform SPARS setup.

Creates/recreates ./SPARS-venv, optionally installs the best available PyTorch
build for this machine, installs the project in editable mode with dev tools,
and verifies the resulting environment.

At startup the installer explains what PyTorch is for and asks
"Install PyTorch? [y/N]". The default (Enter, or no terminal) builds the
environment without PyTorch; everything else in pyproject.toml is still
installed. Use --torch / --no-torch to skip the question.

The installer itself intentionally uses Python 3.8-compatible syntax so it can
run under an older interpreter long enough to locate a Python version that
satisfies the project's requires-python setting.
"""

from __future__ import print_function

import glob
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
VENV = PROJECT_ROOT / "SPARS-venv"
PYTORCH_ROOT = "https://download.pytorch.org/whl/"
VIRTUALENV_BOOTSTRAP = "https://bootstrap.pypa.io/virtualenv.pyz"
PROJECT_NAME = "spars"


class ChannelParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.channels = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() != "href" or not value:
                continue
            name = value.rstrip("/").split("/")[-1]
            if re.match(r"^cu\d+$", name):
                self.channels.add(name)
            elif re.match(r"^rocm\d+(?:\.\d+)+$", name):
                self.channels.add(name)
            elif name in ("cpu", "xpu"):
                self.channels.add(name)


def say(message=""):
    print(message, flush=True)


def ask_yes_no(question, default=True):
    """Ask a Y/n question. Returns default on empty input, EOF or no TTY."""
    suffix = " [Y/n] " if default else " [y/N] "
    if not sys.stdin or not sys.stdin.isatty():
        say("%s%s(non-interactive, using default: %s)"
            % (question, suffix, "yes" if default else "no"))
        return default
    while True:
        try:
            answer = input(question + suffix).strip().lower()
        except EOFError:
            say()
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        say("Please answer 'y' or 'n'.")


TORCH_QUESTION = """
PyTorch is only needed for the reinforcement-learning (RL) power manager:
training and evaluating RL agents. Everything else works without it: the
schedulers, the other power managers, the dashboard, the workload and platform
generators, the SWF converter and the plots. PyTorch is a large download
(about 1 GB, more with GPU support). You can add it later by running
"python install.py --torch".
"""


def decide_install_torch():
    """Resolve whether to install PyTorch from CLI flags or by asking (default: no)."""
    args = sys.argv[1:]
    if "--no-torch" in args:
        say("Install PyTorch? -> no (--no-torch)")
        return False
    if "--torch" in args or "--yes" in args or "-y" in args:
        say("Install PyTorch? -> yes (--torch)")
        return True
    say(TORCH_QUESTION)
    return ask_yes_no("Install PyTorch?", default=False)


def format_command(command):
    display = []
    skip_script = False
    for index, part in enumerate(command):
        text = str(part)
        if index > 0 and str(command[index - 1]) == "-c" and len(text) > 120:
            text = "<verification code>"
        display.append('"%s"' % text if " " in text else text)
    return " ".join(display)


def run(command, cwd=None, capture=False, check=True, env=None):
    command = [str(part) for part in command]
    say("$ " + format_command(command))
    kwargs = {
        "cwd": str(cwd) if cwd else None,
        "env": env,
        "text": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    result = subprocess.run(command, **kwargs)
    if check and result.returncode != 0:
        output = (result.stdout or "").strip() if capture else ""
        if output:
            say(output)
        raise RuntimeError("Command failed with exit code %s" % result.returncode)
    return result


def output_of(command):
    try:
        result = run(command, capture=True, check=False)
    except (OSError, RuntimeError):
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def parse_project_min_python():
    if not PYPROJECT.is_file():
        raise FileNotFoundError("pyproject.toml must be beside install.py")
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"requires-python\s*=\s*['\"]\s*>=\s*(\d+)\.(\d+)",
        text,
    )
    if not match:
        raise RuntimeError(
            "install.py expects pyproject.toml to declare requires-python as >=X.Y"
        )
    return int(match.group(1)), int(match.group(2))


def probe_python(command):
    try:
        result = subprocess.run(
            list(command)
            + [
                "-c",
                "import sys; print('%d.%d.%d' % sys.version_info[:3])",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", result.stdout.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def candidate_python_commands(minimum):
    seen = set()

    def add(command):
        key = tuple(str(part) for part in command)
        if key not in seen:
            seen.add(key)
            return True
        return False

    current = [sys.executable]
    if add(current):
        yield current

    if os.name == "nt" and shutil.which("py"):
        launcher_output = output_of(["py", "-0p"])
        for line in launcher_output.splitlines():
            # The Windows launcher usually prints a version marker followed by a path.
            path_match = re.search(r"([A-Za-z]:\\.*?python(?:\.exe)?)\s*$", line)
            if path_match:
                command = [path_match.group(1)]
                if add(command):
                    yield command

    # Generic PATH names. The upper bound is deliberately generous so future
    # Python 3.x releases are found without changing this installer.
    for minor in range(30, minimum[1] - 1, -1):
        for name in ("python3.%d" % minor, "python%d.%d" % (minimum[0], minor)):
            path = shutil.which(name)
            if path:
                command = [path]
                if add(command):
                    yield command

    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            command = [path]
            if add(command):
                yield command

    if os.name != "nt":
        patterns = (
            "/usr/local/bin/python3.*",
            "/usr/bin/python3.*",
            "/opt/homebrew/bin/python3.*",
            "/opt/local/bin/python3.*",
        )
        for pattern_name in patterns:
            for path in sorted(glob.glob(pattern_name), reverse=True):
                command = [path]
                if add(command):
                    yield command


def find_compatible_python(minimum):
    # Use the interpreter that launched this installer when it is new enough, so
    # "py -3.11 install.py" builds the environment on 3.11, not the newest Python.
    current = probe_python([sys.executable])
    if current is not None and current[:2] >= minimum:
        return current, [sys.executable]

    found = []
    for command in candidate_python_commands(minimum):
        version = probe_python(command)
        if version is None:
            continue
        found.append((version, command))
    compatible = [item for item in found if item[0][:2] >= minimum]
    if not compatible:
        discovered = ", ".join(
            "%s (%s)" % (format_command(command), ".".join(map(str, version)))
            for version, command in sorted(found, reverse=True)
        )
        raise RuntimeError(
            "SPARS requires Python %d.%d or newer. Compatible Python was not found.%s"
            % (
                minimum[0],
                minimum[1],
                (" Found: " + discovered) if discovered else "",
            )
        )
    compatible.sort(key=lambda item: item[0], reverse=True)
    return compatible[0]


def remove_tree(path):
    def onerror(func, failing_path, _exc_info):
        try:
            os.chmod(failing_path, stat.S_IWRITE | stat.S_IREAD)
            func(failing_path)
        except OSError:
            pass

    if path.exists():
        shutil.rmtree(str(path), onerror=onerror)


def create_venv(base_python):
    remove_tree(VENV)
    say("Creating %s" % VENV.name)
    result = subprocess.run(
        list(base_python) + ["-m", "venv", str(VENV)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        # Do not install virtualenv into an OS-managed Python.  Use PyPA's
        # self-contained zipapp instead, which works on Windows/macOS/Linux.
        say("stdlib venv failed; trying PyPA virtualenv bootstrap")
        with tempfile.TemporaryDirectory(prefix="spars-virtualenv-") as directory:
            zipapp = Path(directory) / "virtualenv.pyz"
            request = urllib.request.Request(
                VIRTUALENV_BOOTSTRAP,
                headers={"User-Agent": "SPARS-universal-installer/1"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                zipapp.write_bytes(response.read())
            run(list(base_python) + [str(zipapp), str(VENV)])
    python = venv_python()
    if not python.is_file():
        raise RuntimeError("Virtual environment was created without a Python executable")
    run([python, "-m", "ensurepip", "--upgrade"], check=False)
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    return python


def venv_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def discover_pytorch_channels():
    request = urllib.request.Request(
        PYTORCH_ROOT,
        headers={"User-Agent": "SPARS-universal-installer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as error:
        say("PyTorch channel discovery unavailable: %s" % error)
        return set()
    parser = ChannelParser()
    parser.feed(html)
    return parser.channels


def cuda_channel_version(name):
    match = re.match(r"^cu(\d+)$", name)
    if not match:
        return None
    digits = match.group(1)
    # PyTorch channel names encode CUDA without a dot: cu92 -> 9.2,
    # cu118 -> 11.8, cu132 -> 13.2.  Parsing the major separately
    # also keeps future multi-digit minors sane (for example cu1310 -> 13.10).
    if len(digits) == 2:
        return int(digits[0]), int(digits[1:])
    if len(digits) >= 3:
        return int(digits[:2]), int(digits[2:])
    return None


def rocm_channel_version(name):
    match = re.match(r"^rocm(\d+(?:\.\d+)+)$", name)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def parse_version(text):
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not match:
        return None
    parts = [int(match.group(1)), int(match.group(2))]
    if match.group(3) is not None:
        parts.append(int(match.group(3)))
    return tuple(parts)


def detect_nvidia_cuda():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    text = output_of([executable])
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_rocm():
    if platform.system() != "Linux":
        return None

    version_files = (
        "/opt/rocm/.info/version",
        "/opt/rocm/.info/version-dev",
    )
    for filename in version_files:
        path = Path(filename)
        if path.is_file():
            version = parse_version(path.read_text(encoding="utf-8", errors="ignore"))
            if version:
                return version

    probes = (
        ("hipconfig", ["--version"]),
        ("hipcc", ["--version"]),
        ("rocminfo", []),
    )
    for name, args in probes:
        executable = shutil.which(name)
        if not executable:
            continue
        text = output_of([executable] + args)
        for pattern_name in (
            r"(?:ROCm|HIP)(?:\s+version)?\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)",
            r"version\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)",
        ):
            match = re.search(pattern_name, text, flags=re.IGNORECASE)
            if match:
                return parse_version(match.group(1))
    return None


def describe_accelerators():
    found = []
    cuda = detect_nvidia_cuda()
    if cuda:
        found.append("NVIDIA GPU (CUDA %d.%d)" % cuda)
    rocm = detect_rocm()
    if rocm:
        found.append("AMD GPU (ROCm %s)" % ".".join(map(str, rocm)))
    if xpu_runtime_present():
        found.append("Intel GPU (XPU)")
    if platform.system() == "Darwin":
        found.append("Apple silicon (MPS) if available")
    return ", ".join(found) or "none found, using CPU"


def xpu_runtime_present():
    if platform.system() not in ("Linux", "Windows"):
        return False
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return False
    return bool(shutil.which("xpu-smi") or shutil.which("sycl-ls"))


def version_at_most(channel, runtime):
    if channel is None or runtime is None:
        return False
    # If runtime has no patch component, compare only major/minor. This lets a
    # channel such as 6.2.4 be considered for a runtime reported simply as 6.2.
    if len(runtime) == 2:
        return channel[:2] <= runtime[:2]
    width = max(len(channel), len(runtime))
    channel_padded = channel + (0,) * (width - len(channel))
    runtime_padded = runtime + (0,) * (width - len(runtime))
    return channel_padded <= runtime_padded


def backend_candidates(channels):
    system = platform.system()
    candidates = []

    if system == "Darwin":
        # Official macOS wheels come from PyPI and expose MPS when the machine
        # and macOS version support it.
        candidates.append(("macos", None))
        return candidates

    cuda_runtime = detect_nvidia_cuda()
    if cuda_runtime:
        cuda_channels = []
        for name in channels:
            version = cuda_channel_version(name)
            if version and version <= cuda_runtime:
                cuda_channels.append((version, name))
        cuda_channels.sort(reverse=True)
        for _version, name in cuda_channels:
            candidates.append(("cuda", name))
        if not cuda_channels:
            # If HTML discovery was blocked but pip still has network access,
            # PyPI is worth trying before falling back to CPU.
            candidates.append(("default", None))

    rocm_runtime = detect_rocm()
    if rocm_runtime:
        rocm_channels = []
        for name in channels:
            version = rocm_channel_version(name)
            if version and version_at_most(version, rocm_runtime):
                rocm_channels.append((version, name))
        rocm_channels.sort(reverse=True)
        for _version, name in rocm_channels:
            candidates.append(("rocm", name))

    if xpu_runtime_present():
        # xpu is a stable channel name rather than a version-specific table.
        if "xpu" in channels or not channels:
            candidates.append(("xpu", "xpu"))

    # CPU is always the final accelerator-independent choice. The generic CPU
    # channel name is stable; if it lacks a wheel for this OS/architecture,
    # the final PyPI attempt lets pip use whatever the platform supports.
    candidates.append(("cpu", "cpu"))
    candidates.append(("default", None))

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def install_torch(python, kind, channel):
    command = [python, "-m", "pip", "install", "--upgrade", "torch"]
    if channel:
        command.extend(["--index-url", PYTORCH_ROOT.rstrip("/") + "/" + channel])
    run(command)
    return verify_torch_backend(python, kind)


def verify_torch_backend(python, kind):
    script = r'''
import sys
import torch

kind = sys.argv[1]
selected = "cpu"

def compute(device):
    a = torch.tensor([1.0, 2.0], device=device)
    b = torch.tensor([3.0, 4.0], device=device)
    value = (a + b).sum().item()
    if abs(value - 10.0) > 1e-6:
        raise RuntimeError("tensor verification failed on %s" % device)

if kind == "cuda":
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("CUDA build installed but CUDA is not usable")
    compute("cuda")
    selected = "cuda"
elif kind == "rocm":
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("ROCm build installed but ROCm is not usable")
    compute("cuda")
    selected = "rocm"
elif kind == "xpu":
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU build installed but Intel XPU is not usable")
    compute("xpu")
    selected = "xpu"
elif kind == "macos":
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        compute("mps")
        selected = "mps"
    else:
        compute("cpu")
        selected = "cpu"
elif kind == "default":
    if torch.cuda.is_available() and torch.version.hip:
        compute("cuda")
        selected = "rocm"
    elif torch.cuda.is_available() and torch.version.cuda:
        compute("cuda")
        selected = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        compute("xpu")
        selected = "xpu"
    else:
        compute("cpu")
        selected = "cpu"
else:
    compute("cpu")
    selected = "cpu"

print("TORCH_VERSION=" + str(torch.__version__))
print("SELECTED_BACKEND=" + selected)
'''
    result = run([python, "-c", script, kind], capture=True, check=False)
    if result.returncode != 0:
        if result.stdout:
            say(result.stdout.strip())
        return None
    data = {}
    for line in (result.stdout or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data if data.get("TORCH_VERSION") else None


def install_numpy_compatible_with_torch(python):
    run([python, "-m", "pip", "install", "--upgrade", "numpy"])
    if verify_numpy_bridge(python):
        return installed_version(python, "numpy")

    say("Latest NumPy is incompatible with this selected Torch build; trying NumPy < 2")
    run([python, "-m", "pip", "install", "--upgrade", "--force-reinstall", "numpy<2"])
    if not verify_numpy_bridge(python):
        raise RuntimeError("Could not find a NumPy version compatible with selected Torch")
    return installed_version(python, "numpy")


def verify_numpy_bridge(python):
    script = (
        "import numpy, torch; "
        "x=torch.tensor([1.0,2.0]); "
        "a=x.numpy(); "
        "assert a.shape==(2,) and float(a.sum())==3.0; "
        "print(numpy.__version__)"
    )
    result = run([python, "-c", script], capture=True, check=False)
    if result.returncode != 0:
        if result.stdout:
            say(result.stdout.strip())
        return False
    return True


def installed_version(python, package):
    script = (
        "import importlib.metadata as m; "
        "print(m.version(%r))" % package
    )
    result = run([python, "-c", script], capture=True)
    return (result.stdout or "").strip().splitlines()[-1]


def install_project(python, torch_version, numpy_version):
    constraints_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix="spars-constraints-",
            delete=False,
        ) as handle:
            constraints_path = handle.name
            handle.write("torch==%s\n" % torch_version)
            handle.write("numpy==%s\n" % numpy_version)

        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-c",
                constraints_path,
                "-e",
                ".[dev,rl]",
            ],
            cwd=PROJECT_ROOT,
        )
    finally:
        if constraints_path:
            try:
                os.unlink(constraints_path)
            except OSError:
                pass


def project_requirements_without_torch(python):
    """Read the installed project's requirements (base + dev extra), minus torch."""
    script = (
        "import importlib.metadata as m\n"
        "for r in (m.requires(%r) or []):\n"
        "    print(r)\n" % PROJECT_NAME
    )
    result = run([python, "-c", script], capture=True)
    requirements = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        requirement, _sep, marker = line.partition(";")
        requirement = requirement.strip()
        marker = marker.strip()
        if re.match(r"^torch(?![A-Za-z0-9_.-])", requirement, flags=re.IGNORECASE):
            continue
        if "extra" in marker:
            # Keep only the dev extra, which this installer always installs.
            if not re.search(r"extra\s*==\s*['\"]dev['\"]", marker):
                continue
        elif marker:
            requirement = requirement + "; " + marker
        requirements.append(requirement)
    return requirements


def install_project_without_torch(python):
    # --no-deps: pip installs only the project itself here; its own dependencies
    # (torch is not one of them; it is the separate "rl" extra) are installed below,
    # explicitly, from the project's own metadata.
    run([python, "-m", "pip", "install", "--no-deps", "-e", "."], cwd=PROJECT_ROOT)
    requirements = project_requirements_without_torch(python)
    if requirements:
        run([python, "-m", "pip", "install"] + requirements)


def verify_project(python, expected_backend):
    script = r'''
import importlib
import sys

backend = sys.argv[1]
no_torch = backend == "none"

modules = [
    "gymnasium",
    "matplotlib",
    "numpy",
    "pandas",
    "psutil",
    "yaml",
    "textual",
]
if not no_torch:
    modules += ["torch", "spars"]
for name in modules:
    importlib.import_module(name)

if no_torch:
    try:
        importlib.import_module("spars")
    except ImportError as error:
        print("note: spars imported with missing optional dependency: %s" % error)
    print("verification-ok")
    sys.exit(0)

import numpy as np
import torch
x = torch.tensor([1.0, 2.0])
a = x.numpy()
assert isinstance(a, np.ndarray) and float(a.sum()) == 3.0

if backend == "cuda":
    assert torch.cuda.is_available() and torch.version.cuda and not torch.version.hip
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4.0
elif backend == "rocm":
    assert torch.cuda.is_available() and torch.version.hip
    assert (torch.ones(2, device="cuda") + 1).sum().item() == 4.0
elif backend == "xpu":
    assert hasattr(torch, "xpu") and torch.xpu.is_available()
    assert (torch.ones(2, device="xpu") + 1).sum().item() == 4.0
elif backend == "mps":
    assert torch.backends.mps.is_available()
    assert (torch.ones(2, device="mps") + 1).sum().item() == 4.0
else:
    assert (torch.ones(2) + 1).sum().item() == 4.0

print("verification-ok")
'''
    run([python, "-c", script, expected_backend], cwd=PROJECT_ROOT)


def activation_hint():
    if os.name == "nt":
        return r"SPARS-venv\Scripts\activate"
    return "source SPARS-venv/bin/activate"


def main():
    say("SPARS universal setup")
    # Windows reports every x86-64 CPU (Intel too) as "AMD64".
    machine = {"amd64": "x86_64"}.get(platform.machine().lower(), platform.machine())
    say("Platform: %s %s" % (platform.system() or sys.platform, machine))
    say("Accelerator: %s" % describe_accelerators())

    want_torch = decide_install_torch()

    minimum = parse_project_min_python()
    python_version, base_python = find_compatible_python(minimum)
    say(
        "Python: %s (%s)"
        % (format_command(base_python), ".".join(map(str, python_version)))
    )

    if not want_torch:
        say("\nSkipping PyTorch; installing the project without it")
        python = create_venv(base_python)
        install_project_without_torch(python)
        verify_project(python, "none")
        say("\nSetup complete.")
        say("Environment: %s" % VENV)
        say("Backend: none (torch not installed)")
        say("Note: the RL power manager needs PyTorch; add it with: python install.py --torch")
        say("Activate: %s" % activation_hint())
        return

    channels = discover_pytorch_channels()
    if channels:
        say("Discovered %d PyTorch install channels" % len(channels))

    candidates = backend_candidates(channels)
    selected = None
    python = None

    for kind, channel in candidates:
        label = channel or "PyPI"
        say("\nTrying PyTorch backend: %s (%s)" % (kind, label))
        try:
            python = create_venv(base_python)
            info = install_torch(python, kind, channel)
            if not info:
                raise RuntimeError("Installed Torch build did not pass backend verification")

            # A backend is only accepted after the *whole project* installs and
            # verifies.  If Torch/NumPy/dependencies conflict, the next backend
            # gets a completely clean environment instead of leaving setup broken.
            numpy_version = install_numpy_compatible_with_torch(python)
            install_project(python, info["TORCH_VERSION"], numpy_version)
            verify_project(python, info["SELECTED_BACKEND"])

            selected = (info["SELECTED_BACKEND"], info["TORCH_VERSION"])
            say("Selected Torch %s using %s" % (selected[1], selected[0]))
            break
        except Exception as error:
            say("Backend attempt failed: %s" % error)

    if selected is None or python is None:
        raise RuntimeError("No usable PyTorch/project combination could be installed on this machine")

    say("\nSetup complete.")
    say("Environment: %s" % VENV)
    say("Backend: %s" % selected[0])
    say("Activate: %s" % activation_hint())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nSetup cancelled.")
        raise SystemExit(130)
    except Exception as error:
        say("\nSetup failed: %s" % error)
        raise SystemExit(1)
