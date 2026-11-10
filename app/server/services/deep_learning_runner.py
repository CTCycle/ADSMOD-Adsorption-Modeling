# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Standalone, stdlib-only worker for the optional deep learning dependencies.

The module is deliberately free of ``server`` imports because it runs in two
situations where the application package is not importable:

* ``probe``: executed by the backend with the virtual-environment interpreter
  to verify that torch, keras and scikit-learn really import.
* ``run``: executed detached by the portable (embedded) interpreter after the
  backend has exited, so the virtual environment can be modified safely.

The lock-file helpers are also imported by the backend to build install plans.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROBE_MARKER = "ADSMOD_DL_PROBE:"
OPTIONAL_PACKAGES = ("torch", "keras", "scikit-learn")
PROFILE_BASE = "Base"
PROFILE_ML = "ML"
ROOT_PACKAGE = "adsmod-backend-workspace"
EXTRA_NAME = "ml"
STATE_SCHEMA_VERSION = 1
BACKEND_EXIT_TIMEOUT_SECONDS = 90
HEALTH_TIMEOUT_SECONDS = 90
LOCK_RETRIES = 3
FILE_RETRIES = 6
HINTS = {
    "network": "Check the internet connection and retry. Partially downloaded files are discarded safely.",
    "disk_full": "Free disk space on the drive that holds ADSMOD (about 8 GB) and retry.",
    "resolution": "The locked dependency set could not be resolved. Run 'Install/Update dependencies' from the launcher menu.",
    "file_locked": "Close any program using the ADSMOD environment (terminals, editors, other ADSMOD instances) and retry.",
    "validation": "The installed packages failed the import check. Use Repair/Reinstall; if it persists, review install.log and the NVIDIA driver.",
    "uv_failed": "Review install.log for the uv error output, then retry or use Repair/Reinstall.",
    "interrupted": "The installation was interrupted before it finished. Use Repair/Reinstall to complete or Uninstall to return to the lightweight setup.",
}
ERROR_PATTERNS = (
    ("disk_full", re.compile(r"os error 112|no space left|not enough space", re.I)),
    ("file_locked", re.compile(r"os error 32|os error 5\b|being used by another process|access is denied", re.I)),
    ("network", re.compile(r"error sending request|dns error|connection (?:reset|refused|closed)|timed out|failed to download|tcp connect|network|offline", re.I)),
    ("resolution", re.compile(r"no solution found|failed to resolve|resolution failed|lockfile .* needs to be updated", re.I)),
)


###############################################################################
class RunnerError(RuntimeError):
    def __init__(self, code: str, summary: str) -> None:
        super().__init__(summary)
        self.code = code
        self.summary = summary


###############################################################################
def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# -----------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for attempt in range(FILE_RETRIES):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                # Windows refuses to replace a file a reader currently has open.
                if attempt == FILE_RETRIES - 1:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
def read_json(path: Path) -> dict[str, Any] | None:
    for attempt in range(FILE_RETRIES):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            # A concurrent atomic replace briefly locks the file on Windows.
            time.sleep(0.05)
            continue
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


###############################################################################
# Lock-file analysis
###############################################################################
def load_lock_packages(lock_path: Path) -> dict[str, dict[str, Any]]:
    document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    return {normalize(package["name"]): package for package in document.get("package", [])}


# -----------------------------------------------------------------------------
def _closure(packages: dict[str, dict[str, Any]], seeds: list[dict[str, Any]]) -> set[str]:
    """Name-level dependency closure; markers are ignored on purpose.

    Ignoring markers can only over-approximate a closure, so the core closure
    can only grow and an optional package is never mistaken for a core one.
    """
    seen: set[str] = set()
    queue: deque[dict[str, Any]] = deque(seeds)
    while queue:
        entry = queue.popleft()
        name = normalize(entry["name"])
        package = packages.get(name)
        if package is None or name in seen:
            continue
        seen.add(name)
        queue.extend(package.get("dependencies", []))
        optional = package.get("optional-dependencies", {})
        for extra in entry.get("extra", []):
            queue.extend(optional.get(extra, []))
    return seen


# -----------------------------------------------------------------------------
def exclusive_packages(lock_path: Path, *, include_dev: bool) -> set[str]:
    """Packages reachable only through the ``ml`` extra of the project."""
    packages = load_lock_packages(lock_path)
    root = packages[ROOT_PACKAGE]
    core_seeds = list(root.get("dependencies", []))
    if include_dev:
        core_seeds += root.get("dev-dependencies", {}).get("dev", [])
    core = _closure(packages, core_seeds) | {ROOT_PACKAGE}
    optional = _closure(packages, list(root.get("optional-dependencies", {}).get(EXTRA_NAME, [])))
    return optional - core


# -----------------------------------------------------------------------------
def wheel_for_windows(package: dict[str, Any]) -> dict[str, Any] | None:
    wheels = package.get("wheels", [])
    for marker in ("cp314-cp314-win_amd64", "win_amd64", "none-any"):
        for wheel in wheels:
            if marker in str(wheel.get("url", "")):
                return wheel
    return None


###############################################################################
# Runtime probe
###############################################################################
def probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "versions": {},
        "cuda_runtime": None,
        "cuda_available": False,
        "devices": [],
        "error": None,
    }
    try:
        os.environ.setdefault("KERAS_BACKEND", "torch")
        os.environ.setdefault("MPLBACKEND", "Agg")
        import keras  # noqa: PLC0415
        import sklearn  # noqa: PLC0415
        import torch  # noqa: PLC0415

        result["versions"] = {
            "torch": str(torch.__version__),
            "keras": str(keras.__version__),
            "scikit-learn": str(sklearn.__version__),
        }
        if float((torch.ones(2) * 2).sum().item()) != 4.0:
            raise RuntimeError("torch CPU tensor arithmetic returned a wrong result")
        result["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            if float((torch.ones(2, device="cuda") * 2).sum().item()) != 4.0:
                raise RuntimeError("torch CUDA tensor arithmetic returned a wrong result")
            result["cuda_available"] = True
            result["devices"] = [
                str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
            ]
        result["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# -----------------------------------------------------------------------------
def run_probe(python: str, *, timeout: float = 240.0) -> dict[str, Any]:
    """Run ``probe`` with the given interpreter and return its parsed result."""
    command = [python, "-I", str(Path(__file__).resolve()), "probe"]
    try:
        completed = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Import check timed out after {int(timeout)} s"}
    except OSError as exc:
        return {"ok": False, "error": f"Import check could not start: {exc}"}
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                return json.loads(line[len(PROBE_MARKER) :])
            except ValueError:
                break
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return {
        "ok": False,
        "error": detail[-1] if detail else f"Import check exited with code {completed.returncode}",
    }


###############################################################################
# Worker
###############################################################################
class Worker:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.kind: str = plan["kind"]
        self.operation_id: str = plan["operation_id"]
        self.state_path = Path(plan["state_file"])
        self.log_path = Path(plan["log_file"])
        self.venv = Path(plan["venv_dir"])
        self.venv_python = self.venv / "Scripts" / "python.exe"
        self.venv_state_path = self.venv / ".adsmod-dependency-state.json"
        self.project_dir = Path(plan["project_dir"])
        self.uv = plan["uv_exe"]
        self.python_exe = plan["python_exe"]
        self.exclusive = exclusive_packages(
            self.project_dir / "uv.lock", include_dev=self._installation_type() == "Development"
        )
        self.warnings: list[str] = []

    # -------------------------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")

    # -------------------------------------------------------------------------
    def _installation_type(self) -> str:
        state = read_json(Path(self.plan["venv_dir"]) / ".adsmod-dependency-state.json") or {}
        return "Development" if state.get("installationType") == "Development" else "Standard"

    # -------------------------------------------------------------------------
    def update_operation(self, **changes: Any) -> None:
        state = read_json(self.state_path) or {"schema_version": STATE_SCHEMA_VERSION}
        operation = state.get("operation") or {}
        if operation.get("id") != self.operation_id:
            return
        operation.update(changes)
        state["operation"] = operation
        write_json_atomic(self.state_path, state)

    # -------------------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        self.log(f"Phase: {phase}")
        self.update_operation(phase=phase)

    # -------------------------------------------------------------------------
    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["UV_PROJECT_ENVIRONMENT"] = str(self.venv)
        env.setdefault("UV_CACHE_DIR", str(Path(self.plan["repo_root"]) / "runtimes" / "cache" / "uv"))
        env["PYTHONPATH"] = str(Path(self.plan["repo_root"]) / "app")
        env.pop("PYTHONHOME", None)
        return env

    # -------------------------------------------------------------------------
    def run_command(self, command: list[str], *, cwd: Path | None = None) -> str:
        """Run a command with output appended to the log; return its captured tail."""
        self.log("$ " + " ".join(command))
        with self.log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603
                command,
                cwd=str(cwd or self.project_dir),
                env=self.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            tail: deque[str] = deque(maxlen=60)
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                tail.append(line.rstrip())
            code = process.wait()
        output = "\n".join(tail)
        if code != 0:
            raise RunnerError(self.classify(output), self.summarize(output, code))
        return output

    # -------------------------------------------------------------------------
    @staticmethod
    def classify(output: str) -> str:
        for code, pattern in ERROR_PATTERNS:
            if pattern.search(output):
                return code
        return "uv_failed"

    # -------------------------------------------------------------------------
    @staticmethod
    def summarize(output: str, exit_code: int) -> str:
        lines = [line for line in output.splitlines() if line.strip()]
        error_lines = [line for line in lines if "error" in line.lower()]
        detail = (error_lines or lines or [""])[-1].strip()
        return f"uv exited with code {exit_code}: {detail}" if detail else f"uv exited with code {exit_code}"

    # -------------------------------------------------------------------------
    def with_lock_retries(self, action: Any) -> Any:
        for attempt in range(1, LOCK_RETRIES + 1):
            try:
                return action()
            except RunnerError as exc:
                if exc.code != "file_locked" or attempt == LOCK_RETRIES:
                    raise
                self.log(f"A file is locked; retrying in {attempt * 3} s ({attempt}/{LOCK_RETRIES})")
                time.sleep(attempt * 3)
        raise AssertionError("unreachable")

    # -------------------------------------------------------------------------
    def sync_arguments(self) -> list[str]:
        group = ["--group", "dev"] if self._installation_type() == "Development" else ["--no-dev"]
        return [
            self.uv, "sync", "--locked", "--inexact", "--python", self.python_exe, *group,
        ]

    # -------------------------------------------------------------------------
    def installed_names(self) -> set[str]:
        command = [self.uv, "pip", "list", "--python", str(self.venv_python), "--format", "json"]
        completed = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, env=self.environment(), check=False
        )
        if completed.returncode != 0:
            raise RunnerError("uv_failed", f"Could not list installed packages: {completed.stderr.strip()}")
        return {normalize(item["name"]) for item in json.loads(completed.stdout)}

    # -------------------------------------------------------------------------
    def install(self, *, reinstall: bool) -> None:
        command = [*self.sync_arguments(), "--extra", EXTRA_NAME]
        if reinstall:
            for name in sorted(self.exclusive & self.installed_names()):
                command += ["--reinstall-package", name]
        self.with_lock_retries(lambda: self.run_command(command))

    # -------------------------------------------------------------------------
    def uninstall(self) -> None:
        targets = sorted(self.exclusive & self.installed_names())
        if not targets:
            self.log("No optional deep learning packages are installed.")
            return
        self.log(f"Removing {len(targets)} packages exclusive to the deep learning extra.")
        command = [self.uv, "pip", "uninstall", "--python", str(self.venv_python), *targets]
        self.with_lock_retries(lambda: self.run_command(command))

    # -------------------------------------------------------------------------
    def validate_core(self) -> None:
        self.run_command([self.uv, "pip", "check", "--python", str(self.venv_python)])
        self.run_command(
            [str(self.venv_python), "-c", "import server.app"],
            cwd=Path(self.plan["repo_root"]) / "app",
        )

    # -------------------------------------------------------------------------
    def validate_deep_learning(self) -> dict[str, Any]:
        result = run_probe(str(self.venv_python))
        self.log(f"Import check: {json.dumps(result)}")
        if not result.get("ok"):
            raise RunnerError("validation", f"Deep learning import check failed: {result.get('error')}")
        if not result.get("cuda_available") and self.nvidia_gpu_present():
            self.warnings.append(
                "An NVIDIA GPU was detected but PyTorch cannot use CUDA. Training will run on the CPU; "
                "update the NVIDIA driver (ADSMOD never changes drivers)."
            )
        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def nvidia_gpu_present() -> bool:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False
        completed = subprocess.run(  # noqa: S603
            [executable, "-L"], capture_output=True, text=True, timeout=15, check=False
        )
        return completed.returncode == 0 and "GPU" in completed.stdout

    # -------------------------------------------------------------------------
    def update_profile(self, profile: str) -> None:
        state = read_json(self.venv_state_path)
        if state is None:
            self.log("No launcher dependency state file was found; profile not recorded.")
            return
        state["featureSet"] = profile
        write_json_atomic(self.venv_state_path, state)

    # -------------------------------------------------------------------------
    def previous_profile(self) -> str:
        state = read_json(self.venv_state_path) or {}
        return PROFILE_ML if state.get("featureSet") == PROFILE_ML else PROFILE_BASE

    # -------------------------------------------------------------------------
    def wait_for_backend(self) -> None:
        pid = int(self.plan["backend_pid"])
        deadline = time.monotonic() + BACKEND_EXIT_TIMEOUT_SECONDS
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.5)
        if pid_alive(pid):
            self.log(f"Backend PID {pid} did not exit in time; terminating its process tree.")
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)  # noqa: S603, S607
            time.sleep(2)

    # -------------------------------------------------------------------------
    def recover(self, target: str) -> bool:
        """Converge the environment to ``target`` after a failed operation."""
        self.set_phase("recover")
        try:
            if target == PROFILE_ML:
                self.install(reinstall=False)
            else:
                self.uninstall()
            self.validate_core()
        except RunnerError as exc:
            self.log(f"Recovery failed: {exc.summary}")
            return False
        return True

    # -------------------------------------------------------------------------
    def execute(self) -> None:
        previous = self.previous_profile()
        failure: RunnerError | None = None
        validated: dict[str, Any] | None = None
        try:
            self.set_phase("wait_backend_exit")
            self.wait_for_backend()
            self.set_phase("run_uv")
            if self.kind == "uninstall":
                self.uninstall()
                self.set_phase("validate")
                self.validate_core()
                self.update_profile(PROFILE_BASE)
            else:
                self.install(reinstall=self.kind == "repair")
                self.set_phase("validate")
                self.validate_core()
                validated = self.validate_deep_learning()
                self.update_profile(PROFILE_ML)
        except RunnerError as exc:
            failure = exc
            self.log(f"Operation failed [{exc.code}]: {exc.summary}")
        except Exception as exc:  # noqa: BLE001
            failure = RunnerError("uv_failed", f"{type(exc).__name__}: {exc}")
            self.log(f"Unexpected failure: {failure.summary}")
        if failure is not None:
            recovered = self.recover(previous)
            hint = HINTS.get(failure.code, HINTS["uv_failed"])
            if not recovered:
                hint += " The previous environment could not be restored automatically: run 'Install/Update dependencies' from the launcher menu."
            self.update_operation(
                status="failed",
                phase="done",
                finished_at=utc_now(),
                error={"code": failure.code, "summary": failure.summary, "hint": hint},
            )
        else:
            changes: dict[str, Any] = {"status": "succeeded", "phase": "done", "finished_at": utc_now(), "warnings": self.warnings}
            self.update_operation(**changes)
            if validated is not None:
                self.record_validation(validated)
        self.restart()

    # -------------------------------------------------------------------------
    def record_validation(self, result: dict[str, Any]) -> None:
        state = read_json(self.state_path) or {"schema_version": STATE_SCHEMA_VERSION}
        state["last_validated"] = {
            "at": utc_now(),
            "versions": result.get("versions", {}),
            "cuda_runtime": result.get("cuda_runtime"),
            "cuda_available": bool(result.get("cuda_available")),
            "devices": result.get("devices", []),
        }
        write_json_atomic(self.state_path, state)

    # -------------------------------------------------------------------------
    def restart(self) -> None:
        spec = self.plan["restart"]
        self.set_phase("restart")
        creation = 0
        if sys.platform == "win32":
            creation = (
                subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
                if spec["new_console"]
                else subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        environment = {**os.environ, **spec.get("env", {})}
        environment.pop("PYTHONHOME", None)
        try:
            if spec["new_console"]:
                subprocess.Popen(  # noqa: S603
                    spec["command"], cwd=spec["cwd"], env=environment, creationflags=creation
                )
            else:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    subprocess.Popen(  # noqa: S603
                        spec["command"],
                        cwd=spec["cwd"],
                        env=environment,
                        creationflags=creation,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    )
        except OSError as exc:
            self.log(f"Restart failed to start: {exc}")
            self.update_operation(restart={"attempted": True, "healthy": False})
            return
        healthy = self.wait_for_health(spec["health_url"])
        self.log("ADSMOD restarted." if healthy else "ADSMOD did not become healthy; start it manually with start_on_windows.bat.")
        self.update_operation(restart={"attempted": True, "healthy": healthy})

    # -------------------------------------------------------------------------
    def wait_for_health(self, url: str) -> bool:
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        # The backend is local: never route the health probe through a proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            try:
                with opener.open(url, timeout=2) as response:  # noqa: S310
                    if 200 <= response.status < 300:
                        return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1)
        return False


# -----------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes  # noqa: PLC0415

    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


###############################################################################
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe")
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "probe":
        print(PROBE_MARKER + json.dumps(probe()), flush=True)
        return 0
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    worker = Worker(plan)
    worker.execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
