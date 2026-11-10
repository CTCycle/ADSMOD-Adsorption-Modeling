# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Single source of truth for optional deep learning availability.

Availability is always derived from the real Python environment: the optional
packages must be present and an actual import probe, executed in a subprocess so
the API process never loads torch, must succeed. The persisted operation state
only records lifecycle information (running, failed, interrupted) and recovery
details; it is never treated as proof that the dependencies work.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from server.common.path import resolve_storage_root
from server.common.utils.logger import logger
from server.configurations.bootstrap import REPOSITORY_ROOT
from server.configurations.settings import AdsmodConfig
from server.domain.deep_learning import (
    DeepLearningAction,
    DeepLearningCheck,
    DeepLearningEnvironment,
    DeepLearningError,
    DeepLearningOperation,
    DeepLearningPlanResponse,
    DeepLearningStatusName,
    DeepLearningStatusResponse,
)
from server.services import deep_learning_runner as runner

MIN_FREE_BYTES = 8 * 1024**3
ESTIMATED_TORCH_BYTES = 3 * 1024**3
CHECK_INTERVAL_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 240.0
SPAWN_GRACE_SECONDS = 30.0
SHUTDOWN_DELAY_SECONDS = 1.0
MIN_DRIVER_CUDA = (13, 0)
STATE_SCHEMA_VERSION = runner.STATE_SCHEMA_VERSION
RUNNER_PATH = Path(runner.__file__).resolve()
DEEP_LEARNING_PACKAGES = runner.OPTIONAL_PACKAGES
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


###############################################################################
class DeepLearningOperationError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


###############################################################################
@dataclass
class _Probe:
    fingerprint: dict[str, tuple[str, int]] | None = None
    result: dict[str, Any] | None = None
    running: bool = False
    completed_at: str | None = None


###############################################################################
@dataclass
class _NvidiaInfo:
    gpu_names: list[str] = field(default_factory=list)
    driver_version: str | None = None
    cuda_level: str | None = None


###############################################################################
class DeepLearningManager:
    def __init__(self, config: AdsmodConfig, *, is_busy: Callable[[], bool]) -> None:
        self.config = config
        self.is_busy = is_busy
        self.directory = resolve_storage_root(config) / "deep_learning"
        self.state_path = self.directory / "state.json"
        self.log_path = self.directory / "install.log"
        self.plan_path = self.directory / "plan.json"
        self.venv_dir = Path(sys.prefix)
        self.project_dir = Path(runner.__file__).resolve().parents[1]
        self.lock_path = self.project_dir / "uv.lock"
        self._lock = threading.RLock()
        self._fingerprint: dict[str, tuple[str, int]] | None = None
        self._checked_at = float("-inf")
        self._probe = _Probe()
        self._nvidia_cache: tuple[float, _NvidiaInfo] | None = None

    ###########################################################################
    # Environment inspection
    ###########################################################################
    @staticmethod
    def _read_fingerprint() -> dict[str, tuple[str, int]]:
        """Version and metadata mtime of every optional package that is installed."""
        importlib.invalidate_caches()
        fingerprint: dict[str, tuple[str, int]] = {}
        for name in DEEP_LEARNING_PACKAGES:
            try:
                distribution = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            modified = 0
            for file in distribution.files or ():
                if file.name == "METADATA":
                    try:
                        modified = distribution.locate_file(file).stat().st_mtime_ns
                    except OSError:
                        modified = 0
                    break
            fingerprint[name] = (distribution.version, modified)
        return fingerprint

    # -------------------------------------------------------------------------
    def _nvidia(self) -> _NvidiaInfo:
        now = time.monotonic()
        if self._nvidia_cache is not None and now - self._nvidia_cache[0] < CHECK_INTERVAL_SECONDS:
            return self._nvidia_cache[1]
        info = _NvidiaInfo()
        executable = shutil.which("nvidia-smi")
        if executable is not None:
            try:
                query = subprocess.run(  # noqa: S603
                    [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                for line in query.stdout.splitlines():
                    name, _, driver = line.partition(",")
                    if name.strip():
                        info.gpu_names.append(name.strip())
                        info.driver_version = info.driver_version or driver.strip() or None
                summary = subprocess.run(  # noqa: S603
                    [executable], capture_output=True, text=True, timeout=15, check=False
                )
                match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", summary.stdout)
                info.cuda_level = match.group(1) if match else None
            except (OSError, subprocess.SubprocessError):
                logger.warning("nvidia-smi could not be queried for deep learning diagnostics.")
        self._nvidia_cache = (now, info)
        return info

    ###########################################################################
    # Probe lifecycle
    ###########################################################################
    def start(self) -> None:
        """Begin background verification of an existing installation."""
        with self._lock:
            self._refresh(force=True)

    # -------------------------------------------------------------------------
    def refresh(self) -> None:
        """Invalidate cached verification and re-probe the real environment."""
        with self._lock:
            self._probe = _Probe()
            self._refresh(force=True)

    # -------------------------------------------------------------------------
    def _refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._checked_at < CHECK_INTERVAL_SECONDS:
            return
        self._checked_at = now
        fingerprint = self._read_fingerprint()
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._probe = _Probe()
        complete = len(fingerprint) == len(DEEP_LEARNING_PACKAGES)
        if complete and not self._probe.running and self._probe.fingerprint != fingerprint:
            self._probe = _Probe(fingerprint=fingerprint, running=True)
            threading.Thread(
                target=self._run_probe, args=(fingerprint,), name="adsmod-dl-probe", daemon=True
            ).start()

    # -------------------------------------------------------------------------
    def _run_probe(self, fingerprint: dict[str, tuple[str, int]]) -> None:
        result = runner.run_probe(sys.executable, timeout=PROBE_TIMEOUT_SECONDS)
        with self._lock:
            if self._probe.fingerprint != fingerprint:
                return
            self._probe = _Probe(
                fingerprint=fingerprint,
                result=result,
                running=False,
                completed_at=runner.utc_now(),
            )
        if not result.get("ok"):
            logger.warning("Deep learning verification failed: %s", result.get("error"))

    ###########################################################################
    # Persisted lifecycle state
    ###########################################################################
    def _read_state(self) -> dict[str, Any]:
        return runner.read_json(self.state_path) or {"schema_version": STATE_SCHEMA_VERSION}

    # -------------------------------------------------------------------------
    @staticmethod
    def _runner_alive(operation: dict[str, Any]) -> bool:
        pid = operation.get("pid")
        if pid is None:
            started = operation.get("started_at_epoch", 0)
            return time.time() - float(started) < SPAWN_GRACE_SECONDS
        try:
            process = psutil.Process(int(pid))
            return abs(process.create_time() - float(operation.get("pid_started", 0))) < 2.0 and (
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            )
        except (psutil.Error, ValueError):
            return False

    # -------------------------------------------------------------------------
    def _operation(self) -> dict[str, Any] | None:
        """Return the last operation, converting a dead running one into a failure."""
        state = self._read_state()
        operation = state.get("operation")
        if not isinstance(operation, dict):
            return None
        if operation.get("status") == "running" and not self._runner_alive(operation):
            operation.update(
                status="failed",
                phase="done",
                finished_at=runner.utc_now(),
                error={
                    "code": "interrupted",
                    "summary": f"The {operation.get('kind')} operation was interrupted before it finished.",
                    "hint": runner.HINTS["interrupted"],
                },
            )
            state["operation"] = operation
            runner.write_json_atomic(self.state_path, state)
        return operation

    ###########################################################################
    # Status
    ###########################################################################
    @staticmethod
    def platform_supported() -> bool:
        return sys.platform == "win32" and platform.machine().upper() in {"AMD64", "X86_64"}

    # -------------------------------------------------------------------------
    def status(self) -> DeepLearningStatusResponse:
        with self._lock:
            self._refresh()
            fingerprint = self._fingerprint or {}
            probe = self._probe
            operation = self._operation()
        name, reason = self._derive(operation, fingerprint, probe)
        result = probe.result or {}
        nvidia = self._nvidia()
        warnings = list(operation.get("warnings", [])) if operation and name == "installed" else []
        if name == "installed" and not result.get("cuda_available") and nvidia.gpu_names:
            message = (
                "An NVIDIA GPU was detected but PyTorch cannot use CUDA; training will run on the CPU. "
                "Update the NVIDIA driver (ADSMOD never changes drivers)."
            )
            if message not in warnings:
                warnings.append(message)
        versions = result.get("versions") or {key: value[0] for key, value in fingerprint.items()}
        environment = DeepLearningEnvironment(
            python_version=platform.python_version(),
            platform=f"{sys.platform}/{platform.machine()}",
            torch_version=versions.get("torch"),
            keras_version=versions.get("keras"),
            scikit_learn_version=versions.get("scikit-learn"),
            cuda_runtime=result.get("cuda_runtime"),
            cuda_available=result.get("cuda_available") if probe.result else None,
            devices=list(result.get("devices", [])),
            gpu_names=nvidia.gpu_names,
            driver_version=nvidia.driver_version,
            driver_cuda_level=nvidia.cuda_level,
        )
        return DeepLearningStatusResponse(
            status=name,
            available=name == "installed",
            reason=reason,
            warnings=warnings,
            environment=environment,
            operation=self._operation_model(operation),
            last_validated_at=probe.completed_at if probe.result and probe.result.get("ok") else None,
            log_path=str(self.log_path),
        )

    # -------------------------------------------------------------------------
    def _derive(
        self,
        operation: dict[str, Any] | None,
        fingerprint: dict[str, tuple[str, int]],
        probe: _Probe,
    ) -> tuple[DeepLearningStatusName, str | None]:
        if operation and operation.get("status") == "running":
            return "installing", f"The {operation.get('kind')} operation is running."
        failure = operation.get("error") if operation and operation.get("status") == "failed" else None
        failed_kind = operation.get("kind") if failure else None
        missing = [name for name in DEEP_LEARNING_PACKAGES if name not in fingerprint]
        if len(missing) == len(DEEP_LEARNING_PACKAGES):
            if not self.platform_supported():
                return "incompatible", "Deep learning packages are only available for Windows x64."
            if failure and failed_kind != "uninstall":
                return "failed", str(failure.get("summary"))
            return "not_installed", None
        if missing:
            return "failed", f"Incomplete installation: missing {', '.join(missing)}."
        if probe.result is None:
            return "verifying", "Verifying the installed packages."
        if not probe.result.get("ok"):
            return "failed", str(probe.result.get("error") or "The import check failed.")
        return "installed", None

    # -------------------------------------------------------------------------
    @staticmethod
    def _operation_model(operation: dict[str, Any] | None) -> DeepLearningOperation | None:
        if operation is None:
            return None
        error = operation.get("error")
        restart = operation.get("restart") or {}
        return DeepLearningOperation(
            id=str(operation["id"]),
            kind=operation["kind"],
            status=operation["status"],
            phase=operation.get("phase"),
            started_at=str(operation["started_at"]),
            finished_at=operation.get("finished_at"),
            error=DeepLearningError(**error) if error else None,
            warnings=list(operation.get("warnings", [])),
            restart_healthy=restart.get("healthy"),
        )

    # -------------------------------------------------------------------------
    def availability(self) -> tuple[bool, str | None]:
        """Cheap check used by the request guard and the capabilities endpoint."""
        with self._lock:
            self._refresh()
            fingerprint = self._fingerprint or {}
            probe = self._probe
            operation = self._operation()
        name, reason = self._derive(operation, fingerprint, probe)
        return name == "installed", reason

    ###########################################################################
    # Plans
    ###########################################################################
    def _uv_executable(self) -> str | None:
        bundled = REPOSITORY_ROOT / "runtimes" / "uv" / "uv.exe"
        if bundled.is_file():
            return str(bundled)
        return shutil.which("uv")

    # -------------------------------------------------------------------------
    def _installation_type(self) -> str:
        state = runner.read_json(self.venv_dir / ".adsmod-dependency-state.json") or {}
        return "Development" if state.get("installationType") == "Development" else "Standard"

    # -------------------------------------------------------------------------
    def _installed_names(self) -> set[str]:
        return {
            runner.normalize(distribution.metadata["Name"])
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        }

    # -------------------------------------------------------------------------
    def _dry_run_install(self, uv: str) -> list[str]:
        command = [
            uv, "sync", "--locked", "--inexact", "--dry-run", "--extra", runner.EXTRA_NAME,
            "--python", str(Path(sys.base_prefix) / "python.exe"),
            *(["--group", "dev"] if self._installation_type() == "Development" else ["--no-dev"]),
        ]
        environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(self.venv_dir)}
        completed = subprocess.run(  # noqa: S603
            command, cwd=self.project_dir, env=environment, capture_output=True, text=True,
            timeout=120, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "uv dry run failed")
        return [
            runner.normalize(match.group(1))
            for match in re.finditer(r"^\s*\+\s+([A-Za-z0-9_.\-]+)==", completed.stdout + completed.stderr, re.M)
        ]

    # -------------------------------------------------------------------------
    def _download_size(self, names: list[str]) -> tuple[int | None, bool]:
        packages = runner.load_lock_packages(self.lock_path)
        total, estimated = 0, False
        for name in names:
            package = packages.get(name)
            wheel = runner.wheel_for_windows(package) if package else None
            if wheel is None:
                continue
            size = wheel.get("size")
            if size is None:
                size = self._head_size(str(wheel.get("url", "")))
            if size is None:
                size = ESTIMATED_TORCH_BYTES
                estimated = True
            total += int(size)
        return (total or None), estimated

    # -------------------------------------------------------------------------
    @staticmethod
    def _head_size(url: str) -> int | None:
        if not url.startswith("https://"):
            return None
        try:
            request = urllib.request.Request(url, method="HEAD")  # noqa: S310
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                length = response.headers.get("Content-Length")
            return int(length) if length else None
        except (urllib.error.URLError, OSError, ValueError):
            return None

    # -------------------------------------------------------------------------
    def plan(self, action: DeepLearningAction) -> DeepLearningPlanResponse:
        current = self.status()
        checks: list[DeepLearningCheck] = []

        def add(name: str, status: str, detail: str) -> None:
            checks.append(DeepLearningCheck(name=name, status=status, detail=detail))  # type: ignore[arg-type]

        add("Python", "ok", f"CPython {platform.python_version()} in the project virtual environment.")
        add("Platform", "ok" if self.platform_supported() else "blocked",
            f"{sys.platform}/{platform.machine()}" + ("" if self.platform_supported() else " is not supported; Windows x64 is required."))
        uv = self._uv_executable()
        add("Package manager", "ok" if uv else "blocked", "uv is available." if uv else "uv was not found; run the launcher setup first.")
        add("Lock file", "ok" if self.lock_path.is_file() else "blocked", "Locked dependency set found." if self.lock_path.is_file() else "uv.lock is missing.")
        valid_states = {
            "install": {"not_installed", "failed"},
            "uninstall": {"installed", "failed"},
            "repair": {"installed", "failed"},
        }
        if current.status not in valid_states[action]:
            add("State", "blocked", f"'{action}' is not available while the status is '{current.status}'.")
        else:
            add("State", "ok", f"Status is '{current.status}'.")
        busy = self.is_busy()
        add("Running jobs", "blocked" if busy else "ok",
            "Wait for active jobs to finish." if busy else "No jobs are running.")
        if action != "uninstall":
            free = shutil.disk_usage(self.venv_dir).free
            add("Disk space", "ok" if free >= MIN_FREE_BYTES else "blocked",
                f"{free / 1024**3:.1f} GB free; about {MIN_FREE_BYTES // 1024**3} GB required.")
            nvidia = self._nvidia()
            if not nvidia.gpu_names:
                add("GPU", "warning", "No NVIDIA GPU detected; training will run on the CPU.")
            else:
                level = tuple(int(part) for part in (nvidia.cuda_level or "0").split(".")[:2] if part.isdigit())
                if level < MIN_DRIVER_CUDA:
                    add("GPU", "warning",
                        f"{nvidia.gpu_names[0]}: driver supports CUDA {nvidia.cuda_level or 'unknown'}, "
                        "13.0 is required for GPU training. Update the NVIDIA driver (ADSMOD never changes drivers); CPU training still works.")
                else:
                    add("GPU", "ok", f"{nvidia.gpu_names[0]}, driver {nvidia.driver_version}, CUDA {nvidia.cuda_level}. "
                        "GPU use is confirmed by PyTorch after installation.")
        exclusive = runner.exclusive_packages(self.lock_path, include_dev=self._installation_type() == "Development") if self.lock_path.is_file() else set()
        installed = self._installed_names()
        if action == "install" and uv and not any(check.status == "blocked" for check in checks):
            try:
                packages = sorted(set(self._dry_run_install(uv)) & exclusive)
            except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
                add("Dependency plan", "blocked", f"The dependency plan could not be computed: {exc}")
                packages = []
        else:
            packages = sorted(exclusive & installed)
        size, estimated = (None, False)
        if action != "uninstall" and packages:
            size, estimated = self._download_size(packages)
        allowed = not any(check.status == "blocked" for check in checks)
        return DeepLearningPlanResponse(
            action=action, allowed=allowed, checks=checks, packages=packages,
            download_bytes=size, download_estimated=estimated,
        )

    ###########################################################################
    # Operations
    ###########################################################################
    def _restart_spec(self) -> dict[str, Any]:
        launcher = os.environ.get("ADSMOD_LAUNCHER")
        pwsh = shutil.which("pwsh.exe") or shutil.which("pwsh")
        health = f"{self.config.runtime.backend_url}/health/ready"
        if launcher and Path(launcher).is_file() and pwsh:
            return {
                "command": [pwsh, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", launcher, "-Launch", "-NoBrowser"],
                "cwd": str(REPOSITORY_ROOT), "new_console": True, "health_url": health, "env": {},
            }
        return {
            "command": [sys.executable, "-m", "server.cli"],
            "cwd": str(REPOSITORY_ROOT), "new_console": False, "health_url": health,
            "env": {"PYTHONPATH": str(REPOSITORY_ROOT / "app")},
        }

    # -------------------------------------------------------------------------
    def begin(self, action: DeepLearningAction) -> str:
        """Validate, persist and spawn the detached worker; return the operation id."""
        plan = self.plan(action)
        if not plan.allowed:
            blocked = "; ".join(f"{check.name}: {check.detail}" for check in plan.checks if check.status == "blocked")
            raise DeepLearningOperationError(409, blocked)
        with self._lock:
            current = self._operation()
            if current and current.get("status") == "running":
                raise DeepLearningOperationError(409, "Another deep learning operation is already running.")
            uv = self._uv_executable()
            assert uv is not None
            operation_id = uuid.uuid4().hex[:12]
            now = time.time()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(f"{runner.utc_now()} {action} requested ({operation_id})\n", encoding="utf-8")
            worker_plan = {
                "operation_id": operation_id, "kind": action, "backend_pid": os.getpid(),
                "venv_dir": str(self.venv_dir), "project_dir": str(self.project_dir),
                "repo_root": str(REPOSITORY_ROOT), "uv_exe": uv,
                "python_exe": str(Path(sys.base_prefix) / "python.exe"),
                "state_file": str(self.state_path), "log_file": str(self.log_path),
                "restart": self._restart_spec(),
            }
            runner.write_json_atomic(self.plan_path, worker_plan)
            runner.write_json_atomic(self.state_path, {
                "schema_version": STATE_SCHEMA_VERSION,
                "operation": {
                    "id": operation_id, "kind": action, "status": "running", "phase": "starting",
                    "pid": None, "pid_started": None, "started_at": runner.utc_now(),
                    "started_at_epoch": now, "finished_at": None, "error": None, "warnings": [],
                },
            })
            try:
                process = self._spawn_runner()
            except OSError as exc:
                runner.write_json_atomic(self.state_path, {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "operation": {
                        "id": operation_id, "kind": action, "status": "failed", "phase": "done",
                        "started_at": runner.utc_now(), "finished_at": runner.utc_now(),
                        "error": {"code": "spawn_failed", "summary": f"The installer could not be started: {exc}",
                                  "hint": "Check that the portable Python runtime exists, then retry."},
                    },
                })
                raise DeepLearningOperationError(500, f"The installer could not be started: {exc}") from exc
            state = self._read_state()
            operation = state["operation"]
            try:
                operation["pid"] = process.pid
                operation["pid_started"] = psutil.Process(process.pid).create_time()
            except psutil.Error:
                operation["pid"] = process.pid
            state["operation"] = operation
            runner.write_json_atomic(self.state_path, state)
            return operation_id

    # -------------------------------------------------------------------------
    def _spawn_runner(self) -> subprocess.Popen[bytes]:
        command = [str(Path(sys.base_prefix) / "python.exe"), "-I", str(RUNNER_PATH), "run", "--plan", str(self.plan_path)]
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONHOME"}
        base_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        with self.log_path.open("ab") as log:
            for flags in (base_flags | CREATE_BREAKAWAY_FROM_JOB, base_flags):
                try:
                    return subprocess.Popen(  # noqa: S603
                        command, env=environment, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, creationflags=flags, close_fds=True,
                    )
                except PermissionError:
                    continue
        raise OSError("The installer process could not be created.")

    # -------------------------------------------------------------------------
    def log_tail(self, lines: int) -> list[str]:
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return content[-lines:]


__all__ = ["DeepLearningManager", "DeepLearningOperationError"]
