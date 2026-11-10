# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import psutil
from fastapi.testclient import TestClient

from server.app import create_app
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.services import deep_learning_runner as runner
from server.services.deep_learning import DEEP_LEARNING_PACKAGES, DeepLearningManager

LOCK_PATH = Path(runner.__file__).resolve().parents[1] / "uv.lock"


###############################################################################
def _config(directory: Path) -> AdsmodConfig:
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="core.db"),
        storage_root=directory,
        settings=UserSettings(),
    )


# -----------------------------------------------------------------------------
def _write_operation(directory: Path, **operation: Any) -> None:
    state = directory / "deep_learning" / "state.json"
    base = {"id": "op1", "kind": "install", "status": "running", "phase": "run_uv",
            "started_at": "2026-11-10T10:00:00+00:00", "started_at_epoch": 0.0}
    runner.write_json_atomic(state, {"schema_version": 1, "operation": {**base, **operation}})


###############################################################################
@pytest.mark.parametrize("include_dev", [False, True])
def test_exclusive_packages_never_include_shared_core_dependencies(include_dev: bool) -> None:
    exclusive = runner.exclusive_packages(LOCK_PATH, include_dev=include_dev)
    assert {"torch", "keras", "scikit-learn", "sympy", "h5py"} <= exclusive
    # numpy and scipy are direct core dependencies; packaging/markupsafe/pygments
    # are shared with the core or dev closures and must never be uninstalled.
    assert not {"numpy", "scipy", "pydantic", "fastapi", "pandas", runner.ROOT_PACKAGE} & exclusive
    assert "typing-extensions" not in exclusive
    packages = runner.load_lock_packages(LOCK_PATH)
    root = packages[runner.ROOT_PACKAGE]
    core = {runner.normalize(item["name"]) for item in root["dependencies"]}
    assert core.isdisjoint(exclusive)


###############################################################################
def test_interrupted_operation_is_reported_failed_and_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(dict))
    # The recorded installer PID belongs to a process that no longer exists.
    _write_operation(tmp_path, pid=2_000_000_000, pid_started=1.0)
    manager = DeepLearningManager(_config(tmp_path), is_busy=lambda: False)

    status = manager.status()

    assert status.status == "failed"
    assert status.operation is not None and status.operation.error is not None
    assert status.operation.error.code == "interrupted"
    persisted = json.loads((tmp_path / "deep_learning" / "state.json").read_text("utf-8"))
    assert persisted["operation"]["status"] == "failed"


###############################################################################
def test_live_installer_keeps_status_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(dict))
    _write_operation(tmp_path, pid=os.getpid(), pid_started=psutil.Process().create_time())
    manager = DeepLearningManager(_config(tmp_path), is_busy=lambda: False)

    assert manager.status().status == "installing"
    assert manager.availability()[0] is False


###############################################################################
def test_partial_installation_is_failed_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = {DEEP_LEARNING_PACKAGES[0]: ("1.0", 1)}
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(lambda: partial))
    manager = DeepLearningManager(_config(tmp_path), is_busy=lambda: False)

    status = manager.status()

    assert status.status == "failed"
    assert "missing" in (status.reason or "")


###############################################################################
def test_probe_result_is_cached_until_the_environment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {name: ("1.0", 1) for name in DEEP_LEARNING_PACKAGES}
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(lambda: dict(environment)))
    calls: list[int] = []

    def probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return {"ok": True, "versions": {}, "cuda_available": False, "devices": []}

    monkeypatch.setattr(runner, "run_probe", probe)
    manager = DeepLearningManager(_config(tmp_path), is_busy=lambda: False)

    def settle() -> str:
        deadline = time.monotonic() + 10
        while manager.status().status == "verifying" and time.monotonic() < deadline:
            time.sleep(0.05)
        return manager.status().status

    manager.start()
    assert settle() == "installed"
    manager.start()
    assert settle() == "installed"
    assert len(calls) == 1
    environment["torch"] = ("2.0", 2)  # externally modified package
    manager.start()
    assert settle() == "installed"
    assert len(calls) == 2


###############################################################################
def test_operations_require_confirmation_and_a_restartable_process(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        unconfirmed = client.post("/api/v1/deep-learning/operations", json={"action": "install"})
        assert unconfirmed.status_code == 422
        confirmed = client.post(
            "/api/v1/deep-learning/operations", json={"action": "install", "confirm": True}
        )
        # The test client has no process supervisor, so the app must refuse
        # rather than spawn an installer it cannot hand over to.
        assert confirmed.status_code == 409
        assert "restart itself" in confirmed.json()["detail"]
        assert not (tmp_path / "deep_learning" / "state.json").exists()


###############################################################################
def test_operations_are_blocked_while_jobs_are_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(dict))
    with TestClient(create_app(_config(tmp_path))) as client:
        client.app.state.request_shutdown = lambda: None
        client.app.state.deep_learning.is_busy = lambda: True
        response = client.post(
            "/api/v1/deep-learning/operations", json={"action": "install", "confirm": True}
        )
        assert response.status_code == 409
        assert "Wait for active jobs" in response.json()["detail"]


###############################################################################
@pytest.mark.parametrize(
    ("output", "code"),
    [
        ("error: Failed to download `torch`: error sending request", "network"),
        ("error: failed to write file (os error 112)", "disk_full"),
        ("error: failed to remove file: Access is denied. (os error 5)", "file_locked"),
        ("  x No solution found when resolving dependencies", "resolution"),
        ("something unexpected", "uv_failed"),
    ],
)
def test_uv_failures_are_classified_with_actionable_hints(output: str, code: str) -> None:
    assert runner.Worker.classify(output) == code
    assert code in runner.HINTS


###############################################################################
def test_profile_update_preserves_launcher_state(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    state_path = venv / ".adsmod-dependency-state.json"
    runner.write_json_atomic(
        state_path,
        {"schemaVersion": 1, "fingerprint": "abc", "pythonVersion": "3.14.7",
         "installationType": "Standard", "featureSet": "Base"},
    )
    worker = runner.Worker.__new__(runner.Worker)
    worker.venv_state_path = state_path
    worker.log_path = tmp_path / "install.log"

    worker.update_profile(runner.PROFILE_ML)

    updated = json.loads(state_path.read_text("utf-8"))
    assert updated["featureSet"] == "ML"
    assert updated["fingerprint"] == "abc"
    assert updated["installationType"] == "Standard"
    assert worker.previous_profile() == runner.PROFILE_ML
