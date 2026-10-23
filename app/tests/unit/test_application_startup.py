# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

import time
import server.app as app_module
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from fastapi.testclient import TestClient
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.app import create_app


# Core runtime


###############################################################################
def _shutdown_process_runner(stop_event: Any) -> dict[str, str]:
    while not stop_event.is_set():
        time.sleep(0.01)
    return {"stopped": "true"}


###############################################################################
def _temporary_config(directory: str):
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="core.db"),
        storage_root=Path(directory),
        settings=UserSettings(),
    )


###############################################################################
def test_unified_runtime_contracts(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            assert client.get("/health/live").json()["service"] == "backend"
            assert client.get("/health/ready").json()["state"] == "ready"
            capabilities = client.get("/api/v1/system/capabilities").json()
            assert capabilities["features"]["datasets"] is True
            assert capabilities["features"]["machine_learning"] is True, (
                client.app.state.runtime.machine_learning_reason
            )
            assert client.get("/api/v1/system/configuration").status_code == 200
            assert client.get("/api/v1/training/configuration").status_code == 200


###############################################################################
def test_factory_uses_single_backend_port(tmp_path: Path) -> None:
    application = create_app(_temporary_config(str(tmp_path)))
    assert application.state.config.runtime.backend_port > 0
    assert not hasattr(application.state.config.runtime, "ml_port")


###############################################################################
def test_optional_ml_missing_package_is_reported_as_unavailable(
    monkeypatch,
    caplog,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    def missing_package(name: str):
        raise ModuleNotFoundError(
            f"No module named {name!r}", name="server.services.ml_container"
        )

    monkeypatch.setattr(app_module, "import_module", missing_package)
    runtime = SimpleNamespace(
        config=_temporary_config(str(tmp_path)),
        training_data=None,
        machine_learning_available=True,
        machine_learning_reason=None,
    )

    app_module._register_optional_ml(object(), runtime)

    assert runtime.machine_learning_available is False
    assert "not installed" in caplog.text


###############################################################################
def test_optional_ml_bootstrap_failure_is_logged_as_initialization_error(
    monkeypatch,
    caplog,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    def broken_package(name: str):
        raise RuntimeError(f"broken optional package at {name}")

    monkeypatch.setattr(app_module, "import_module", broken_package)
    runtime = SimpleNamespace(
        config=_temporary_config(str(tmp_path)),
        training_data=None,
        machine_learning_available=True,
        machine_learning_reason=None,
    )

    app_module._register_optional_ml(object(), runtime)

    assert runtime.machine_learning_available is False
    assert "initialization failed" in caplog.text


###############################################################################
def test_in_process_snapshot_access_preserves_hash(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            access = client.app.state.runtime.training_data
            rows = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
            reference = access.create_snapshot(rows)
            rows[0]["value"] = "mutated-after-create"
            payload = access.fetch_snapshot(reference.snapshot_id)
            assert payload.rows[0] == {"id": 1, "value": "alpha"}
            assert payload.content_hash == reference.content_hash


###############################################################################
def test_in_process_snapshot_access_reconstructs_multiple_pages(
    tmp_path: Path,
) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            access = client.app.state.runtime.training_data
            rows = [{"id": index, "value": f"value-{index}"} for index in range(1001)]
            reference = access.create_snapshot(rows)

            payload = access.fetch_snapshot(reference.snapshot_id)

            assert payload.snapshot_id == reference.snapshot_id
            assert payload.content_hash == reference.content_hash
            assert payload.rows == tuple(rows)


###############################################################################
def test_lifespan_stops_active_process_jobs(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        manager = None
        worker = None
        with TestClient(create_app(_temporary_config(directory))) as client:
            manager = client.app.state.core_container.job_manager
            job_id = manager.start_job(
                "shutdown-test",
                _shutdown_process_runner,
                run_mode="process",
            )
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                process_state = manager.processes.get(job_id)
                worker = process_state.process if process_state is not None else None
                if worker is not None and worker.is_alive():
                    break
                status = manager.get_job_status(job_id)
                if status and status["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            assert worker is not None
            assert worker.is_alive()

        assert manager is not None
        if worker is not None:
            worker.join(timeout=3.0)
            assert not worker.is_alive()
        assert manager.processes == {}
        assert manager.threads == {}


# Launcher startup

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPOSITORY_ROOT / "start_on_windows.ps1"


###############################################################################
def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8-sig")


###############################################################################
def start_application_body() -> str:
    text = launcher_text()
    start = text.index("function Start-Application")
    end = text.index("function Install-UpdateDependencies", start)
    return text[start:end]


###############################################################################
def test_launcher_has_no_unconditional_rebuild_or_import_probe() -> None:
    text = launcher_text()

    assert "ALWAYS_REBUILD" not in text
    assert "import server.app, fastapi, uvicorn" not in text
    assert "function Install-FrontendDependencies" in text
    assert "function Build-Frontend" in text
    assert "function Sync-BackendDependencies" in text
    assert "function Sync-FrontendDependencies" not in text


###############################################################################
def test_launcher_loads_endpoints_from_environment_and_database_bootstrap() -> None:
    text = launcher_text()

    assert "python -m server.runtime_config --json" in text
    assert "Environment runtime settings" in text
    assert "$runtime.backend_url" in text
    assert "$runtime.frontend_url" in text
    assert "ADSMOD_DATA_DIR" not in text
    assert "adsmod.json" not in text


###############################################################################
def test_frontend_build_fingerprint_covers_only_declared_inputs() -> None:
    text = launcher_text()
    fingerprint_start = text.index("function Get-FrontendBuildFingerprint")
    fingerprint_end = text.index("function Read-StateFile", fingerprint_start)
    fingerprint = text[fingerprint_start:fingerprint_end]

    for input_name in (
        "package.json",
        "package-lock.json",
        "angular.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "index.html",
        "Join-Path $ClientDir 'src'",
        "Join-Path $ClientDir 'public'",
        "scripts\\verify-angular-migration.mjs",
        "launcher-node-version",
    ):
        assert input_name in fingerprint

    assert "proxy.conf.cjs" not in fingerprint
    assert "playwright.config.ts" not in fingerprint
    assert "eslint.config.js" not in fingerprint


###############################################################################
def test_startup_preflights_all_ports_before_dependency_or_build_work() -> None:
    body = start_application_body()

    assert body.index("Get-PortConflicts") < body.index(
        "Test-PortableRuntimeFilesReady"
    )
    assert body.index("Resolve-PortConflicts") < body.index(
        "Test-BackendDependenciesCurrent"
    )
    assert "@($settings.BackendPort)" in body
    assert "Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60 -Process" in body
    assert "preview" not in body.lower()


###############################################################################
def test_launcher_owns_single_backend_process_in_its_console() -> None:
    text = launcher_text()
    body = start_application_body()

    assert "'Launch' { [void](Start-Application) }" in text
    assert "-NoNewWindow -PassThru" in body
    assert "$settings.BackendUrl" in body
    assert "finally" in body
    assert "Stop-OwnedProcess -Process $script:BackendProcess" in body


###############################################################################
def test_port_conflict_resolution_is_fail_closed_and_deduplicates_pids() -> None:
    text = launcher_text()
    conflict_start = text.index("function Get-PortConflicts")
    conflict_end = text.index("function Get-ApplicationProcessRecords", conflict_start)
    conflict_code = text[conflict_start:conflict_end]

    assert "GetActiveTcpListeners" in text
    assert "function Resolve-PortConflicts" in conflict_code
    assert "HashSet[int]" in conflict_code
    assert "OwnershipResolved" in conflict_code
    assert "non-interactive" in conflict_code
    assert "ownership of port" in conflict_code
    assert "remaining" in conflict_code


###############################################################################
def test_port_conflict_termination_rechecks_process_identity() -> None:
    text = launcher_text()
    conflict_start = text.index("function Get-PortConflicts")
    conflict_end = text.index("function Get-ApplicationProcessRecords", conflict_start)
    conflict_code = text[conflict_start:conflict_end]

    assert "ProcessStartTime" in conflict_code
    assert "$process.Refresh()" in conflict_code
    assert "$process.HasExited" in conflict_code
    assert "$process.ProcessName" in conflict_code
    assert "$process.StartTime.ToUniversalTime().Ticks" in conflict_code
    assert "changed process identity before termination" in conflict_code
