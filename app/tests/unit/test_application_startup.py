# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

import time
import server.app as app_module
import server.services.deep_learning as deep_learning_module
from server.services.deep_learning import DeepLearningManager
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
def test_unified_runtime_contracts(tmp_path: Path, wait_for_deep_learning) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            wait_for_deep_learning(client)
            assert client.get("/health/live").json()["service"] == "backend"
            assert client.get("/health/ready").json()["state"] == "ready"
            capabilities = client.get("/api/v1/system/capabilities").json()
            assert capabilities["features"]["datasets"] is True
            assert capabilities["features"]["machine_learning"] is True, (
                client.get("/api/v1/deep-learning/status").json()["reason"]
            )
            assert client.get("/api/v1/system/configuration").status_code == 200
            assert client.get("/api/v1/training/configuration").status_code == 200


###############################################################################
def test_unavailable_database_returns_503_for_database_routes(
    tmp_path: Path, monkeypatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("unsupported schema")

    monkeypatch.setattr(app_module, "prepare_database_for_startup", fail)
    with TestClient(create_app(_temporary_config(str(tmp_path)))) as client:
        assert client.get("/api/v1/datasets/supported-units").status_code == 503
        assert client.get("/api/v1/system/capabilities").status_code == 200
        assert client.get("/api/v1/database/status").status_code != 503


###############################################################################
def test_factory_uses_single_backend_port(tmp_path: Path) -> None:
    application = create_app(_temporary_config(str(tmp_path)))
    assert application.state.config.runtime.backend_port > 0
    assert not hasattr(application.state.config.runtime, "ml_port")


###############################################################################
def test_missing_deep_learning_packages_gate_training_but_keep_core(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(dict))
    with TestClient(create_app(_temporary_config(str(tmp_path)))) as client:
        assert client.get("/api/v1/deep-learning/status").json()["status"] == "not_installed"
        features = client.get("/api/v1/system/capabilities").json()["features"]
        assert features["machine_learning"] is False
        assert features["training"] is False
        gated = client.get("/api/v1/training/configuration")
        assert gated.status_code == 503
        assert "Settings" in gated.json()["detail"]
        assert client.get("/api/v1/datasets").status_code == 200


###############################################################################
def test_broken_deep_learning_install_is_reported_as_failed(
    monkeypatch, tmp_path: Path, wait_for_deep_learning
) -> None:  # type: ignore[no-untyped-def]
    installed = {name: ("1.0", 1) for name in deep_learning_module.DEEP_LEARNING_PACKAGES}
    monkeypatch.setattr(DeepLearningManager, "_read_fingerprint", staticmethod(lambda: installed))
    monkeypatch.setattr(
        deep_learning_module.runner,
        "run_probe",
        lambda *args, **kwargs: {"ok": False, "error": "ImportError: DLL load failed"},
    )
    with TestClient(create_app(_temporary_config(str(tmp_path)))) as client:
        status = wait_for_deep_learning(client)
        assert status["status"] == "failed"
        assert "DLL load failed" in status["reason"]
        assert client.get("/api/v1/system/capabilities").json()["features"]["machine_learning"] is False
        assert client.get("/api/v1/training/status").status_code == 503


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
