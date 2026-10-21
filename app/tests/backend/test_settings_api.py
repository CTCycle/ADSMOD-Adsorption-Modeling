# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy import delete

from server.app import create_app
from server.configurations.database_config import read_database_config, write_database_config
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsRecord
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.initializer import prepare_database_for_startup
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import Dataset


def _config(root: Path) -> AdsmodConfig:
    return AdsmodConfig.assemble(
        database=DatabaseConfig(embedded_database=True, sqlite_path="data/settings.db"),
        storage_root=root,
        settings=UserSettings(),
    )


def test_settings_save_applies_to_new_requests_and_requires_restarts_when_needed(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        payload = client.get("/api/v1/settings").json()["current"]
        payload["runtime"]["backend_port"] = 6501
        payload["datasets"]["allowed_extensions"] = [".csv"]
        payload["datasets"]["column_detection_cutoff"] = 0.99
        payload["nist"]["parallel_tasks"] = 6
        payload["public_data"]["request_timeout_seconds"] = 11
        payload["public_data"]["retry_attempts"] = 4
        payload["public_data"]["cod_max_interactive_results"] = 321
        payload["public_data"]["sync_batch_size"] = 25
        payload["public_data"]["sync_max_records_per_run"] = 333
        payload["public_data"]["auto_sync_enabled"] = True
        payload["public_data"]["auto_sync_categories"] = ["guest"]
        payload["public_data"]["auto_sync_interval_hours"] = 12
        payload["public_data"]["auto_sync_fraction"] = 0.5
        payload["public_data"]["sync_resume_on_startup"] = True
        payload["fitting"]["default_max_iterations"] = 2500
        payload["fitting"]["max_iterations_upper_bound"] = 5000
        payload["fitting"]["default_parameter_initial"] = 2
        payload["fitting"]["default_parameter_max"] = 9
        payload["fitting"]["preview_row_limit"] = 7
        payload["fitting"]["best_model_metric"] = "rmse"
        payload["jobs"]["polling_interval"] = 2.5
        payload["training"]["dataloader_workers"] = 2
        payload["training"]["persistent_workers"] = True

        response = client.put("/api/v1/settings", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["current"]["fitting"]["best_model_metric"] == "rmse"
        assert body["restart_required"] is True
        assert body["pending_restart_fields"] == [
            "runtime.backend_port",
            "public_data.sync_resume_on_startup",
        ]
        runtime = client.app.state.runtime
        assert runtime.config.runtime.backend_port == 6501
        assert runtime.container.dataset_service.allowed_extensions == [".csv"]
        assert runtime.container.dataset_service.importer.column_detection_cutoff == 0.99
        assert runtime.container.dataset_service.importer.preview_row_limit == 7
        assert runtime.container.public_data_sync.config.request_timeout_seconds == 11
        assert runtime.container.nist_service._api_client().parallel_tasks == 6
        assert runtime.container.public_data_service.providers["cod"].max_interactive_results == 321
        assert runtime.container.public_data_sync.config.sync_batch_size == 25
        assert runtime.container.public_data_sync.config.sync_max_records_per_run == 333
        assert runtime.container.public_data_sync.config.auto_sync_enabled is True
        assert runtime.container.public_data_sync.config.auto_sync_categories == ("guest",)
        assert runtime.container.public_data_sync.config.auto_sync_interval_hours == 12
        assert runtime.container.public_data_sync.config.auto_sync_fraction == 0.5
        assert runtime.container.fitting_service.config.application.fitting.best_model_metric == "rmse"
        assert runtime.container.fitting_service.config.application.jobs.polling_interval == 2.5
        assert runtime.container.config.application.training.persistent_workers is True
        assert client.get("/api/v1/system/configuration").json()["parameter_defaults"] == {
            "lower": 0.0,
            "upper": 9.0,
            "initial": 2.0,
        }
        assert client.get("/api/v1/system/configuration").json()["default_max_evaluations"] == 2500
        assert client.get("/api/v1/system/configuration").json()["max_evaluations_bounds"]["maximum"] == 5000
        assert client.get("/api/v1/datasets/supported-units").json()["allowed_extensions"] == [".csv"]
        training_configuration = client.get("/api/v1/training/configuration")
        if training_configuration.status_code == 200:
            assert training_configuration.json()["defaults"]["dataloader_workers"] == 2
            assert training_configuration.json()["defaults"]["persistent_workers"] is True
        preview = client.post(
            "/api/v1/datasets/import/preview",
            files={
                "file": (
                    "sample_adsorption.csv",
                    (Path(__file__).resolve().parents[1] / "fixtures" / "sample_adsorption.csv").read_bytes(),
                    "text/csv",
                )
            },
        ).json()
        assert len(preview["preview_rows"]) == 7
        assert {issue["code"] for issue in preview["issues"]} >= {
            "unmapped_pressure",
            "unmapped_uptake",
        }
        rejected_fit = client.post(
            "/api/v1/fitting/run",
            json={"dataset_id": 1, "isotherm_id": 1, "models": ["langmuir"], "max_evaluations": 5001},
        )
        assert rejected_fit.status_code == 400
        assert client.get("/api/v1/settings").json()["current"] == payload


def test_settings_validation_is_atomic_and_reset_returns_defaults(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        original = client.get("/api/v1/settings").json()["current"]
        invalid = {**original, "runtime": {**original["runtime"], "frontend_port": original["runtime"]["backend_port"]}}

        assert client.put("/api/v1/settings", json=invalid).status_code == 422
        assert client.get("/api/v1/settings").json()["current"] == original

        invalid_workers = {**original, "training": {**original["training"], "persistent_workers": True}}
        assert client.put("/api/v1/settings", json=invalid_workers).status_code == 422

        saved = {**original, "fitting": {**original["fitting"], "preview_row_limit": 31}}
        assert client.put("/api/v1/settings", json=saved).status_code == 200
        reset = client.post("/api/v1/settings/reset")
        assert reset.status_code == 200
        assert reset.json()["current"] == reset.json()["defaults"]
        assert reset.json()["pending_restart_fields"] == []


def test_settings_apply_failure_restores_database_and_runtime_state(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        original = client.get("/api/v1/settings").json()["current"]
        runtime = client.app.state.runtime
        apply_settings = runtime.container.apply_settings

        def fail_after_applying_candidate(config: AdsmodConfig) -> None:
            apply_settings(config)
            if config.application.datasets.column_detection_cutoff == 0.91:
                raise RuntimeError("injected settings apply failure")

        runtime.container.apply_settings = fail_after_applying_candidate
        candidate = {
            **original,
            "datasets": {**original["datasets"], "column_detection_cutoff": 0.91},
        }

        response = client.put("/api/v1/settings", json=candidate)

        assert response.status_code == 503
        assert "previous settings were restored" in response.json()["detail"]
        assert client.get("/api/v1/settings").json()["current"] == original
        assert runtime.config.user_settings().model_dump(mode="json") == original
        assert runtime.container.dataset_service.importer.column_detection_cutoff == original["datasets"]["column_detection_cutoff"]


def test_settings_persist_across_app_restart_and_missing_row_recovers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        payload = client.get("/api/v1/settings").json()["current"]
        payload["public_data"]["cod_max_interactive_results"] = 321
        assert client.put("/api/v1/settings", json=payload).status_code == 200

    with TestClient(create_app(config)) as restarted:
        assert restarted.get("/api/v1/settings").json()["current"]["public_data"]["cod_max_interactive_results"] == 321
        store = restarted.app.state.settings_store
        with store.engine.begin() as connection:
            connection.execute(delete(ApplicationSettingsRecord))
        recovered = restarted.get("/api/v1/settings")
        assert recovered.status_code == 200
        assert recovered.json()["current"]["public_data"]["cod_max_interactive_results"] == 321


def _database_payload(path: str) -> dict[str, object]:
    return {
        "backend": "sqlite",
        "sqlite_path": path,
        "engine": "postgresql+psycopg",
        "host": "",
        "port": 5432,
        "database_name": "",
        "username": "",
        "password": "",
        "clear_password": False,
        "ssl": False,
        "ssl_ca": "",
        "connect_timeout": 30,
    }


def test_runtime_database_switch_reloads_settings_and_keeps_stores_independent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    target_database = DatabaseConfig(
        embedded_database=True, sqlite_path="data/second.db"
    )
    prepare_database_for_startup(target_database, storage_root=tmp_path)
    target_manager = DatabaseManager(target_database, storage_root=tmp_path)
    target_store = ApplicationSettingsStore(target_manager)
    defaults = UserSettings()
    target_settings = UserSettings.model_validate(
        {
            **defaults.model_dump(),
            "runtime": {
                **defaults.runtime.model_dump(),
                "frontend_port": 6200,
            },
            "public_data": {
                **defaults.public_data.model_dump(),
                "cod_max_interactive_results": 111,
            },
        }
    )
    target_store.initialize(target_settings)
    target_manager.dispose()

    with TestClient(create_app(config)) as client:
        source_settings = client.get("/api/v1/settings").json()["current"]
        source_settings["public_data"]["cod_max_interactive_results"] = 321
        assert client.put("/api/v1/settings", json=source_settings).status_code == 200
        with client.app.state.core_container.database.transaction() as session:
            session.add(Dataset(name="Source only", source="uploaded"))

        candidate = _database_payload("data/second.db")
        tested = client.post("/api/v1/database/test", json=candidate)
        assert tested.status_code == 200
        assert tested.json()["success"] is True

        switched = client.post("/api/v1/database/switch", json=candidate)
        assert switched.status_code == 200
        assert switched.json()["backend"] == "sqlite"
        assert switched.json()["sqlite_path"] == "data/second.db"
        assert switched.json()["connected"] is True
        settings_after_switch = client.get("/api/v1/settings").json()
        assert settings_after_switch["current"]["public_data"]["cod_max_interactive_results"] == 111
        assert settings_after_switch["current"]["runtime"]["frontend_port"] == 6200
        assert settings_after_switch["pending_restart_fields"] == ["runtime.frontend_port"]
        assert client.get("/api/v1/datasets").json()["datasets"] == []

        back = client.post(
            "/api/v1/database/switch", json=_database_payload("data/settings.db")
        )
        assert back.status_code == 200
        settings_after_switchback = client.get("/api/v1/settings").json()
        assert settings_after_switchback["current"]["public_data"]["cod_max_interactive_results"] == 321
        assert settings_after_switchback["pending_restart_fields"] == []
        assert len(client.get("/api/v1/datasets").json()["datasets"]) == 1


def test_failed_runtime_database_switch_rolls_back_and_active_work_blocks_switch(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        runtime = client.app.state.runtime
        original_database = runtime.config.application.database
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("file", encoding="utf-8")
        failed = client.post(
            "/api/v1/database/switch",
            json=_database_payload(str(blocker / "target.db")),
        )
        assert failed.status_code == 503
        assert runtime.config.application.database == original_database
        assert client.get("/api/v1/database").json()["sqlite_path"] == "data/settings.db"

        runtime.container.job_manager.is_job_running = lambda job_type=None: True
        blocked = client.post(
            "/api/v1/database/switch", json=_database_payload("data/blocked.db")
        )
        assert blocked.status_code == 409
        assert runtime.config.application.database == original_database

        runtime.container.job_manager.is_job_running = lambda job_type=None: False
        runtime.container.public_data_sync.pause_for_database_switch = AsyncMock(
            return_value=False
        )
        blocked_sync = client.post(
            "/api/v1/database/switch", json=_database_payload("data/sync-blocked.db")
        )
        assert blocked_sync.status_code == 409
        assert "sync runs" in blocked_sync.json()["detail"]
        assert runtime.config.application.database == original_database


def test_activation_failure_restores_runtime_services_and_database_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original_database = config.application.database
    write_database_config(original_database, tmp_path)

    with TestClient(create_app(config)) as client:
        container = client.app.state.core_container
        apply_settings = container.apply_settings

        def fail_after_candidate_is_applied(next_config: AdsmodConfig) -> None:
            apply_settings(next_config)
            if next_config.application.database != original_database:
                raise RuntimeError("injected activation failure")

        container.apply_settings = fail_after_candidate_is_applied
        failed = client.post(
            "/api/v1/database/switch",
            json=_database_payload("data/activation-failure.db"),
        )

        assert failed.status_code == 503
        assert client.app.state.runtime.config.application.database == original_database
        assert container.config.application.database == original_database
        assert container.database.database == original_database
        assert client.get("/api/v1/database").json()["sqlite_path"] == "data/settings.db"
        assert read_database_config(tmp_path) == original_database
