# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from server.configurations.runtime import RuntimeConfiguration
from server.configurations.settings import AdsmodConfig, DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.manager import DatabaseManager
from server.runtime_config import build_runtime_record
import server.runtime_config as runtime_config


def test_runtime_endpoint_record_uses_settings_loaded_from_the_database_config(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    database = DatabaseConfig(
        embedded_database=False,
        engine="postgres",
        host="127.0.0.1",
        port=1,
        database_name="unavailable",
        username="unavailable",
        password="must-not-be-returned",
        sqlite_path="settings/settings.db",
    )
    manager = DatabaseManager(database, storage_root=tmp_path)
    store = ApplicationSettingsStore(manager)
    config = AdsmodConfig.assemble(
        database=database, storage_root=tmp_path, settings=UserSettings()
    )
    runtime = RuntimeConfiguration(config=config, settings_store=store, database_ready=False)
    monkeypatch.setattr(
        runtime_config, "load_runtime_configuration", lambda *, initialize_database=False: runtime
    )

    record = build_runtime_record()

    assert record["database_backend"] == "postgres"
    assert record["backend_port"] == UserSettings().runtime.backend_port
    assert record["frontend_port"] == UserSettings().runtime.frontend_port
    assert record["database_path"] is None
    assert "settings_database_path" not in record
    assert "must-not-be-returned" not in str(record)
