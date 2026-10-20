# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

from server.configurations.settings import DatabaseConfig, UserSettings
from server.configurations.store import ApplicationSettingsStore
from server.runtime_config import build_runtime_record
import server.runtime_config as runtime_config


def test_runtime_endpoint_reader_uses_local_settings_without_connecting_to_postgres(
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
    monkeypatch.setattr(runtime_config, "load_database_config", lambda: database)
    monkeypatch.setattr(runtime_config, "load_storage_root", lambda: tmp_path)

    record = build_runtime_record()

    assert record["embedded_database"] is False
    assert record["backend_port"] == UserSettings().runtime.backend_port
    assert record["frontend_port"] == UserSettings().runtime.frontend_port
    assert record["settings_database_path"] == str(
        (tmp_path / "settings" / "settings.db").resolve()
    )
    assert "must-not-be-returned" not in str(record)
    store = ApplicationSettingsStore(Path(str(record["settings_database_path"])))
    try:
        assert store.get() == UserSettings()
    finally:
        store.dispose()
