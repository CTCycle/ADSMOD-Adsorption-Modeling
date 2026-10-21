# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
from pathlib import Path

from server.configurations.bootstrap import (
    DatabaseConfigurationLoad,
    RETIRED_DATABASE_KEYS,
    load_database_configuration,
)
from server.configurations.database_config import database_config_path
from server.configurations.runtime import load_runtime_configuration
from server.configurations.settings import DatabaseConfig, RuntimeConfig, UserSettings
from server.runtime_config import build_runtime_record
import server.runtime_config as runtime_config


###############################################################################
def test_runtime_endpoint_record_uses_environment_endpoints_and_database_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    database = DatabaseConfig(
        embedded_database=True,
        sqlite_path=":memory:",
    )
    endpoints = RuntimeConfig(host="::1", backend_port=6046, frontend_port=5174)
    monkeypatch.setattr(runtime_config, "load_runtime_settings", lambda: endpoints)
    monkeypatch.setattr(runtime_config, "load_storage_root", lambda: tmp_path)
    monkeypatch.setattr(
        runtime_config,
        "load_database_configuration",
        lambda **_: DatabaseConfigurationLoad(config=database, needs_persist=False),
    )
    monkeypatch.setattr(
        runtime_config,
        "resolve_sqlite_path",
        lambda *_args, **_kwargs: tmp_path / "database.db",
    )

    record = build_runtime_record()

    assert record["host"] == "::1"
    assert record["backend_port"] == 6046
    assert record["frontend_port"] == 5174
    assert record["backend_url"] == "http://[::1]:6046"
    assert record["frontend_url"] == "http://[::1]:5174"
    assert record["database_backend"] == "sqlite"
    assert record["database_path"] is None


###############################################################################
def test_first_runtime_startup_seeds_defaults_and_persists_current_config(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DATABASE_HOST=retired-host\nDATABASE_PASSWORD=retired-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "server.configurations.runtime.load_database_configuration",
        lambda **kwargs: load_database_configuration(**kwargs, dotenv_path=env_path),
    )
    monkeypatch.setenv("DATABASE_PASSWORD", "process-secret")

    runtime = load_runtime_configuration(storage_root=tmp_path)
    try:
        assert runtime.database_ready is True
        assert runtime.settings_store.get() == UserSettings()
        loaded = load_database_configuration(
            dotenv_path=env_path, storage_root=tmp_path
        )
        assert loaded.needs_persist is False
        assert loaded.config == runtime.config.application.database
        assert "DATABASE_" not in env_path.read_text(encoding="utf-8")
        assert all(key not in os.environ for key in RETIRED_DATABASE_KEYS)
        assert database_config_path(tmp_path).is_file()
    finally:
        runtime.settings_store.database.dispose()


###############################################################################
def test_openapi_runtime_configuration_does_not_create_partial_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "not-created.db"
    runtime = load_runtime_configuration(
        database=DatabaseConfig(embedded_database=True, sqlite_path=str(database_path)),
        storage_root=tmp_path,
        initialize_database=False,
    )
    try:
        assert runtime.database_ready is False
        assert runtime.config.user_settings() == UserSettings()
        assert not database_path.exists()
        assert not database_config_path(tmp_path).exists()
    finally:
        runtime.settings_store.database.dispose()
