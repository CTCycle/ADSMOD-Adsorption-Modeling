# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

from server.configurations.legacy_import import import_legacy_configuration
from server.configurations.settings import UserSettings


def test_legacy_import_seeds_sqlite_and_preserves_explicit_storage_root(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    for key in (
        "ADSMOD_STORAGE_ROOT",
        "EMBEDDED_DATABASE",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_SQLITE_PATH",
        "ADSMOD_DATA_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    legacy_root = tmp_path / "legacy-root"
    explicit_root = tmp_path / "custom-existing-root"
    dotenv_path = tmp_path / "settings" / ".env"
    dotenv_path.parent.mkdir(parents=True)
    dotenv_path.write_text(f'ADSMOD_STORAGE_ROOT="{explicit_root}"\n', encoding="utf-8")

    app_settings = UserSettings().model_dump(mode="json")
    app_settings["datasets"]["column_detection_cutoff"] = 0.77
    app_settings["fitting"]["best_model_metric"] = "rmse"
    legacy = {
        "runtime": app_settings.pop("runtime"),
        "storage": {"root": str(legacy_root)},
        "application": {
            "database": {
                "embedded_database": True,
                "engine": "postgres",
                "sqlite_path": "data/legacy.db",
                "password": "bootstrap-secret-marker",
            },
            **app_settings,
        },
    }
    legacy_path = tmp_path / "adsmod.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    store, added_bootstrap_keys = import_legacy_configuration(
        legacy_path,
        dotenv_path=dotenv_path,
    )
    try:
        imported = store.get()
        assert imported.datasets.column_detection_cutoff == 0.77
        assert imported.fitting.best_model_metric == "rmse"
        assert store.database_path == (explicit_root / "data" / "legacy.db").resolve()
        assert added_bootstrap_keys
        env_text = dotenv_path.read_text(encoding="utf-8")
        assert f'ADSMOD_STORAGE_ROOT="{explicit_root}"' in env_text
        assert "DATABASE_SQLITE_PATH=data/legacy.db" in env_text
        assert "bootstrap-secret-marker" in env_text

        store.save(UserSettings())
        store.dispose()
        store, _ = import_legacy_configuration(legacy_path, dotenv_path=dotenv_path)
        assert store.get() == UserSettings()
    finally:
        store.dispose()


def test_legacy_import_expands_windows_storage_root_tokens(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    for key in (
        "ADSMOD_STORAGE_ROOT",
        "EMBEDDED_DATABASE",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_SQLITE_PATH",
        "ADSMOD_DATA_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    legacy = {
        "runtime": UserSettings().model_dump(mode="json").pop("runtime"),
        "storage": {"root": "%LOCALAPPDATA%/ADSMOD"},
        "application": {
            "database": {"embedded_database": True, "sqlite_path": "data/legacy.db"},
            "datasets": UserSettings().datasets.model_dump(mode="json"),
        },
    }
    legacy_path = tmp_path / "adsmod.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    dotenv_path = tmp_path / "settings" / ".env"

    store, _ = import_legacy_configuration(legacy_path, dotenv_path=dotenv_path)
    try:
        assert store.database_path == (local_app_data / "ADSMOD" / "data" / "legacy.db").resolve()
        assert dotenv_path.read_text(encoding="utf-8").splitlines()[0] == (
            f'ADSMOD_STORAGE_ROOT="{(local_app_data / "ADSMOD").resolve()}"'
        )
    finally:
        store.dispose()
