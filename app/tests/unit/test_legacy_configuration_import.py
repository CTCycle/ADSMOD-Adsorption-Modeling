# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

from server.configurations.database_config import database_config_path
from server.configurations.legacy_import import import_legacy_configuration
from server.configurations.settings import UserSettings


def _legacy_file(path: Path, storage_root: str) -> None:
    settings = UserSettings().model_dump(mode="json")
    settings["datasets"]["column_detection_cutoff"] = 0.77
    settings["fitting"]["best_model_metric"] = "rmse"
    runtime = settings.pop("runtime")
    path.write_text(
        json.dumps(
            {
                "runtime": runtime,
                "storage": {"root": storage_root},
                "application": {
                    "database": {
                        "embedded_database": True,
                        "sqlite_path": "data/legacy.db",
                        "password": "legacy-secret-marker",
                    },
                    **settings,
                },
            }
        ),
        encoding="utf-8",
    )


def test_legacy_import_moves_settings_to_active_database_and_removes_env_database_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "application-data"
    dotenv_path = tmp_path / "settings" / ".env"
    dotenv_path.parent.mkdir(parents=True)
    dotenv_path.write_text(
        "ADSMOD_STORAGE_ROOT=keep-this\nDATABASE_PASSWORD=old-secret\nDATABASE_SQLITE_PATH=old.db\n",
        encoding="utf-8",
    )
    legacy_path = tmp_path / "adsmod.json"
    _legacy_file(legacy_path, str(root))

    store = import_legacy_configuration(
        legacy_path, storage_root_override=root, dotenv_path=dotenv_path
    )
    try:
        imported = store.get()
        assert imported.datasets.column_detection_cutoff == 0.77
        assert imported.fitting.best_model_metric == "rmse"
        config_text = database_config_path(root).read_text(encoding="utf-8")
        assert "legacy-secret-marker" not in config_text
        assert "old-secret" not in config_text
        env_text = dotenv_path.read_text(encoding="utf-8")
        assert "ADSMOD_STORAGE_ROOT=keep-this" in env_text
        assert "DATABASE_" not in env_text
    finally:
        store.database.dispose()

    store = import_legacy_configuration(
        legacy_path, storage_root_override=root, dotenv_path=dotenv_path
    )
    try:
        assert store.get() == imported
    finally:
        store.database.dispose()


def test_legacy_import_expands_windows_storage_root_tokens(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    legacy_path = tmp_path / "adsmod.json"
    _legacy_file(legacy_path, "%LOCALAPPDATA%/ADSMOD")
    dotenv_path = tmp_path / ".env"

    store = import_legacy_configuration(legacy_path, dotenv_path=dotenv_path)
    try:
        assert database_config_path(local_app_data / "ADSMOD").is_file()
        assert store.get().datasets.column_detection_cutoff == 0.77
    finally:
        store.database.dispose()
