# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""One-time reader for the pre-database.json ADSMOD configuration format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from server.configurations.bootstrap import (
    DOTENV_PATH,
    expand_bootstrap_path,
    persist_database_configuration,
)
from server.configurations.database_config import database_config_path, read_database_config
from server.configurations.settings import (
    DatabaseConfig,
    DatasetConfig,
    FittingConfig,
    JobConfig,
    NISTConfig,
    PublicDataConfig,
    RuntimeConfig,
    TrainingConfig,
    UserSettings,
)
from server.configurations.store import ApplicationSettingsStore
from server.repositories.database.initializer import prepare_database_for_startup
from server.repositories.database.manager import DatabaseManager


def _legacy_settings(path: Path) -> tuple[DatabaseConfig, Path, UserSettings]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Legacy configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Legacy configuration is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Legacy configuration root must be a JSON object.")
    application = payload.get("application")
    if not isinstance(application, dict):
        raise ValueError("Legacy configuration has no application section.")
    database_values = {
        key: value
        for key, value in dict(application.get("database") or {}).items()
        if key in DatabaseConfig.model_fields
    }
    database = DatabaseConfig.model_validate(database_values)
    storage_root_value = str((payload.get("storage") or {}).get("root", "")).strip()
    if not storage_root_value:
        raise ValueError("Legacy configuration has no storage root.")
    storage_root = expand_bootstrap_path(storage_root_value)
    fitting_values = dict(application.get("fitting") or {})
    fitting_values["best_model_metric"] = str(
        fitting_values.get("best_model_metric", "aicc")
    ).strip().casefold()
    settings = UserSettings.model_validate(
        {
            "runtime": RuntimeConfig.model_validate(payload.get("runtime") or {}),
            "datasets": DatasetConfig.model_validate(application.get("datasets") or {}),
            "nist": NISTConfig.model_validate(application.get("nist") or {}),
            "public_data": PublicDataConfig.model_validate(application.get("public_data") or {}),
            "fitting": FittingConfig.model_validate(fitting_values),
            "jobs": JobConfig.model_validate(application.get("jobs") or {}),
            "training": TrainingConfig.model_validate(application.get("training") or {}),
        }
    )
    return database, storage_root, settings


def read_legacy_application_settings(path: Path) -> UserSettings | None:
    if not path.is_file():
        return None
    return _legacy_settings(path)[2]


def import_legacy_configuration(
    legacy_path: Path,
    *,
    storage_root_override: Path | None = None,
    dotenv_path: Path = DOTENV_PATH,
) -> ApplicationSettingsStore:
    legacy_database, legacy_storage_root, settings = _legacy_settings(legacy_path)
    storage_root = (storage_root_override or legacy_storage_root).expanduser().resolve()
    if database_config_path(storage_root).is_file():
        database = read_database_config(storage_root)
    else:
        database = legacy_database
    prepare_database_for_startup(database, storage_root=storage_root)
    manager = DatabaseManager(database, storage_root=storage_root)
    store = ApplicationSettingsStore(manager)
    try:
        if not store.has_record():
            store.initialize(settings)
        if not database_config_path(storage_root).is_file():
            persist_database_configuration(
                database, storage_root, dotenv_path=dotenv_path
            )
        return store
    except Exception:
        manager.dispose()
        raise


__all__ = ["import_legacy_configuration", "read_legacy_application_settings"]
