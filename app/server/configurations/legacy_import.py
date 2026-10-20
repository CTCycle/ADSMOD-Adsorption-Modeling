# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""One-time importer for pre-SQLite data/adsmod.json installations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from server.configurations.bootstrap import (
    BOOTSTRAP_KEYS,
    DOTENV_PATH,
    expand_bootstrap_path,
    load_database_config,
    load_storage_root,
)
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
from server.configurations.bootstrap import resolve_settings_database_path


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
    metric = str(fitting_values.get("best_model_metric", "aicc")).strip().casefold()
    fitting_values["best_model_metric"] = metric
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


def _quote_env(value: str) -> str:
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"' if any(char.isspace() for char in value) or "#" in value else value


def write_missing_bootstrap_values(
    *,
    database: DatabaseConfig,
    storage_root: Path,
    dotenv_path: Path = DOTENV_PATH,
) -> tuple[str, ...]:
    existing_lines = dotenv_path.read_text(encoding="utf-8").splitlines() if dotenv_path.is_file() else []
    existing_keys: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        existing_keys.add(key)
    env_values = {
        "ADSMOD_STORAGE_ROOT": str(storage_root),
        "EMBEDDED_DATABASE": str(database.embedded_database).lower(),
        "DATABASE_ENGINE": database.engine,
        "DATABASE_HOST": database.host or "",
        "DATABASE_PORT": str(database.port),
        "DATABASE_NAME": database.database_name or "",
        "DATABASE_USERNAME": database.username or "",
        "DATABASE_PASSWORD": database.password or "",
        "DATABASE_SSL": str(database.ssl).lower(),
        "DATABASE_SSL_CA": database.ssl_ca or "",
        "DATABASE_CONNECT_TIMEOUT": str(database.connect_timeout),
        "DATABASE_SQLITE_PATH": database.sqlite_path,
    }
    added: list[str] = []
    for key in BOOTSTRAP_KEYS:
        if key in existing_keys or key in os.environ:
            continue
        value = env_values[key]
        existing_lines.append(f"{key}={_quote_env(value)}")
        added.append(key)
    existing_lines = [
        line
        for line in existing_lines
        if line.strip().split("=", 1)[0].strip() != "ADSMOD_DATA_DIR"
    ]
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    dotenv_path.write_text("\n".join(existing_lines).rstrip() + "\n", encoding="utf-8")
    return tuple(added)


def import_legacy_configuration(
    legacy_path: Path,
    *,
    storage_root_override: Path | None = None,
    dotenv_path: Path = DOTENV_PATH,
) -> tuple[ApplicationSettingsStore, tuple[str, ...]]:
    database, legacy_storage_root, settings = _legacy_settings(legacy_path)
    storage_root = (storage_root_override or legacy_storage_root).expanduser().resolve()
    added_bootstrap_keys = write_missing_bootstrap_values(
        database=database,
        storage_root=storage_root,
        dotenv_path=dotenv_path,
    )
    database = load_database_config(dotenv_path=dotenv_path)
    storage_root = storage_root_override or load_storage_root(dotenv_path=dotenv_path)
    storage_root = storage_root.expanduser().resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    prepare_database_for_startup(database, storage_root=storage_root)
    store = ApplicationSettingsStore(
        resolve_settings_database_path(database, storage_root)
    )
    store.initialize(settings)
    return store, added_bootstrap_keys


__all__ = ["import_legacy_configuration", "write_missing_bootstrap_values"]
