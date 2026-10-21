# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Any

from server.configurations.database_config import (
    database_config_path,
    read_database_config,
    write_database_config,
)
from server.configurations.settings import DatabaseConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = REPOSITORY_ROOT / "settings" / ".env"
BOOTSTRAP_KEYS = ("ADSMOD_STORAGE_ROOT",)
LEGACY_DATABASE_KEYS = (
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
)
_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")


###############################################################################
def _dotenv_values(path: Path = DOTENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in BOOTSTRAP_KEYS:
            continue
        values[key] = _parse_env_value(raw_value)
    return values


###############################################################################
def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


###############################################################################
def expand_bootstrap_path(value: str) -> Path:
    expanded = _WINDOWS_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)), value
    )
    return Path(os.path.expandvars(expanded)).expanduser().resolve()


###############################################################################
def load_bootstrap_values(*, dotenv_path: Path = DOTENV_PATH) -> dict[str, str]:
    values = _dotenv_values(dotenv_path)
    for key in BOOTSTRAP_KEYS:
        explicit = os.environ.get(key)
        if explicit is not None:
            values[key] = explicit
    return values


###############################################################################
def load_storage_root(*, dotenv_path: Path = DOTENV_PATH) -> Path:
    configured = load_bootstrap_values(dotenv_path=dotenv_path).get("ADSMOD_STORAGE_ROOT")
    if configured:
        return expand_bootstrap_path(configured)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (expand_bootstrap_path(local_app_data) / "ADSMOD").resolve()
    return (Path.home() / "AppData" / "Local" / "ADSMOD").resolve()


###############################################################################
def _legacy_environment_values(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if dotenv_path.is_file():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() in LEGACY_DATABASE_KEYS:
                values[key.strip()] = _parse_env_value(raw_value)
    for key in LEGACY_DATABASE_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


###############################################################################
def _legacy_json_config() -> dict[str, Any]:
    legacy_file = REPOSITORY_ROOT / "data" / "adsmod.json"
    if not legacy_file.is_file():
        return {}
    try:
        payload = json.loads(legacy_file.read_text(encoding="utf-8"))
        return dict(payload.get("application", {}).get("database", {}))
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise ValueError("Legacy database configuration is invalid; repair data/adsmod.json.") from exc


###############################################################################
def _legacy_database_payload(
    values: dict[str, str], legacy_json: dict[str, Any]
) -> dict[str, Any]:
    return {
        **legacy_json,
        "embedded_database": values.get("EMBEDDED_DATABASE", legacy_json.get("embedded_database", "true")),
        "engine": values.get("DATABASE_ENGINE", legacy_json.get("engine", "postgres")),
        "host": values.get("DATABASE_HOST", legacy_json.get("host")),
        "port": values.get("DATABASE_PORT", legacy_json.get("port", 5432)),
        "database_name": values.get("DATABASE_NAME", legacy_json.get("database_name")),
        "username": values.get("DATABASE_USERNAME", legacy_json.get("username")),
        "password": values.get("DATABASE_PASSWORD", legacy_json.get("password")),
        "ssl": values.get("DATABASE_SSL", legacy_json.get("ssl", "false")),
        "ssl_ca": values.get("DATABASE_SSL_CA", legacy_json.get("ssl_ca")),
        "connect_timeout": values.get("DATABASE_CONNECT_TIMEOUT", legacy_json.get("connect_timeout", 30)),
        "sqlite_path": values.get("DATABASE_SQLITE_PATH", legacy_json.get("sqlite_path", "data/database.db")),
    }


###############################################################################
def _legacy_database_config(dotenv_path: Path) -> DatabaseConfig:
    values = _legacy_environment_values(dotenv_path)
    legacy_json = _legacy_json_config()
    if not values and legacy_json:
        return DatabaseConfig.model_validate(legacy_json)
    if not values:
        return DatabaseConfig()
    try:
        return DatabaseConfig.model_validate(_legacy_database_payload(values, legacy_json))
    except Exception as exc:
        raise ValueError("Legacy database settings are invalid; repair settings/.env before startup.") from exc


###############################################################################
def _remove_legacy_database_values(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    if not any(
        "=" in line and line.split("=", 1)[0].strip() in LEGACY_DATABASE_KEYS
        for line in lines
    ):
        return
    kept = [
        line for line in lines
        if not ("=" in line and line.split("=", 1)[0].strip() in LEGACY_DATABASE_KEYS)
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".env-", suffix=".tmp", dir=dotenv_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(kept).rstrip() + ("\n" if kept else ""))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, dotenv_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


###############################################################################
@dataclass(frozen=True)
class DatabaseConfigurationLoad:
    config: DatabaseConfig
    migrated: bool
    legacy_sqlite_path: str | None = None


###############################################################################
def load_database_configuration(
    *,
    dotenv_path: Path = DOTENV_PATH,
    storage_root: Path | None = None,
    persist_migration: bool = True,
) -> DatabaseConfigurationLoad:
    root = (storage_root or load_storage_root(dotenv_path=dotenv_path)).resolve()
    path = database_config_path(root)
    if path.exists():
        loaded = read_database_config(root)
        _remove_legacy_database_values(dotenv_path)
        for key in LEGACY_DATABASE_KEYS:
            os.environ.pop(key, None)
        return DatabaseConfigurationLoad(loaded, migrated=False)

    legacy = _legacy_database_config(dotenv_path)
    legacy_path = legacy.sqlite_path if not legacy.embedded_database else None
    if persist_migration:
        persist_database_configuration(legacy, root, dotenv_path=dotenv_path)
    return DatabaseConfigurationLoad(legacy, migrated=True, legacy_sqlite_path=legacy_path)


###############################################################################
def persist_database_configuration(
    config: DatabaseConfig,
    storage_root: Path,
    *,
    dotenv_path: Path = DOTENV_PATH,
) -> Path:
    path = write_database_config(config, storage_root)
    _remove_legacy_database_values(dotenv_path)
    for key in LEGACY_DATABASE_KEYS:
        os.environ.pop(key, None)
    return path


###############################################################################
def load_database_config(
    *, dotenv_path: Path = DOTENV_PATH, storage_root: Path | None = None
) -> DatabaseConfig:
    return load_database_configuration(
        dotenv_path=dotenv_path, storage_root=storage_root
    ).config


__all__ = [
    "BOOTSTRAP_KEYS",
    "DatabaseConfigurationLoad",
    "DOTENV_PATH",
    "LEGACY_DATABASE_KEYS",
    "REPOSITORY_ROOT",
    "database_config_path",
    "expand_bootstrap_path",
    "load_bootstrap_values",
    "load_database_config",
    "load_database_configuration",
    "load_storage_root",
    "persist_database_configuration",
    "read_database_config",
    "write_database_config",
]
