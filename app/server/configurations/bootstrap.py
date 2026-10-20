# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from server.configurations.settings import DatabaseConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = REPOSITORY_ROOT / "settings" / ".env"
_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")

BOOTSTRAP_KEYS = (
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
)


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
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def _boolean(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Database bootstrap boolean values must be true or false.")


def expand_bootstrap_path(value: str) -> Path:
    expanded = _WINDOWS_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)), value
    )
    return Path(os.path.expandvars(expanded)).expanduser().resolve()


def load_bootstrap_values(*, dotenv_path: Path = DOTENV_PATH) -> dict[str, str]:
    values = _dotenv_values(dotenv_path)
    for key in BOOTSTRAP_KEYS:
        explicit = os.environ.get(key)
        if explicit is not None:
            values[key] = explicit
    return values


def load_database_config(*, dotenv_path: Path = DOTENV_PATH) -> DatabaseConfig:
    values = load_bootstrap_values(dotenv_path=dotenv_path)
    try:
        config: dict[str, Any] = {
            "embedded_database": _boolean(values.get("EMBEDDED_DATABASE"), default=True),
            "engine": values.get("DATABASE_ENGINE", "postgres").casefold(),
            "host": values.get("DATABASE_HOST"),
            "port": values.get("DATABASE_PORT", "5432"),
            "database_name": values.get("DATABASE_NAME"),
            "username": values.get("DATABASE_USERNAME"),
            "password": values.get("DATABASE_PASSWORD"),
            "ssl": _boolean(values.get("DATABASE_SSL"), default=False),
            "ssl_ca": values.get("DATABASE_SSL_CA"),
            "connect_timeout": values.get("DATABASE_CONNECT_TIMEOUT", "30"),
            "sqlite_path": values.get("DATABASE_SQLITE_PATH", "data/database.db"),
        }
        return DatabaseConfig.model_validate(config)
    except Exception as exc:
        # Pydantic error details can include submitted values, so keep bootstrap
        # diagnostics generic and never disclose a password or connection URL.
        raise ValueError("Database bootstrap settings are invalid; review settings/.env.") from exc


def load_storage_root(*, dotenv_path: Path = DOTENV_PATH) -> Path:
    values = load_bootstrap_values(dotenv_path=dotenv_path)
    configured = values.get("ADSMOD_STORAGE_ROOT")
    if configured:
        return expand_bootstrap_path(configured)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (expand_bootstrap_path(local_app_data) / "ADSMOD").resolve()
    return (Path.home() / "AppData" / "Local" / "ADSMOD").resolve()


def resolve_settings_database_path(
    database: DatabaseConfig, storage_root: Path
) -> Path:
    raw = database.sqlite_path.strip()
    if raw == ":memory:":
        return Path(":memory:")
    expanded = _WINDOWS_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)), raw
    )
    path = Path(os.path.expandvars(expanded)).expanduser()
    if not path.is_absolute():
        path = storage_root / path
    return path.resolve()


__all__ = [
    "BOOTSTRAP_KEYS",
    "DOTENV_PATH",
    "REPOSITORY_ROOT",
    "expand_bootstrap_path",
    "load_bootstrap_values",
    "load_database_config",
    "load_storage_root",
    "resolve_settings_database_path",
]
