# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass

from server.configurations.database_config import (
    database_config_path,
    read_database_config,
    write_database_config,
)
from server.configurations.settings import DatabaseConfig, RuntimeConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = REPOSITORY_ROOT / "settings" / ".env"
BOOTSTRAP_KEYS = ("ADSMOD_STORAGE_ROOT",)
RUNTIME_KEYS = (
    "ADSMOD_HOST",
    "ADSMOD_BACKEND_PORT",
    "ADSMOD_FRONTEND_PORT",
    "ADSMOD_CORS_ORIGINS",
    "ADSMOD_CORS_ALLOW_CREDENTIALS",
    "ADSMOD_CORS_ALLOW_METHODS",
    "ADSMOD_CORS_ALLOW_HEADERS",
)
RETIRED_DATABASE_KEYS = (
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
def _dotenv_values(
    path: Path = DOTENV_PATH, *, keys: tuple[str, ...] = BOOTSTRAP_KEYS
) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in keys:
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
    values = _dotenv_values(dotenv_path, keys=BOOTSTRAP_KEYS)
    for key in BOOTSTRAP_KEYS:
        explicit = os.environ.get(key)
        if explicit is not None:
            values[key] = explicit
    return values


###############################################################################
def load_storage_root(*, dotenv_path: Path = DOTENV_PATH) -> Path:
    configured = load_bootstrap_values(dotenv_path=dotenv_path).get(
        "ADSMOD_STORAGE_ROOT"
    )
    if configured:
        return expand_bootstrap_path(configured)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (expand_bootstrap_path(local_app_data) / "ADSMOD").resolve()
    return (Path.home() / "AppData" / "Local" / "ADSMOD").resolve()


###############################################################################
def load_runtime_settings(*, dotenv_path: Path = DOTENV_PATH) -> RuntimeConfig:
    """Load process runtime endpoints and CORS policy from .env or the environment."""

    values = _dotenv_values(dotenv_path, keys=RUNTIME_KEYS)
    for key in RUNTIME_KEYS:
        explicit = os.environ.get(key)
        if explicit is not None:
            values[key] = explicit

    supplied = {
        model_key: values[environment_key]
        for environment_key, model_key in (
            ("ADSMOD_HOST", "host"),
            ("ADSMOD_BACKEND_PORT", "backend_port"),
            ("ADSMOD_FRONTEND_PORT", "frontend_port"),
        )
        if environment_key in values
    }
    base = RuntimeConfig.model_validate(supplied)

    if "ADSMOD_CORS_ORIGINS" in values:
        origins: str | tuple[str, ...] = values["ADSMOD_CORS_ORIGINS"]
    else:
        hosts = dict.fromkeys((base.host, "127.0.0.1", "localhost"))
        origins = tuple(
            f"http://{'[' + host + ']' if ':' in host else host}:{base.frontend_port}"
            for host in hosts
        )

    return RuntimeConfig.model_validate(
        {
            **base.model_dump(),
            "cors_origins": origins,
            "cors_allow_credentials": values.get("ADSMOD_CORS_ALLOW_CREDENTIALS", True),
            "cors_allow_methods": values.get("ADSMOD_CORS_ALLOW_METHODS", "*"),
            "cors_allow_headers": values.get("ADSMOD_CORS_ALLOW_HEADERS", "*"),
        }
    )


###############################################################################
def _remove_retired_database_values(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    if not any(
        "=" in line and line.split("=", 1)[0].strip() in RETIRED_DATABASE_KEYS
        for line in lines
    ):
        return
    kept = [
        line
        for line in lines
        if not ("=" in line and line.split("=", 1)[0].strip() in RETIRED_DATABASE_KEYS)
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
    needs_persist: bool


###############################################################################
def load_database_configuration(
    *,
    dotenv_path: Path = DOTENV_PATH,
    storage_root: Path | None = None,
) -> DatabaseConfigurationLoad:
    root = (storage_root or load_storage_root(dotenv_path=dotenv_path)).resolve()
    path = database_config_path(root)
    if path.exists():
        loaded = read_database_config(root)
        _remove_retired_database_values(dotenv_path)
        for key in RETIRED_DATABASE_KEYS:
            os.environ.pop(key, None)
        return DatabaseConfigurationLoad(loaded, needs_persist=False)

    _remove_retired_database_values(dotenv_path)
    for key in RETIRED_DATABASE_KEYS:
        os.environ.pop(key, None)
    return DatabaseConfigurationLoad(DatabaseConfig(), needs_persist=True)


###############################################################################
def persist_database_configuration(
    config: DatabaseConfig,
    storage_root: Path,
    *,
    dotenv_path: Path = DOTENV_PATH,
) -> Path:
    path = write_database_config(config, storage_root)
    _remove_retired_database_values(dotenv_path)
    for key in RETIRED_DATABASE_KEYS:
        os.environ.pop(key, None)
    return path


__all__ = [
    "BOOTSTRAP_KEYS",
    "DatabaseConfigurationLoad",
    "DOTENV_PATH",
    "RETIRED_DATABASE_KEYS",
    "RUNTIME_KEYS",
    "REPOSITORY_ROOT",
    "database_config_path",
    "expand_bootstrap_path",
    "load_bootstrap_values",
    "load_database_configuration",
    "load_runtime_settings",
    "load_storage_root",
    "persist_database_configuration",
    "read_database_config",
    "write_database_config",
]
