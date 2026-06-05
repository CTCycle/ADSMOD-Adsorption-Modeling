from __future__ import annotations

import os
from os import PathLike
from pathlib import Path

from core_service.common.constants import CONFIGURATION_FILE
from shared.common.env import load_environment
from shared.common.settings import AppSettings, ServerSettings, get_server_settings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def get_env_value(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    load_environment()
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def get_core_host() -> str:
    return get_env_value("CORE_SERVICE_HOST", "FASTAPI_HOST", default="127.0.0.1")


def get_core_port() -> int:
    raw = get_env_value("CORE_SERVICE_PORT", "FASTAPI_PORT", default="6045")
    return int(raw)


def core_reload_enabled() -> bool:
    value = get_env_value("CORE_SERVICE_RELOAD", "RELOAD", default="true").lower()
    return value in {"1", "true", "yes", "on"}


def get_app_settings(config_path: str | None = None) -> AppSettings:
    return AppSettings.load(config_path or CONFIGURATION_FILE)


def get_server_settings_runtime(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path or CONFIGURATION_FILE)


def public_host_mode_enabled() -> bool:
    host = get_core_host().strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)


def resolve_spa_file_path(
    client_dist_path: str | PathLike[str], requested_path: str | PathLike[str]
) -> str | None:
    normalized_path = str(requested_path or "").replace("\\", "/").lstrip("/")
    absolute_root = Path(client_dist_path).resolve()
    candidate = (absolute_root / normalized_path).resolve()
    try:
        candidate.relative_to(absolute_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return str(candidate)

def direct_api_enabled() -> bool:
    return not public_host_mode_enabled()


def tauri_mode_enabled() -> bool:
    value = get_env_value("ADSMOD_TAURI_MODE", default="false").lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = Path(__file__).resolve().parents[5]
    return str(project_path / "app" / "client" / "dist")


def packaged_client_available() -> bool:
    return tauri_mode_enabled() and Path(get_client_dist_path()).is_dir()
