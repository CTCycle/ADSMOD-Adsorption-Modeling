from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Any

from app.server.configurations.environment import load_environment
from app.server.configurations.management import ConfigurationManager
from app.server.domain.settings import AppSettings, ServerSettings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


###############################################################################
def build_configuration_runtime(config_path: str | None = None) -> ConfigurationManager:
    load_environment()
    manager = ConfigurationManager(config_path=config_path)
    manager.load()
    return manager


# -----------------------------------------------------------------------------
def get_app_settings(config_path: str | None = None) -> AppSettings:
    manager = build_configuration_runtime(config_path=config_path)
    if manager.settings is None:
        raise RuntimeError("Application settings are not available.")
    return manager.settings


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    return build_configuration_runtime(config_path=config_path).to_server_settings()


# -----------------------------------------------------------------------------
def load_configuration_data(path: str | None = None) -> dict[str, Any]:
    return ConfigurationManager.load_configuration_data(path)


###############################################################################
def public_host_mode_enabled() -> bool:
    host = os.getenv("FASTAPI_HOST")
    if host is None:
        load_environment()
        host = os.getenv("FASTAPI_HOST", "")
    host = host.strip().lower()
    if not host:
        return False
    return host not in LOOPBACK_HOSTS


# -----------------------------------------------------------------------------
def direct_api_enabled() -> bool:
    return not public_host_mode_enabled()


# -----------------------------------------------------------------------------
def tauri_mode_enabled() -> bool:
    load_environment()
    value = os.getenv("ADSMOD_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


# -----------------------------------------------------------------------------
def get_client_dist_path() -> str:
    project_path = Path(__file__).resolve().parents[2]
    return str(project_path / "client" / "dist")


# -----------------------------------------------------------------------------
def packaged_client_available() -> bool:
    return tauri_mode_enabled() and Path(get_client_dist_path()).is_dir()


# -----------------------------------------------------------------------------
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


