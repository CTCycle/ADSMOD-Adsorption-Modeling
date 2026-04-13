from __future__ import annotations

import os
from typing import Any

from ADSMOD.server.configurations.environment import load_environment
from ADSMOD.server.configurations.management import ConfigurationManager
from ADSMOD.server.domain.settings import AppSettings, ServerSettings

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
    load_environment()
    host = os.getenv("FASTAPI_HOST", "").strip().lower()
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
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_path, "client", "dist")


# -----------------------------------------------------------------------------
def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())


# -----------------------------------------------------------------------------
def resolve_spa_file_path(client_dist_path: str, requested_path: str) -> str | None:
    normalized_path = str(requested_path or "").lstrip("/\\")
    absolute_root = os.path.abspath(client_dist_path)
    candidate = os.path.abspath(os.path.join(absolute_root, normalized_path))
    if os.path.commonpath([absolute_root, candidate]) != absolute_root:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


