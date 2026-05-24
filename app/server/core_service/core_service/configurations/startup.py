from __future__ import annotations

import os

from shared.common.env import load_environment
from shared.common.settings import AppSettings, ServerSettings, get_server_settings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def get_core_host() -> str:
    load_environment()
    return (os.getenv("CORE_SERVICE_HOST") or os.getenv("FASTAPI_HOST") or "127.0.0.1").strip()


def get_core_port() -> int:
    load_environment()
    raw = (os.getenv("CORE_SERVICE_PORT") or os.getenv("FASTAPI_PORT") or "6045").strip()
    return int(raw)


def core_reload_enabled() -> bool:
    load_environment()
    value = (os.getenv("CORE_SERVICE_RELOAD") or os.getenv("RELOAD") or "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_app_settings(config_path: str | None = None) -> AppSettings:
    _ = config_path
    return AppSettings.load()


def get_server_settings_runtime(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path)


def public_host_mode_enabled() -> bool:
    host = get_core_host().strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)


def resolve_spa_file_path(client_dist_path: str, requested_path: str) -> str | None:
    normalized_path = str(requested_path or "").lstrip("/\\")
    absolute_root = os.path.abspath(client_dist_path)
    candidate = os.path.abspath(os.path.join(absolute_root, normalized_path))
    if os.path.commonpath([absolute_root, candidate]) != absolute_root:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate

def direct_api_enabled() -> bool:
    return not public_host_mode_enabled()


def tauri_mode_enabled() -> bool:
    load_environment()
    value = os.getenv("ADSMOD_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    return os.path.join(project_path, "app", "client", "dist")


def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())
