from __future__ import annotations

import os

from shared.common.env import load_environment
from shared.common.settings import AppSettings, ServerSettings, get_server_settings


def get_ml_host() -> str:
    load_environment()
    return (os.getenv("ML_SERVICE_HOST") or "127.0.0.1").strip()


def get_ml_port() -> int:
    load_environment()
    raw = (os.getenv("ML_SERVICE_PORT") or "6046").strip()
    return int(raw)


def ml_reload_enabled() -> bool:
    load_environment()
    value = (os.getenv("ML_SERVICE_RELOAD") or "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_app_settings(config_path: str | None = None) -> AppSettings:
    _ = config_path
    return AppSettings.load()


def get_server_settings_runtime(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path)
