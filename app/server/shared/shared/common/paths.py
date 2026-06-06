from __future__ import annotations

from pathlib import Path


ROOT_PATH = Path(__file__).resolve().parents[5]
APP_PATH = ROOT_PATH / "app"
SERVER_PATH = APP_PATH / "server"
SETTING_DIR = ROOT_PATH / "settings"
RESOURCES_DIR = APP_PATH / "resources"
LOGS_DIR = RESOURCES_DIR / "logs"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
CHECKPOINTS_DIR = RESOURCES_DIR / "checkpoints"
ENV_FILE = SETTING_DIR / ".env"
CORE_CONFIGURATION_FILE = SETTING_DIR / "core_service.json"
ML_CONFIGURATION_FILE = SETTING_DIR / "ml_service.json"
CLIENT_DIST_DIR = APP_PATH / "client" / "dist"
CLIENT_INDEX_FILE = CLIENT_DIST_DIR / "index.html"
CLIENT_ASSETS_DIR = CLIENT_DIST_DIR / "assets"


__all__ = [
    "APP_PATH",
    "CHECKPOINTS_DIR",
    "CLIENT_ASSETS_DIR",
    "CLIENT_DIST_DIR",
    "CLIENT_INDEX_FILE",
    "CORE_CONFIGURATION_FILE",
    "ENV_FILE",
    "LOGS_DIR",
    "ML_CONFIGURATION_FILE",
    "RESOURCES_DIR",
    "ROOT_PATH",
    "SERVER_PATH",
    "SETTING_DIR",
    "TEMPLATES_DIR",
]
