from __future__ import annotations

import os
from pathlib import Path


ROOT_PATH = Path(__file__).resolve().parents[5]
APP_PATH = ROOT_PATH / "app"
SERVER_PATH = APP_PATH / "server"
SETTING_DIR = ROOT_PATH / "settings"
DEFAULT_RESOURCES_DIR = APP_PATH / "resources"

###############################################################################
def resolve_resources_dir(configured_path: str | Path | None = None) -> Path:
    raw_path = (
        os.getenv("ADSMOD_RESOURCES_DIR")
        if configured_path is None
        else str(configured_path)
    )
    normalized_path = str(raw_path or "").strip().strip('"').strip("'")
    if not normalized_path:
        return DEFAULT_RESOURCES_DIR.resolve()

    expanded_path = Path(os.path.expandvars(normalized_path)).expanduser()
    if not expanded_path.is_absolute():
        expanded_path = ROOT_PATH / expanded_path
    return expanded_path.resolve()


RESOURCES_DIR = resolve_resources_dir()
CANONICAL_CONFIGURATION_FILE = RESOURCES_DIR / "adsmod.json"
LOGS_DIR = RESOURCES_DIR / "logs"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
CHECKPOINTS_DIR = RESOURCES_DIR / "checkpoints"
ENV_FILE = SETTING_DIR / ".env"
CLIENT_DIST_DIR = APP_PATH / "client" / "dist"
CLIENT_INDEX_FILE = CLIENT_DIST_DIR / "index.html"
CLIENT_ASSETS_DIR = CLIENT_DIST_DIR / "assets"


__all__ = [
    "APP_PATH",
    "CHECKPOINTS_DIR",
    "CLIENT_ASSETS_DIR",
    "CLIENT_DIST_DIR",
    "CLIENT_INDEX_FILE",
    "CANONICAL_CONFIGURATION_FILE",
    "DEFAULT_RESOURCES_DIR",
    "ENV_FILE",
    "LOGS_DIR",
    "RESOURCES_DIR",
    "ROOT_PATH",
    "SERVER_PATH",
    "SETTING_DIR",
    "TEMPLATES_DIR",
    "resolve_resources_dir",
]
