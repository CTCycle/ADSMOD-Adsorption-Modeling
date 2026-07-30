from __future__ import annotations

from shared.common.paths import CANONICAL_CONFIGURATION_FILE
from shared.common.settings import AppSettings, ServerSettings, get_runtime_config, get_server_settings

###############################################################################
def get_ml_host() -> str:
    return str(get_runtime_config().get("host", "127.0.0.1"))

###############################################################################
def get_ml_port() -> int:
    return int(get_runtime_config()["ml_port"])

###############################################################################
def get_app_settings(config_path: str | None = None) -> AppSettings:
    return AppSettings.load(config_path or CANONICAL_CONFIGURATION_FILE)

###############################################################################
def get_server_settings_runtime(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path or CANONICAL_CONFIGURATION_FILE)
