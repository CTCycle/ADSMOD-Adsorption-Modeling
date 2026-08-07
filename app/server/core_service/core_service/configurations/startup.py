from __future__ import annotations

from os import PathLike
from pathlib import Path

from shared.common.paths import CANONICAL_CONFIGURATION_FILE
from shared.common.settings import AppSettings, ServerSettings, get_runtime_config, get_server_settings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

###############################################################################
def get_core_host() -> str:
    return str(get_runtime_config().get("host", "127.0.0.1"))

###############################################################################
def get_core_port() -> int:
    return int(get_runtime_config()["core_port"])

###############################################################################
def get_app_settings(config_path: str | None = None) -> AppSettings:
    return AppSettings.load(config_path or CANONICAL_CONFIGURATION_FILE)

###############################################################################
def get_server_settings_runtime(config_path: str | None = None) -> ServerSettings:
    return get_server_settings(config_path or CANONICAL_CONFIGURATION_FILE)

###############################################################################
def public_host_mode_enabled() -> bool:
    host = get_core_host().strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)

###############################################################################
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

###############################################################################
def direct_api_enabled() -> bool:
    return not public_host_mode_enabled()
