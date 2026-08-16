from __future__ import annotations

from shared.common.settings import get_runtime_config

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

###############################################################################
def get_core_host() -> str:
    return str(get_runtime_config().get("host", "127.0.0.1"))

###############################################################################
def public_host_mode_enabled() -> bool:
    host = get_core_host().strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)
