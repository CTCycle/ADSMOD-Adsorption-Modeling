from __future__ import annotations

from adsmod_common.config import load_config
from shared.common.paths import CANONICAL_CONFIGURATION_FILE

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

###############################################################################
def get_core_host() -> str:
    return load_config(CANONICAL_CONFIGURATION_FILE).runtime.host

###############################################################################
def public_host_mode_enabled() -> bool:
    host = get_core_host().strip().lower()
    return bool(host and host not in LOOPBACK_HOSTS)
