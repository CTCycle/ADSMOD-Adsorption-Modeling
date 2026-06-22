from __future__ import annotations

import os

from shared.common.constants import SERVICE_CONFIG_PATH_ENV
from shared.common.paths import CORE_CONFIGURATION_FILE


###############################################################################
def configure_environment() -> None:
    os.environ.setdefault(SERVICE_CONFIG_PATH_ENV, str(CORE_CONFIGURATION_FILE))
