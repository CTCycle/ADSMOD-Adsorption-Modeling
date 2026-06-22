from __future__ import annotations

import os

from shared.common.constants import SERVICE_CONFIG_PATH_ENV
from shared.common.paths import ML_CONFIGURATION_FILE


###############################################################################
def configure_environment() -> None:
    os.environ.setdefault(SERVICE_CONFIG_PATH_ENV, str(ML_CONFIGURATION_FILE))
    os.environ.setdefault("KERAS_BACKEND", "torch")
    os.environ.setdefault("MPLBACKEND", "Agg")
