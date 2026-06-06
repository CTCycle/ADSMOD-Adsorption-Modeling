from __future__ import annotations

import os

from dotenv import load_dotenv

from ml_service.common.utils.logger import logger
from shared.common.paths import ENV_FILE


###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = ENV_FILE
        if self.env_path.exists():
            load_dotenv(self.env_path, override=True)
        else:
            logger.info(
                "Environment file not found at %s; default values will be used",
                self.env_path,
            )

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)


env_variables = EnvironmentVariables()

