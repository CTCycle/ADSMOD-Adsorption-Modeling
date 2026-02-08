from __future__ import annotations

import os

from dotenv import load_dotenv

from ADSMOD.server.common.constants import ENV_FILE_PATH
from ADSMOD.server.common.utils.logger import logger


###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = ENV_FILE_PATH
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)
        else:
            logger.info(
                "Environment file not found at %s; default values will be used",
                self.env_path,
            )

    # -------------------------------------------------------------------------
    def get(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)


env_variables = EnvironmentVariables()
