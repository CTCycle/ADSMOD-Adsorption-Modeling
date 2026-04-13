from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from ADSMOD.server.common.constants import ENV_FILE_PATH
from ADSMOD.server.common.utils.logger import logger


###############################################################################
def load_environment(
    env_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Path | None:
    _ = force
    resolved_path = Path(env_path) if env_path is not None else Path(ENV_FILE_PATH)
    if not resolved_path.exists():
        logger.warning(".env file not found at: %s", resolved_path)
        return None

    load_dotenv(dotenv_path=resolved_path, override=True)
    return resolved_path


# -----------------------------------------------------------------------------
def reset_environment_for_tests(keys: Iterable[str]) -> None:
    for key in keys:
        os.environ.pop(str(key), None)


