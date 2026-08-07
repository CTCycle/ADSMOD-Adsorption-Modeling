from __future__ import annotations

import os
import re
from pathlib import Path

from adsmod_common.config import AdsmodConfig

_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")

###############################################################################
def _replace_windows_variable(match: re.Match[str]) -> str:
    return os.environ.get(match.group(1), match.group(0))

###############################################################################
def resolve_storage_root(config: AdsmodConfig) -> Path:
    raw_root = str(config.storage.root)
    expanded = _WINDOWS_VARIABLE.sub(_replace_windows_variable, raw_root)
    return Path(os.path.expandvars(expanded)).expanduser().resolve()

###############################################################################
def resolve_database_path(config: AdsmodConfig) -> Path:
    database = Path(config.storage.database)
    if database.is_absolute():
        return database.resolve()
    return (resolve_storage_root(config) / database).resolve()
