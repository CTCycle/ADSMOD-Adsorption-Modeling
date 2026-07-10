from __future__ import annotations

import os
import re
from pathlib import Path

from adsmod_common.config import AdsmodConfig

_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")


def resolve_storage_root(config: AdsmodConfig) -> Path:
    raw_root = str(config.storage.root)

    def replace_variable(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    expanded = _WINDOWS_VARIABLE.sub(replace_variable, raw_root)
    return Path(os.path.expandvars(expanded)).expanduser().resolve()


def resolve_database_path(config: AdsmodConfig) -> Path:
    database = Path(config.storage.database)
    if database.is_absolute():
        return database.resolve()
    return (resolve_storage_root(config) / database).resolve()