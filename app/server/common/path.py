from __future__ import annotations

import os
from pathlib import Path
import re

from server.configurations.settings import AdsmodConfig

_WINDOWS_VARIABLE = re.compile(r"%([^%]+)%")

###############################################################################
def _expand(value: str | Path) -> Path:
    raw = str(value)
    expanded = _WINDOWS_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)),
        raw,
    )
    return Path(os.path.expandvars(expanded)).expanduser().resolve()

###############################################################################
def resolve_storage_root(config: AdsmodConfig) -> Path:
    return _expand(config.storage.root)

###############################################################################
def resolve_checkpoint_root(config: AdsmodConfig) -> Path:
    return resolve_storage_root(config) / "checkpoints"

###############################################################################
def resolve_log_root(config: AdsmodConfig) -> Path:
    return resolve_storage_root(config) / "logs"


###############################################################################
def resolve_database_path(config: AdsmodConfig) -> Path:
    database_value = config.application.database.sqlite_path
    if not database_value:
        raise ValueError(
            "application.database.sqlite_path is required for embedded databases"
        )
    expanded_database = _WINDOWS_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)),
        database_value,
    )
    database = Path(os.path.expandvars(expanded_database)).expanduser()
    if database.is_absolute():
        return database.resolve()
    return (resolve_storage_root(config) / database).resolve()


__all__ = [
    "resolve_checkpoint_root",
    "resolve_database_path",
    "resolve_log_root",
    "resolve_storage_root",
]
