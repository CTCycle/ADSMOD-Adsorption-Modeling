# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Shared write locks for separate SQLAlchemy engines targeting one SQLite file."""

from __future__ import annotations

from pathlib import Path
from threading import Lock, RLock

_registry_lock = Lock()
_write_locks: dict[str, RLock] = {}


def sqlite_write_lock(path: Path | str) -> RLock:
    if str(path) == ":memory:":
        key = f":memory:{id(path)}"
    else:
        key = str(Path(path).expanduser().resolve()).casefold()
    with _registry_lock:
        return _write_locks.setdefault(key, RLock())


__all__ = ["sqlite_write_lock"]
