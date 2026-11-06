# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Cheap process and database resource sampling for the status bar."""

from __future__ import annotations

import time
from pathlib import Path

import psutil
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from server.configurations.settings import DatabaseConfig
from server.domain.system import SystemStatusResponse
from server.repositories.database.manager import DatabaseManager, resolve_sqlite_path

DATABASE_SIZE_TTL_SECONDS = 30.0


###############################################################################
class SystemStatusSampler:
    """Report backend CPU, memory and database size without blocking polling."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        self._size_key: tuple[object, ...] | None = None
        self._size_value: int | None = None
        self._size_at = 0.0

    # -------------------------------------------------------------------------
    def sample(
        self,
        database: DatabaseConfig,
        manager: DatabaseManager,
        storage_root: Path,
    ) -> SystemStatusResponse:
        cpu = self.process.cpu_percent(interval=None) / max(1, psutil.cpu_count() or 1)
        return SystemStatusResponse(
            cpu_percent=round(min(100.0, cpu), 1),
            memory_mb=round(self.process.memory_info().rss / 1024**2, 1),
            database_size_bytes=self._database_size(database, manager, storage_root),
        )

    # -------------------------------------------------------------------------
    def _database_size(
        self, database: DatabaseConfig, manager: DatabaseManager, storage_root: Path
    ) -> int | None:
        key = (database.embedded_database, database.sqlite_path, database.database_name)
        now = time.monotonic()
        if key == self._size_key and now - self._size_at < DATABASE_SIZE_TTL_SECONDS:
            return self._size_value
        if database.embedded_database:
            size = self._sqlite_size(database, storage_root)
        else:
            size = self._postgres_size(manager)
        self._size_key, self._size_value, self._size_at = key, size, now
        return size

    # -------------------------------------------------------------------------
    @staticmethod
    def _sqlite_size(database: DatabaseConfig, storage_root: Path) -> int | None:
        path = resolve_sqlite_path(database, storage_root=storage_root)
        try:
            return sum(
                candidate.stat().st_size
                for candidate in (path, Path(f"{path}-wal"))
                if candidate.exists()
            )
        except OSError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _postgres_size(manager: DatabaseManager) -> int | None:
        try:
            with manager.engine.connect() as connection:
                return int(
                    connection.execute(
                        text("SELECT pg_database_size(current_database())")
                    ).scalar_one()
                )
        except SQLAlchemyError:
            return None


__all__ = ["SystemStatusSampler"]
