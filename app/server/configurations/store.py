# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, select, update

from server.configurations.settings import UserSettings
from server.repositories.database.manager import DatabaseManager
from server.repositories.database.sqlite_lock import sqlite_write_lock
from server.repositories.schemas.models import ApplicationSettingsRecord, Base


###############################################################################
def _record_values(settings: UserSettings) -> dict[str, Any]:
    return {
        "id": 1,
        "runtime_host": settings.runtime.host,
        "runtime_backend_port": settings.runtime.backend_port,
        "runtime_frontend_port": settings.runtime.frontend_port,
        "datasets_allowed_extensions": list(settings.datasets.allowed_extensions),
        "datasets_column_detection_cutoff": settings.datasets.column_detection_cutoff,
        "nist_parallel_tasks": settings.nist.parallel_tasks,
        **{
            f"public_data_{name}": value
            for name, value in settings.public_data.model_dump(mode="json").items()
        },
        **{
            f"fitting_{name}": value
            for name, value in settings.fitting.model_dump(mode="json").items()
        },
        **{f"jobs_{name}": value for name, value in settings.jobs.model_dump(mode="json").items()},
        **{
            f"training_{name}": value
            for name, value in settings.training.model_dump(mode="json").items()
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


###############################################################################
def _settings_from_values(values: dict[str, Any]) -> UserSettings:
    def section(prefix: str) -> dict[str, Any]:
        return {
            key.removeprefix(f"{prefix}_"): value
            for key, value in values.items()
            if key.startswith(f"{prefix}_")
        }

    return UserSettings.model_validate(
        {
            "runtime": {
                "host": values["runtime_host"],
                "backend_port": values["runtime_backend_port"],
                "frontend_port": values["runtime_frontend_port"],
            },
            "datasets": {
                "allowed_extensions": values["datasets_allowed_extensions"],
                "column_detection_cutoff": values["datasets_column_detection_cutoff"],
            },
            "nist": {"parallel_tasks": values["nist_parallel_tasks"]},
            "public_data": section("public_data"),
            "fitting": section("fitting"),
            "jobs": section("jobs"),
            "training": section("training"),
        }
    )


###############################################################################
class ApplicationSettingsStore:
    """Atomic singleton settings record in the active database."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database
        self._write_lock = (
            sqlite_write_lock(database.engine.url.database or ":memory:")
            if database.backend == "sqlite"
            else None
        )

    # -------------------------------------------------------------------------
    @property
    def engine(self):
        return self.database.engine

    # -------------------------------------------------------------------------
    def initialize(self, defaults: UserSettings | None = None) -> UserSettings:
        lock = self._write_lock if self._write_lock is not None else _null_lock()
        with lock:
            ApplicationSettingsRecord.__table__.create(self.database.engine, checkfirst=True)
            with self.database.transaction() as session:
                exists = session.execute(
                    select(ApplicationSettingsRecord.id).where(ApplicationSettingsRecord.id == 1)
                ).first()
                if exists is None:
                    seed = defaults or UserSettings()
                    session.execute(
                        ApplicationSettingsRecord.__table__.insert().values(**_record_values(seed))
                    )
        return self.get()

    # -------------------------------------------------------------------------
    def get(self) -> UserSettings:
        with self.database.session_factory() as session:
            row = session.execute(
                select(ApplicationSettingsRecord.__table__).where(
                    ApplicationSettingsRecord.id == 1
                )
            ).mappings().first()
        if row is None:
            raise RuntimeError("Application settings have not been initialized.")
        return _settings_from_values(dict(row))

    # -------------------------------------------------------------------------
    def has_record(self) -> bool:
        with self.database.session_factory() as session:
            return session.execute(
                select(ApplicationSettingsRecord.id).where(ApplicationSettingsRecord.id == 1)
            ).first() is not None

    # -------------------------------------------------------------------------
    def save(self, settings: UserSettings) -> UserSettings:
        values = _record_values(settings)
        with self.database.transaction() as session:
            exists = session.execute(
                select(ApplicationSettingsRecord.id).where(ApplicationSettingsRecord.id == 1)
            ).first()
            if exists is None:
                session.execute(ApplicationSettingsRecord.__table__.insert().values(**values))
            else:
                session.execute(
                    update(ApplicationSettingsRecord)
                    .where(ApplicationSettingsRecord.id == 1)
                    .values(**{key: value for key, value in values.items() if key != "id"})
                )
        return self.get()

    # -------------------------------------------------------------------------
    def dispose(self) -> None:
        """The shared DatabaseManager owns the engine lifecycle."""


###############################################################################
def read_legacy_sqlite_settings(database_path: Path) -> UserSettings | None:
    """Read an existing settings row once during the database-config cutover."""
    if not database_path.is_file():
        return None
    engine = create_engine(f"sqlite:///{database_path.as_posix()}", future=True)
    try:
        if "application_settings" not in inspect(engine).get_table_names():
            return None
        with engine.connect() as connection:
            row = connection.execute(
                select(ApplicationSettingsRecord.__table__).where(
                    ApplicationSettingsRecord.id == 1
                )
            ).mappings().first()
        return _settings_from_values(dict(row)) if row is not None else None
    finally:
        engine.dispose()


###############################################################################
class _null_lock:

    # -------------------------------------------------------------------------
    def __enter__(self):
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, exc_type, exc, traceback):
        return False


SettingsBase = Base

__all__ = [
    "ApplicationSettingsRecord",
    "ApplicationSettingsStore",
    "SettingsBase",
    "read_legacy_sqlite_settings",
]
