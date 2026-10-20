# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Float, Integer, JSON, String, create_engine, select, update
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from server.configurations.settings import UserSettings
from server.repositories.database.sqlite_lock import sqlite_write_lock


class SettingsBase(DeclarativeBase):
    pass


class ApplicationSettingsRecord(SettingsBase):
    __tablename__ = "application_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    runtime_host: Mapped[str] = mapped_column(String(16), nullable=False)
    runtime_backend_port: Mapped[int] = mapped_column(Integer, nullable=False)
    runtime_frontend_port: Mapped[int] = mapped_column(Integer, nullable=False)
    datasets_allowed_extensions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    datasets_column_detection_cutoff: Mapped[float] = mapped_column(Float, nullable=False)
    nist_parallel_tasks: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_request_timeout_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    public_data_retry_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_retry_backoff_max_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    public_data_pubchem_parallel_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_cod_max_interactive_results: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_health_cache_ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_sync_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_sync_max_records_per_run: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_sync_max_concurrent_jobs: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_sync_response_size_limit_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_sync_resume_on_startup: Mapped[bool] = mapped_column(Boolean, nullable=False)
    public_data_auto_sync_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    public_data_auto_sync_categories: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    public_data_auto_sync_interval_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    public_data_auto_sync_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    fitting_default_max_iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    fitting_max_iterations_upper_bound: Mapped[int] = mapped_column(Integer, nullable=False)
    fitting_default_parameter_initial: Mapped[float] = mapped_column(Float, nullable=False)
    fitting_default_parameter_min: Mapped[float] = mapped_column(Float, nullable=False)
    fitting_default_parameter_max: Mapped[float] = mapped_column(Float, nullable=False)
    fitting_preview_row_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    fitting_best_model_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    jobs_polling_interval: Mapped[float] = mapped_column(Float, nullable=False)
    training_use_jit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    training_jit_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    training_use_mixed_precision: Mapped[bool] = mapped_column(Boolean, nullable=False)
    training_dataloader_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    training_persistent_workers: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)


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


def _settings_from_values(values: dict[str, Any]) -> UserSettings:
    public_data = {
        key.removeprefix("public_data_"): value
        for key, value in values.items()
        if key.startswith("public_data_")
    }
    fitting = {
        key.removeprefix("fitting_"): value
        for key, value in values.items()
        if key.startswith("fitting_")
    }
    jobs = {
        key.removeprefix("jobs_"): value
        for key, value in values.items()
        if key.startswith("jobs_")
    }
    training = {
        key.removeprefix("training_"): value
        for key, value in values.items()
        if key.startswith("training_")
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
            "public_data": public_data,
            "fitting": fitting,
            "jobs": jobs,
            "training": training,
        }
    )


class ApplicationSettingsStore:
    """Atomic singleton settings record in the local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        if str(database_path) == ":memory:":
            self.database_path = Path(":memory:")
            self._write_lock = sqlite_write_lock(":memory:")
            self.engine = create_engine(
                "sqlite://",
                future=True,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self.database_path = database_path.resolve()
            self._write_lock = sqlite_write_lock(self.database_path)
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f"sqlite:///{self.database_path.as_posix()}",
                future=True,
                connect_args={"timeout": 30},
            )

    def initialize(self, defaults: UserSettings | None = None) -> UserSettings:
        with self._write_lock:
            SettingsBase.metadata.create_all(self.engine)
            with self.engine.begin() as connection:
                exists = connection.execute(
                    select(ApplicationSettingsRecord.id).where(
                        ApplicationSettingsRecord.id == 1
                    )
                ).first()
                if exists is None:
                    seed = defaults or UserSettings()
                    connection.execute(
                        ApplicationSettingsRecord.__table__.insert().values(
                            **_record_values(seed)
                        )
                    )
        return self.get()

    def get(self) -> UserSettings:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(ApplicationSettingsRecord.__table__).where(
                    ApplicationSettingsRecord.id == 1
                )
            ).mappings().first()
        if row is None:
            raise RuntimeError("Application settings have not been initialized.")
        return _settings_from_values(dict(row))

    def save(self, settings: UserSettings) -> UserSettings:
        values = _record_values(settings)
        with self._write_lock:
            with self.engine.begin() as connection:
                exists = connection.execute(
                    select(ApplicationSettingsRecord.id)
                    .where(ApplicationSettingsRecord.id == 1)
                ).first()
                if exists is None:
                    connection.execute(
                        ApplicationSettingsRecord.__table__.insert().values(**values)
                    )
                else:
                    connection.execute(
                        update(ApplicationSettingsRecord)
                        .where(ApplicationSettingsRecord.id == 1)
                        .values(**{key: value for key, value in values.items() if key != "id"})
                    )
        return self.get()

    def dispose(self) -> None:
        self.engine.dispose()


__all__ = ["ApplicationSettingsRecord", "ApplicationSettingsStore", "SettingsBase"]
