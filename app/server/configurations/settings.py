# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

###############################################################################
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


DEFAULT_DATASET_ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv", ".xls", ".xlsx")
MODEL_SELECTION_METRICS = (
    "sse",
    "rmse",
    "mae",
    "r_squared",
    "adjusted_r_squared",
    "chi_square",
    "aic",
    "aicc",
    "bic",
)
MINIMIZE_MODEL_METRICS = frozenset(
    {"sse", "rmse", "mae", "chi_square", "aic", "aicc", "bic"}
)

###############################################################################
class RuntimeConfig(StrictModel):
    host: Literal["127.0.0.1", "localhost", "::1"] = "127.0.0.1"
    backend_port: int = Field(default=6045, ge=1024, le=65535)
    frontend_port: int = Field(default=5173, ge=1024, le=65535)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeConfig":
        if self.backend_port == self.frontend_port:
            raise ValueError("runtime.backend_port and runtime.frontend_port must be distinct")
        return self

###############################################################################
class StorageConfig(StrictModel):
    root: Path

###############################################################################
class DatabaseConfig(StrictModel):
    """Database bootstrap values, loaded from process environment or settings/.env."""

    embedded_database: bool = True
    engine: Literal["postgres", "postgresql", "postgresql+psycopg"] = "postgres"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1, le=300)
    sqlite_path: str = "data/database.db"

    # -------------------------------------------------------------------------
    @field_validator("host", "database_name", "username", "password", "ssl_ca", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    @field_validator("sqlite_path", mode="before")
    @classmethod
    def normalize_sqlite_path(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("DATABASE_SQLITE_PATH must not be empty")
        return text

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_selected_database(self) -> "DatabaseConfig":
        if not self.embedded_database and not all(
            (self.host, self.database_name, self.username)
        ):
            raise ValueError(
                "DATABASE_HOST, DATABASE_NAME, and DATABASE_USERNAME are required when EMBEDDED_DATABASE is false"
            )
        return self

###############################################################################
class DatasetConfig(StrictModel):
    allowed_extensions: tuple[Literal[".csv", ".xls", ".xlsx"], ...] = DEFAULT_DATASET_ALLOWED_EXTENSIONS
    column_detection_cutoff: float = Field(default=0.6, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_DATASET_ALLOWED_EXTENSIONS
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value]
        else:
            raise ValueError("datasets.allowed_extensions must be a sequence or string")
        cleaned = tuple(
            (part.strip().casefold() if part.strip().startswith(".") else f".{part.strip().casefold()}")
            for part in values
            if str(part).strip()
        )
        if not cleaned:
            raise ValueError("datasets.allowed_extensions must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("datasets.allowed_extensions cannot contain duplicates")
        return cleaned  # type: ignore[return-value]

###############################################################################
class NISTConfig(StrictModel):
    parallel_tasks: int = Field(default=40, ge=1, le=100)

###############################################################################
class PublicDataConfig(StrictModel):
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    retry_attempts: int = Field(default=3, ge=1, le=6)
    retry_backoff_max_seconds: float = Field(default=30.0, ge=0.1, le=120.0)
    pubchem_parallel_requests: int = Field(default=2, ge=1, le=3)
    cod_max_interactive_results: int = Field(default=250, ge=1, le=1000)
    health_cache_ttl_seconds: int = Field(default=900, ge=30, le=86400)
    sync_batch_size: int = Field(default=50, ge=1, le=500)
    sync_max_records_per_run: int = Field(default=200, ge=1, le=10000)
    sync_max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    sync_response_size_limit_bytes: int = Field(default=67108864, ge=1024, le=268435456)
    sync_resume_on_startup: bool = False
    auto_sync_enabled: bool = False
    auto_sync_categories: tuple[Literal["experiments", "guest", "host"], ...] = ()
    auto_sync_interval_hours: int = Field(default=24, ge=1, le=720)
    auto_sync_fraction: float = Field(default=1.0, gt=0.0, le=1.0)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_auto_sync_scope(self) -> "PublicDataConfig":
        if self.auto_sync_enabled and not self.auto_sync_categories:
            raise ValueError(
                "auto_sync_categories must include at least one NIST collection when automatic sync is enabled."
            )
        if len(set(self.auto_sync_categories)) != len(self.auto_sync_categories):
            raise ValueError("auto_sync_categories cannot contain duplicates.")
        return self

###############################################################################
class FittingConfig(StrictModel):
    default_max_iterations: int = Field(default=1000, ge=10, le=1000000000)
    max_iterations_upper_bound: int = Field(default=1000000000, ge=10, le=1000000000)
    default_parameter_initial: float = Field(default=1.0, ge=0.0)
    default_parameter_min: float = Field(default=0.0, ge=0.0)
    default_parameter_max: float = Field(default=100.0, ge=0.0)
    preview_row_limit: int = Field(default=20, ge=1, le=500)
    best_model_metric: Literal["sse", "rmse", "mae", "r_squared", "adjusted_r_squared", "chi_square", "aic", "aicc", "bic"] = "aicc"

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_bounds(self) -> "FittingConfig":
        if self.max_iterations_upper_bound < self.default_max_iterations:
            raise ValueError(
                "fitting.max_iterations_upper_bound must be >= fitting.default_max_iterations"
            )
        if self.default_parameter_max <= self.default_parameter_min:
            raise ValueError(
                "fitting.default_parameter_max must be greater than fitting.default_parameter_min"
            )
        if not self.default_parameter_min <= self.default_parameter_initial <= self.default_parameter_max:
            raise ValueError(
                "fitting.default_parameter_initial must be between the minimum and maximum"
            )
        return self

###############################################################################
class JobConfig(StrictModel):
    polling_interval: float = Field(default=1.0, ge=0.1, le=60.0)

###############################################################################
class TrainingConfig(StrictModel):
    use_jit: bool = False
    jit_backend: Literal["inductor", "eager", "aot_eager", "cudagraphs"] = "inductor"
    use_mixed_precision: bool = False
    dataloader_workers: int = Field(default=0, ge=0, le=64)
    persistent_workers: bool = False

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_worker_settings(self) -> "TrainingConfig":
        if self.persistent_workers and self.dataloader_workers < 1:
            raise ValueError("persistent_workers requires dataloader_workers to be greater than zero")
        return self

###############################################################################
class UserSettings(StrictModel):
    """All application behavior settings editable in the console."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    nist: NISTConfig = Field(default_factory=NISTConfig)
    public_data: PublicDataConfig = Field(default_factory=PublicDataConfig)
    fitting: FittingConfig = Field(default_factory=FittingConfig)
    jobs: JobConfig = Field(default_factory=JobConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

###############################################################################
class ApplicationConfig(StrictModel):
    database: DatabaseConfig
    datasets: DatasetConfig
    nist: NISTConfig
    public_data: PublicDataConfig
    fitting: FittingConfig
    jobs: JobConfig
    training: TrainingConfig

###############################################################################
class AdsmodConfig(StrictModel):
    version: Literal["3.0.0"] = "3.0.0"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig
    application: ApplicationConfig

    # -------------------------------------------------------------------------
    @classmethod
    def assemble(
        cls,
        *,
        database: DatabaseConfig,
        storage_root: Path,
        settings: UserSettings,
    ) -> "AdsmodConfig":
        return cls(
            runtime=settings.runtime,
            storage=StorageConfig(root=storage_root),
            application=ApplicationConfig(
                database=database,
                datasets=settings.datasets,
                nist=settings.nist,
                public_data=settings.public_data,
                fitting=settings.fitting,
                jobs=settings.jobs,
                training=settings.training,
            ),
        )

    # -------------------------------------------------------------------------
    def user_settings(self) -> UserSettings:
        return UserSettings(
            runtime=self.runtime,
            datasets=self.application.datasets,
            nist=self.application.nist,
            public_data=self.application.public_data,
            fitting=self.application.fitting,
            jobs=self.application.jobs,
            training=self.application.training,
        )


__all__ = [
    "AdsmodConfig",
    "ApplicationConfig",
    "DatabaseConfig",
    "DatasetConfig",
    "FittingConfig",
    "JobConfig",
    "MINIMIZE_MODEL_METRICS",
    "MODEL_SELECTION_METRICS",
    "NISTConfig",
    "PublicDataConfig",
    "RuntimeConfig",
    "StorageConfig",
    "TrainingConfig",
    "UserSettings",
]
