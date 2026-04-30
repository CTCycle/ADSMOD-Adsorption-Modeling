from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.server.common.constants import CONFIGURATION_FILE


DEFAULT_ALLOWED_EXTENSIONS = (".csv", ".xls", ".xlsx")
DEFAULT_PREFETCH_FACTOR = 1
DEFAULT_PIN_MEMORY = True
DEFAULT_PLOT_UPDATE_BATCH_INTERVAL = 10


###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
    engine: str | None
    host: str | None
    port: int | None
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool
    ssl_ca: str | None
    connect_timeout: int
    insert_batch_size: int


###############################################################################
@dataclass(frozen=True)
class DatasetSettings:
    allowed_extensions: tuple[str, ...]
    column_detection_cutoff: float


###############################################################################
@dataclass(frozen=True)
class NISTSettings:
    parallel_tasks: int
    pubchem_parallel_tasks: int


###############################################################################
@dataclass(frozen=True)
class FittingSettings:
    default_max_iterations: int
    max_iterations_upper_bound: int
    parameter_initial_default: float
    parameter_min_default: float
    parameter_max_default: float
    preview_row_limit: int
    best_model_metric: str


###############################################################################
@dataclass(frozen=True)
class JobSettings:
    polling_interval: float


###############################################################################
@dataclass(frozen=True)
class TrainingSettings:
    use_jit: bool
    jit_backend: str
    use_mixed_precision: bool
    dataloader_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    plot_update_batch_interval: int


###############################################################################
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    datasets: DatasetSettings
    nist: NISTSettings
    fitting: FittingSettings
    jobs: JobSettings
    training: TrainingSettings


###############################################################################
class JsonDatabaseSettings(BaseModel):
    embedded_database: bool = True
    engine: str = "postgres"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1)
    insert_batch_size: int = Field(default=5000, ge=1)

    @field_validator(
        "host",
        "database_name",
        "username",
        "password",
        "ssl_ca",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("engine", mode="before")
    @classmethod
    def normalize_engine(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "postgres"

class JsonDatasetSettings(BaseModel):
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    column_detection_cutoff: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return DEFAULT_ALLOWED_EXTENSIONS
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value]
        else:
            return DEFAULT_ALLOWED_EXTENSIONS

        cleaned = tuple(part.strip() for part in values if str(part).strip())
        return cleaned or DEFAULT_ALLOWED_EXTENSIONS


###############################################################################
class JsonNISTSettings(BaseModel):
    parallel_tasks: int = Field(default=20, ge=1)
    pubchem_parallel_tasks: int = Field(default=10, ge=1)


###############################################################################
class JsonFittingSettings(BaseModel):
    default_max_iterations: int = Field(default=1000, ge=1)
    max_iterations_upper_bound: int = Field(default=1_000_000, ge=1)
    default_parameter_initial: float = Field(default=1.0, ge=0.0)
    default_parameter_min: float = Field(default=0.0, ge=0.0)
    default_parameter_max: float = Field(default=100.0, ge=0.0)
    preview_row_limit: int = Field(default=5, ge=1)
    best_model_metric: str = "AICc"

    @field_validator("best_model_metric", mode="before")
    @classmethod
    def normalize_metric(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "AICc"

    @model_validator(mode="after")
    def validate_bounds(self) -> "JsonFittingSettings":
        if self.max_iterations_upper_bound < self.default_max_iterations:
            raise ValueError(
                "fitting.max_iterations_upper_bound must be >= fitting.default_max_iterations"
            )
        if self.default_parameter_max < self.default_parameter_min:
            raise ValueError(
                "fitting.default_parameter_max must be >= fitting.default_parameter_min"
            )
        return self


###############################################################################
class JsonJobSettings(BaseModel):
    polling_interval: float = 1.0


###############################################################################
class JsonTrainingSettings(BaseModel):
    use_jit: bool = False
    jit_backend: str = "inductor"
    use_mixed_precision: bool = False
    dataloader_workers: int = Field(default=0, ge=0)
    persistent_workers: bool = False

    @field_validator("jit_backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "inductor"


###############################################################################
def _load_configuration_payload(path: str) -> dict[str, Any]:
    configuration_file = Path(path)
    if not configuration_file.exists():
        raise RuntimeError(f"Configuration file not found: {configuration_file}")

    try:
        payload = json.loads(configuration_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load configuration from {configuration_file}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Configuration must be a JSON object.")
    return payload


###############################################################################
class AppSettings(BaseModel):
    _configuration_file: ClassVar[str] = CONFIGURATION_FILE

    database: JsonDatabaseSettings = Field(default_factory=JsonDatabaseSettings)
    datasets: JsonDatasetSettings = Field(default_factory=JsonDatasetSettings)
    nist: JsonNISTSettings = Field(default_factory=JsonNISTSettings)
    fitting: JsonFittingSettings = Field(default_factory=JsonFittingSettings)
    jobs: JsonJobSettings = Field(default_factory=JsonJobSettings)
    training: JsonTrainingSettings = Field(default_factory=JsonTrainingSettings)

    @classmethod
    def load(cls) -> "AppSettings":
        payload = _load_configuration_payload(getattr(cls, "_configuration_file"))

        values: dict[str, Any] = {
            "database": payload.get("database", {}),
            "datasets": payload.get("datasets", {}),
            "nist": payload.get("nist", {}),
            "fitting": payload.get("fitting", {}),
            "jobs": payload.get("jobs", {}),
            "training": payload.get("training", {}),
        }
        return cls.model_validate(values)

    # -------------------------------------------------------------------------
    def to_server_settings(self) -> ServerSettings:
        return ServerSettings(
            database=build_database_settings(self.database.model_dump(mode="python")),
            datasets=build_dataset_settings(self.datasets.model_dump(mode="python")),
            nist=build_nist_settings(self.nist.model_dump(mode="python")),
            fitting=build_fitting_settings(self.fitting.model_dump(mode="python")),
            jobs=build_job_settings(self.jobs.model_dump(mode="python")),
            training=build_training_settings(self.training.model_dump(mode="python")),
        )


###############################################################################
def _ensure_mapping(value: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


# -----------------------------------------------------------------------------
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    json_settings = JsonDatabaseSettings.model_validate(_ensure_mapping(payload))
    if json_settings.embedded_database:
        return DatabaseSettings(
            embedded_database=True,
            engine=None,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=json_settings.connect_timeout,
            insert_batch_size=json_settings.insert_batch_size,
        )

    normalized_engine = json_settings.engine.strip().lower()
    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=json_settings.host,
        port=json_settings.port,
        database_name=json_settings.database_name,
        username=json_settings.username,
        password=json_settings.password,
        ssl=json_settings.ssl,
        ssl_ca=json_settings.ssl_ca,
        connect_timeout=json_settings.connect_timeout,
        insert_batch_size=json_settings.insert_batch_size,
    )


# -----------------------------------------------------------------------------
def build_dataset_settings(payload: dict[str, Any] | Any) -> DatasetSettings:
    json_settings = JsonDatasetSettings.model_validate(_ensure_mapping(payload))
    return DatasetSettings(
        allowed_extensions=json_settings.allowed_extensions,
        column_detection_cutoff=json_settings.column_detection_cutoff,
    )


# -----------------------------------------------------------------------------
def build_nist_settings(payload: dict[str, Any] | Any) -> NISTSettings:
    json_settings = JsonNISTSettings.model_validate(_ensure_mapping(payload))
    return NISTSettings(
        parallel_tasks=json_settings.parallel_tasks,
        pubchem_parallel_tasks=json_settings.pubchem_parallel_tasks,
    )


# -----------------------------------------------------------------------------
def build_fitting_settings(payload: dict[str, Any] | Any) -> FittingSettings:
    json_settings = JsonFittingSettings.model_validate(_ensure_mapping(payload))
    return FittingSettings(
        default_max_iterations=json_settings.default_max_iterations,
        max_iterations_upper_bound=json_settings.max_iterations_upper_bound,
        parameter_initial_default=json_settings.default_parameter_initial,
        parameter_min_default=json_settings.default_parameter_min,
        parameter_max_default=json_settings.default_parameter_max,
        preview_row_limit=json_settings.preview_row_limit,
        best_model_metric=json_settings.best_model_metric,
    )


# -----------------------------------------------------------------------------
def build_job_settings(payload: dict[str, Any] | Any) -> JobSettings:
    json_settings = JsonJobSettings.model_validate(_ensure_mapping(payload))
    return JobSettings(
        polling_interval=json_settings.polling_interval,
    )


# -----------------------------------------------------------------------------
def build_training_settings(payload: dict[str, Any] | Any) -> TrainingSettings:
    json_settings = JsonTrainingSettings.model_validate(_ensure_mapping(payload))
    return TrainingSettings(
        use_jit=json_settings.use_jit,
        jit_backend=json_settings.jit_backend,
        use_mixed_precision=json_settings.use_mixed_precision,
        dataloader_workers=json_settings.dataloader_workers,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=json_settings.persistent_workers,
        plot_update_batch_interval=DEFAULT_PLOT_UPDATE_BATCH_INTERVAL,
    )


# -----------------------------------------------------------------------------
def build_server_settings(payload: dict[str, Any] | Any) -> ServerSettings:
    section_payload = _ensure_mapping(payload)
    return ServerSettings(
        database=build_database_settings(section_payload.get("database")),
        datasets=build_dataset_settings(section_payload.get("datasets")),
        nist=build_nist_settings(section_payload.get("nist")),
        fitting=build_fitting_settings(section_payload.get("fitting")),
        jobs=build_job_settings(section_payload.get("jobs")),
        training=build_training_settings(section_payload.get("training")),
    )


__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "DatasetSettings",
    "NISTSettings",
    "FittingSettings",
    "JobSettings",
    "TrainingSettings",
    "ServerSettings",
    "build_database_settings",
    "build_dataset_settings",
    "build_nist_settings",
    "build_fitting_settings",
    "build_job_settings",
    "build_training_settings",
    "build_server_settings",
    "ValidationError",
]

