from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adsmod_common.config import (
    AdsmodConfig,
    DatabaseConfig,
    DatasetConfig,
    FittingConfig,
    JobConfig,
    NISTConfig,
    TrainingConfig,
    load_config,
)

from shared.common.paths import CANONICAL_CONFIGURATION_FILE

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
    sqlite_path: str | None = None

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
def build_database_settings(config: DatabaseConfig) -> DatabaseSettings:
    if config.embedded_database:
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
            connect_timeout=config.connect_timeout,
            insert_batch_size=config.insert_batch_size,
            sqlite_path=config.sqlite_path,
        )

    return DatabaseSettings(
        embedded_database=False,
        engine=config.engine.strip().lower(),
        host=config.host,
        port=config.port,
        database_name=config.database_name,
        username=config.username,
        password=config.password,
        ssl=config.ssl,
        ssl_ca=config.ssl_ca,
        connect_timeout=config.connect_timeout,
        insert_batch_size=config.insert_batch_size,
        sqlite_path=config.sqlite_path,
    )

###############################################################################
def build_dataset_settings(config: DatasetConfig) -> DatasetSettings:
    return DatasetSettings(
        allowed_extensions=config.allowed_extensions,
        column_detection_cutoff=config.column_detection_cutoff,
    )

###############################################################################
def build_nist_settings(config: NISTConfig) -> NISTSettings:
    return NISTSettings(
        parallel_tasks=config.parallel_tasks,
        pubchem_parallel_tasks=config.pubchem_parallel_tasks,
    )

###############################################################################
def build_fitting_settings(config: FittingConfig) -> FittingSettings:
    return FittingSettings(
        default_max_iterations=config.default_max_iterations,
        max_iterations_upper_bound=config.max_iterations_upper_bound,
        parameter_initial_default=config.default_parameter_initial,
        parameter_min_default=config.default_parameter_min,
        parameter_max_default=config.default_parameter_max,
        preview_row_limit=config.preview_row_limit,
        best_model_metric=config.best_model_metric,
    )

###############################################################################
def build_job_settings(config: JobConfig) -> JobSettings:
    return JobSettings(polling_interval=config.polling_interval)

###############################################################################
def build_training_settings(config: TrainingConfig) -> TrainingSettings:
    return TrainingSettings(
        use_jit=config.use_jit,
        jit_backend=config.jit_backend,
        use_mixed_precision=config.use_mixed_precision,
        dataloader_workers=config.dataloader_workers,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=config.persistent_workers,
        plot_update_batch_interval=DEFAULT_PLOT_UPDATE_BATCH_INTERVAL,
    )

###############################################################################
def to_server_settings(config: AdsmodConfig) -> ServerSettings:
    application = config.application
    return ServerSettings(
        database=build_database_settings(application.database),
        datasets=build_dataset_settings(application.datasets),
        nist=build_nist_settings(application.nist),
        fitting=build_fitting_settings(application.fitting),
        jobs=build_job_settings(application.jobs),
        training=build_training_settings(application.training),
    )

###############################################################################
def get_server_settings(config_path: str | Path | None = None) -> ServerSettings:
    path = Path(config_path) if config_path else CANONICAL_CONFIGURATION_FILE
    return to_server_settings(load_config(path))


__all__ = [
    "DatabaseSettings",
    "DatasetSettings",
    "FittingSettings",
    "JobSettings",
    "NISTSettings",
    "ServerSettings",
    "TrainingSettings",
    "build_database_settings",
    "build_dataset_settings",
    "build_fitting_settings",
    "build_job_settings",
    "build_nist_settings",
    "build_training_settings",
    "get_server_settings",
    "to_server_settings",
]
