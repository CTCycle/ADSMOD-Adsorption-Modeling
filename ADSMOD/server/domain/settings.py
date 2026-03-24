from __future__ import annotations

from dataclasses import dataclass


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
