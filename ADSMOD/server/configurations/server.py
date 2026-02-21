from __future__ import annotations

from typing import Any

from ADSMOD.server.common.utils.types import coerce_str_sequence

from ADSMOD.server.configurations import (
    ensure_mapping,
    load_configurations,
)
from ADSMOD.server.common.constants import CONFIGURATION_FILE
from ADSMOD.server.common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str,
    coerce_str_or_none,
)
from ADSMOD.server.entities.settings import (
    DatabaseSettings,
    DatasetSettings,
    FittingSettings,
    JobSettings,
    NISTSettings,
    ServerSettings,
    TrainingSettings,
)
from ADSMOD.server.common.utils.variables import env_variables

DEFAULT_PREFETCH_FACTOR = 1
DEFAULT_PIN_MEMORY = True
PLOT_UPDATE_BATCH_INTERVAL = 10
DEFAULT_DB_EMBEDDED = True
DEFAULT_DB_ENGINE = "postgres"
DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_NAME = "ADSMOD"
DEFAULT_DB_USER = "postgres"
DEFAULT_DB_PASSWORD = "admin"
DEFAULT_DB_SSL = False
DEFAULT_DB_CONNECT_TIMEOUT = 30
DEFAULT_DB_INSERT_BATCH_SIZE = 5000

# [BUILDER FUNCTIONS]
###############################################################################
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    embedded = coerce_bool(env_variables.get("DB_EMBEDDED"), DEFAULT_DB_EMBEDDED)

    if embedded:
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
            connect_timeout=coerce_int(
                env_variables.get("DB_CONNECT_TIMEOUT"),
                DEFAULT_DB_CONNECT_TIMEOUT,
                minimum=1,
            ),
            insert_batch_size=coerce_int(
                env_variables.get("DB_INSERT_BATCH_SIZE"),
                DEFAULT_DB_INSERT_BATCH_SIZE,
                minimum=1,
            ),
        )

    engine_value = (
        coerce_str_or_none(env_variables.get("DB_ENGINE"))
        or DEFAULT_DB_ENGINE
    )
    normalized_engine = engine_value.lower() if engine_value else None

    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str(
            env_variables.get("DB_HOST"),
            DEFAULT_DB_HOST,
        ),
        port=coerce_int(
            env_variables.get("DB_PORT"), DEFAULT_DB_PORT, minimum=1, maximum=65535
        ),
        database_name=coerce_str(
            env_variables.get("DB_NAME"),
            DEFAULT_DB_NAME,
        ),
        username=coerce_str(
            env_variables.get("DB_USER"),
            DEFAULT_DB_USER,
        ),
        password=coerce_str(
            env_variables.get("DB_PASSWORD"),
            DEFAULT_DB_PASSWORD,
        ),
        ssl=coerce_bool(env_variables.get("DB_SSL"), DEFAULT_DB_SSL),
        ssl_ca=coerce_str_or_none(env_variables.get("DB_SSL_CA")),
        connect_timeout=coerce_int(
            env_variables.get("DB_CONNECT_TIMEOUT"), DEFAULT_DB_CONNECT_TIMEOUT, minimum=1
        ),
        insert_batch_size=coerce_int(
            env_variables.get("DB_INSERT_BATCH_SIZE"),
            DEFAULT_DB_INSERT_BATCH_SIZE,
            minimum=1,
        ),
    )


# -------------------------------------------------------------------------
def build_dataset_settings(payload: dict[str, Any] | Any) -> DatasetSettings:
    return DatasetSettings(
        allowed_extensions=coerce_str_sequence(
            payload.get("allowed_extensions"), [".csv", ".xls", ".xlsx"]
        ),
        column_detection_cutoff=coerce_float(
            payload.get("column_detection_cutoff"), 0.6, minimum=0.0, maximum=1.0
        ),
    )


# -------------------------------------------------------------------------
def build_nist_settings(payload: dict[str, Any] | Any) -> NISTSettings:
    return NISTSettings(
        parallel_tasks=coerce_int(payload.get("parallel_tasks"), 20, minimum=1),
        pubchem_parallel_tasks=coerce_int(
            payload.get("pubchem_parallel_tasks"), 10, minimum=1
        ),
    )


# -------------------------------------------------------------------------
def build_fitting_settings(payload: dict[str, Any] | Any) -> FittingSettings:
    default_iterations = coerce_int(
        payload.get("default_max_iterations"), 1000, minimum=1
    )
    upper_bound = coerce_int(
        payload.get("max_iterations_upper_bound"), 1_000_000, minimum=default_iterations
    )
    parameter_initial_default = coerce_float(
        payload.get("default_parameter_initial"), 1.0, minimum=0.0
    )
    parameter_min_default = coerce_float(
        payload.get("default_parameter_min"), 0.0, minimum=0.0
    )
    parameter_max_default = coerce_float(
        payload.get("default_parameter_max"), 100.0, minimum=parameter_min_default
    )
    best_model_metric = coerce_str(payload.get("best_model_metric"), "AICc")
    return FittingSettings(
        default_max_iterations=default_iterations,
        max_iterations_upper_bound=upper_bound,
        parameter_initial_default=parameter_initial_default,
        parameter_min_default=parameter_min_default,
        parameter_max_default=parameter_max_default,
        preview_row_limit=coerce_int(payload.get("preview_row_limit"), 5, minimum=1),
        best_model_metric=best_model_metric,
    )


# -------------------------------------------------------------------------
def build_job_settings(payload: dict[str, Any] | Any) -> JobSettings:
    return JobSettings(
        polling_interval=coerce_float(payload.get("polling_interval"), 1.0),
    )


# -------------------------------------------------------------------------
def build_training_settings(payload: dict[str, Any] | Any) -> TrainingSettings:
    return TrainingSettings(
        use_jit=coerce_bool(payload.get("use_jit"), False),
        jit_backend=coerce_str(payload.get("jit_backend"), "inductor"),
        use_mixed_precision=coerce_bool(payload.get("use_mixed_precision"), False),
        dataloader_workers=coerce_int(payload.get("dataloader_workers"), 0, minimum=0),
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=coerce_bool(payload.get("persistent_workers"), False),
        plot_update_batch_interval=PLOT_UPDATE_BATCH_INTERVAL,
    )


# -------------------------------------------------------------------------
def build_server_settings(payload: dict[str, Any] | Any) -> ServerSettings:
    dataset_payload = ensure_mapping(payload.get("datasets"))
    nist_payload = ensure_mapping(payload.get("nist"))
    fitting_payload = ensure_mapping(payload.get("fitting"))
    jobs_payload = ensure_mapping(payload.get("jobs"))
    training_payload = ensure_mapping(payload.get("training"))

    return ServerSettings(
        database=build_database_settings({}),
        datasets=build_dataset_settings(dataset_payload),
        nist=build_nist_settings(nist_payload),
        fitting=build_fitting_settings(fitting_payload),
        jobs=build_job_settings(jobs_payload),
        training=build_training_settings(training_payload),
    )


# [SERVER CONFIGURATION LOADER]
###############################################################################
# -------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    path = config_path or CONFIGURATION_FILE
    payload = load_configurations(path)

    return build_server_settings(payload)


server_settings = get_server_settings()
