from __future__ import annotations

DATABASE_FILENAME = "database.db"


###############################################################################
FASTAPI_TITLE = "ADSMOD Model Fitting Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_VERSION = "3.0.0"
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


###############################################################################
COLUMN_ID = "id"
COLUMN_EXPERIMENT = "experiment"
COLUMN_EXPERIMENT_NAME = "experiment name"
COLUMN_DATASET_NAME = "dataset_name"
COLUMN_FILENAME = "filename"
COLUMN_ADSORBENT = "adsorbent"
COLUMN_ADSORBATE = "adsorbate"
COLUMN_TEMPERATURE_K = "temperature [K]"
COLUMN_PRESSURE_PA = "pressure [Pa]"
COLUMN_UPTAKE_MOL_G = "uptake [mol/g]"
COLUMN_MEASUREMENT_COUNT = "measurement_count"
COLUMN_MIN_PRESSURE = "min_pressure"
COLUMN_MAX_PRESSURE = "max_pressure"
COLUMN_MIN_UPTAKE = "min_uptake"
COLUMN_MAX_UPTAKE = "max_uptake"
COLUMN_OPTIMIZATION_METHOD = "optimization method"
COLUMN_SCORE = "score"
COLUMN_AIC = "AIC"
COLUMN_AICC = "AICc"
COLUMN_BEST_MODEL = "best model"
COLUMN_WORST_MODEL = "worst model"


###############################################################################
DEFAULT_DATASET_COLUMN_MAPPING = {
    "experiment": COLUMN_EXPERIMENT,
    "temperature": COLUMN_TEMPERATURE_K,
    "pressure": COLUMN_PRESSURE_PA,
    "uptake": COLUMN_UPTAKE_MOL_G,
}

DATASET_FALLBACK_DELIMITERS = (";", "\t", "|")
PAD_VALUE = 0.0
SCADS_SERIES_MODEL = "SCADS Series"
SCADS_ATOMIC_MODEL = "SCADS Atomic"

###############################################################################
DATASETS_ROUTER_PREFIX = "/datasets"
FITTING_ROUTER_PREFIX = "/fitting"
FITTING_RUN_ENDPOINT = "/run"
FITTING_JOBS_ENDPOINT = "/jobs"
FITTING_JOB_STATUS_ENDPOINT = "/jobs/{job_id}"
NIST_ROUTER_PREFIX = "/nist"
NIST_FETCH_ENDPOINT = "/fetch"
NIST_PROPERTIES_ENDPOINT = "/properties"
NIST_STATUS_ENDPOINT = "/status"
NIST_CATEGORY_STATUS_ENDPOINT = "/categories/status"
NIST_CATEGORY_PING_ENDPOINT = "/categories/{category}/ping"
NIST_CATEGORY_INDEX_ENDPOINT = "/categories/{category}/index"
NIST_CATEGORY_FETCH_ENDPOINT = "/categories/{category}/fetch"
NIST_CATEGORY_ENRICH_ENDPOINT = "/categories/{category}/enrich"
NIST_JOBS_ENDPOINT = "/jobs"
NIST_JOB_STATUS_ENDPOINT = "/jobs/{job_id}"
ROOT_ENDPOINT = "/"
DOCS_ENDPOINT = "/docs"

