from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "ADSMOD")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
LOGS_PATH = join(RESOURCES_PATH, "logs")
TEMPLATES_PATH = join(RESOURCES_PATH, "templates")
CHECKPOINTS_PATH = join(RESOURCES_PATH, "checkpoints")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
DATABASE_FILENAME = "database.db"


###############################################################################
CONFIGURATION_FILE = join(SETTING_PATH, "configurations.json")


###############################################################################
FASTAPI_TITLE = "ADSMOD Model Fitting Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_VERSION = "1.2.0"


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
MODELS_LIST = [
    "Langmuir",
    "Sips",
    "Freundlich",
    "Temkin",
    "Toth",
    "Dubinin-Radushkevich",
    "Dual-Site Langmuir",
    "Redlich-Peterson",
    "Jovanovic",
]

MODEL_PARAMETER_DEFAULTS: dict[str, dict[str, tuple[float, float]]] = {
    "Langmuir": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
    },
    "Sips": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
        "exponent": (0.1, 10.0),
    },
    "Freundlich": {
        "k": (1e-06, 10.0),
        "exponent": (0.1, 10.0),
    },
    "Temkin": {
        "k": (1e-06, 10.0),
        "beta": (0.1, 10.0),
    },
    "Toth": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
        "exponent": (0.1, 10.0),
    },
    "Dubinin-Radushkevich": {
        "qsat": (0.0, 100.0),
        "beta": (1e-06, 10.0),
    },
    "Dual-Site Langmuir": {
        "k1": (1e-06, 10.0),
        "qsat1": (0.0, 100.0),
        "k2": (1e-06, 10.0),
        "qsat2": (0.0, 100.0),
    },
    "Redlich-Peterson": {
        "k": (1e-06, 10.0),
        "a": (1e-06, 10.0),
        "beta": (0.1, 1.0),
    },
    "Jovanovic": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
    },
}

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

FITTING_MODEL_NAMES = (
    "LANGMUIR",
    "SIPS",
    "FREUNDLICH",
    "TEMKIN",
    "TOTH",
    "DUBININ_RADUSHKEVICH",
    "DUAL_SITE_LANGMUIR",
    "REDLICH_PETERSON",
    "JOVANOVIC",
)


###############################################################################
DATASETS_ROUTER_PREFIX = "/datasets"
DATASETS_LOAD_ENDPOINT = "/load"
DATASETS_NAMES_ENDPOINT = "/names"
DATASETS_FETCH_ENDPOINT = "/by-name/{dataset_name}"
FITTING_ROUTER_PREFIX = "/fitting"
FITTING_RUN_ENDPOINT = "/run"
FITTING_NIST_DATASET_ENDPOINT = "/nist-dataset"
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
