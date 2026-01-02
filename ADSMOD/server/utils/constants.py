from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "ADSMOD")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RESOURCES_PATH, "database")
LOGS_PATH = join(RESOURCES_PATH, "logs")
TEMPLATES_PATH = join(RESOURCES_PATH, "templates")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
DATABASE_FILENAME = "sqlite.db"


###############################################################################
SERVER_CONFIGURATION_FILE = join(SETTING_PATH, "server_configurations.json")


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
    "experiment": "experiment",
    "temperature": "temperature [K]",
    "pressure": "pressure [Pa]",
    "uptake": "uptake [mol/g]",
}

DATASET_FALLBACK_DELIMITERS = (";", "\t", "|")

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
FITTING_ROUTER_PREFIX = "/fitting"
FITTING_RUN_ENDPOINT = "/run"
NIST_ROUTER_PREFIX = "/nist"
NIST_FETCH_ENDPOINT = "/fetch"
NIST_PROPERTIES_ENDPOINT = "/properties"
NIST_STATUS_ENDPOINT = "/status"
BROWSER_ROUTER_PREFIX = "/browser"
BROWSER_TABLES_ENDPOINT = "/tables"
BROWSER_DATA_ENDPOINT = "/data"
ROOT_ENDPOINT = "/"
DOCS_ENDPOINT = "/docs"

# Table name to friendly display name mapping for database browser
BROWSER_TABLE_DISPLAY_NAMES: dict[str, str] = {
    "SINGLE_COMPONENT_ADSORPTION": "Single-Component Adsorption Data",
    "BINARY_MIXTURE_ADSORPTION": "Binary Mixture Adsorption Data",
    "ADSORBATES": "Adsorbate Materials",
    "ADSORBENTS": "Adsorbent Materials",
    "ADSORPTION_DATA": "Uploaded Adsorption Data",
    "ADSORPTION_LANGMUIR": "Langmuir Model Results",
    "ADSORPTION_SIPS": "Sips Model Results",
    "ADSORPTION_FREUNDLICH": "Freundlich Model Results",
    "ADSORPTION_TEMKIN": "Temkin Model Results",
    "ADSORPTION_TOTH": "Toth Model Results",
    "ADSORPTION_DUBININ_RADUSHKEVICH": "Dubinin-Radushkevich Model Results",
    "ADSORPTION_DUAL_SITE_LANGMUIR": "Dual-Site Langmuir Model Results",
    "ADSORPTION_REDLICH_PETERSON": "Redlich-Peterson Model Results",
    "ADSORPTION_JOVANOVIC": "Jovanovic Model Results",
    "ADSORPTION_BEST_FIT": "Best Fit Results",
}
