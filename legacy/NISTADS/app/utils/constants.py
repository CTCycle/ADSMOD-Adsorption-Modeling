from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "NISTADS")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RESOURCES_PATH, "database")
METADATA_PATH = join(DATA_PATH, "metadata")
EVALUATION_PATH = join(DATA_PATH, "validation")
DATA_SOURCE_PATH = join(DATA_PATH, "dataset")
CHECKPOINT_PATH = join(RESOURCES_PATH, "checkpoints")
CONFIG_PATH = join(RESOURCES_PATH, "configurations")
LOGS_PATH = join(RESOURCES_PATH, "logs")

# files
###############################################################################
PROCESS_METADATA_FILE = join(METADATA_PATH, "preprocessing_metadata.json")
SCADS_SERIES_MODEL = "SCADS Adsorption Isotherm"
SCADS_ATOMIC_MODEL = "SCADS Single Measurement"
PAD_VALUE = -1

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, "app", "layout", "main_window.ui")
