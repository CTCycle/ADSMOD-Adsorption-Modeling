import json
from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'NISTADS')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
METADATA_PATH = join(DATA_PATH, 'metadata')
VALIDATION_PATH = join(DATA_PATH, 'validation')
INFERENCE_PATH = join(DATA_PATH, 'inference')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
LOGS_PATH = join(RSC_PATH, 'logs')

###############################################################################
PAD_VALUE = -1

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configuration.json')
with open(self.configuration_PATH, 'r') as file:
    CONFIG = json.load(file)