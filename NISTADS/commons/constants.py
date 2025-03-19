import json
from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'NISTADS')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'dataset')
PROCESSED_PATH = join(DATA_PATH, 'processed_data')
NLP_PATH = join(RSC_PATH, 'NLP models')
VALIDATION_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
PREDS_PATH = join(RSC_PATH, 'predictions')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')


###############################################################################
PAD_VALUE = -10

# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)