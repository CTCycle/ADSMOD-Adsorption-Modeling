import json
from os.path import join, dirname, abspath 

# [PATHS]
###############################################################################
PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'dataset')
PROCESSED_PATH = join(DATA_PATH, 'processed_data')
NLP_PATH = join(RSC_PATH, 'NLP models')
VALIDATION_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
PREDS_PATH = join(RSC_PATH, 'predictions')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')

# [FILENAMES]
###############################################################################


# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)