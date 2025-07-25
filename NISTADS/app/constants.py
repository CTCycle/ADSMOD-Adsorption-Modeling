import json
from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'NISTADS')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
METADATA_PATH = join(DATA_PATH, 'metadata')
EVALUATION_PATH = join(DATA_PATH, 'validation')
INFERENCE_PATH = join(DATA_PATH, 'inference')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
LOGS_PATH = join(RSC_PATH, 'logs')


PROCESS_METADATA_FILE = join(METADATA_PATH, 'preprocessing_metadata.json')
SMILE_VOCABULARY_FILE = join(METADATA_PATH, 'SMILE_tokenization_index.json')
ADS_VOCABULARY_FILE = join(METADATA_PATH, 'adsorbents_index.json')
        

###############################################################################
PAD_VALUE = -1

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, 'app', 'assets', 'window_layout.ui')
QSS_PATH = join(PROJECT_DIR, 'app', 'assets', 'stylesheet.qss')


