# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.validation.dataset import AdsorptionDataValidation
from NISTADS.commons.utils.data.serializer import DataSerializer
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    dataserializer = DataSerializer(CONFIG)
    adsorption_data, guest_data, host_data = dataserializer.load_datasets() 
    logger.info(f'{adsorption_data.shape[0]} measurements in the dataset')
    logger.info(f'{guest_data.shape[0]} adsorbates species in the dataset')
    logger.info(f'{host_data.shape[0]} adsorbent materials in the dataset')

    # load data from csv, add paths to images 
    dataserializer = DataSerializer(CONFIG)
    processed_data, metadata, smile_vocabulary, ads_vocabulary = dataserializer.load_preprocessed_data()   
    
