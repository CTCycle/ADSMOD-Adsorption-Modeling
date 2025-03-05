# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.validation.reports import DataAnalysisPDF
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.validation.dataset import AdsorptionDataValidation
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
    
    # 2. [COMPUTE IMAGE STATISTICS]
    #--------------------------------------------------------------------------
    # validate splitting based on random seed
    # print('\nValidation best random seed for data splitting\n')
    # min_diff, best_seed, best_split = validator.data_split_validation(dataset, cnf.TEST_SIZE, 500)
    # print(f'''\nBest split found with split_seed of {best_seed}, with total difference equal to {round(min_diff, 3)}
    # Mean and standard deviation differences per features (X and Y):''')
    # for key, val in best_split.items():
    #     print(f'{key} ---> mean difference = {val[0]}, STD difference = {val[1]}')

    # 3. [COMPARE TRAIN AND TEST DATASETS]
    #--------------------------------------------------------------------------
    

    # 2. [INITIALIZE PDF REPORT]
    #--------------------------------------------------------------------------
    report = DataAnalysisPDF()
