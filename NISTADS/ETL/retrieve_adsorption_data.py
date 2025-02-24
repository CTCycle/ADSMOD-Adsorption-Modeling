# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.datafetch.experiments import AdsorptionDataFetch
from NISTADS.commons.utils.datamaker.datasets import BuildAdsorptionDataset
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':      

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect adsorption isotherm indices from NIST-ARPA-E database')
    webworker = AdsorptionDataFetch(CONFIG)
    experiments_index = webworker.get_experiments_index()     

    # 2. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorption isotherms data from JSON response')
    adsorption_data = webworker.get_experiments_data(experiments_index) 
        
    # 3. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    builder = BuildAdsorptionDataset()
    serializer = DataSerializer(CONFIG)
    logger.info('Cleaning and processing adsorption dataset, experiments will be split based on mixture complexity')
    # remove excluded columns from the dataframe
    adsorption_data = builder.drop_excluded_columns(adsorption_data)
    # split current dataframe by complexity of the mixture (single component or binary mixture)
    single_component, binary_mixture = builder.split_by_mixture_complexity(adsorption_data) 
    # extract nested data in dataframe rows and reorganise them into columns
    single_component = builder.extract_nested_data(single_component)
    binary_mixture = builder.extract_nested_data(binary_mixture)     

    # 4. [SAVE DATASET]
    # finally expand the dataset to represent each measurement with a single row
    # save the final version of the adsorption dataset
    #--------------------------------------------------------------------------    
    single_component, binary_mixture = builder.expand_dataset(single_component, binary_mixture)
    serializer.save_adsorption_datasets(single_component, binary_mixture)     
    logger.info('Experiments data collection is concluded')
    logger.info(f'Generated files have been saved in {DATA_PATH}')    

    # 5. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get guest and host indexes invoking API
    logger.info('Collect guest and host indices from NIST-ARPA-E database')
    webworker = GuestHostDataFetch(CONFIG)
    guest_index, host_index = webworker.get_guest_host_index()     

    # 6. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data from relative indices')
    guest_data, host_data = webworker.get_guest_host_data(guest_index, host_index)   
     
    # 7. [SAVE EXPERIMENTS DATA]
    #--------------------------------------------------------------------------  
    # save the final version of the materials dataset 
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info('Materials data collection is concluded')
    logger.info(f'Generated files have been saved in {DATA_PATH}')   
  

    