# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.datafetch.experiments import AdsorptionDataFetch
from NISTADS.commons.utils.datamaker.datasets import BuildAdsorptionDataset
from NISTADS.commons.utils.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':      

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect adsorption isotherm indexes')
    webworker = AdsorptionDataFetch(CONFIG)
    experiments_index = webworker.get_experiments_index()     

    # 2. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorption isotherms data')
    adsorption_data = webworker.get_experiments_data(experiments_index) 
        
    # 3. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    builder = BuildAdsorptionDataset()
    serializer = DataSerializer(CONFIG)
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
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')    

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
     
    # 7. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    sanitizer = DataSanitizer(CONFIG)      
    guest_data = sanitizer.convert_series_to_string(guest_data) 
    host_data = sanitizer.convert_series_to_string(host_data)  
  
    # save the final version of the materials dataset 
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')
  

    