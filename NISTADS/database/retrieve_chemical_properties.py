# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.datamaker.properties import MolecularProperties

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':     
    
    # 1. [COLLECT EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info(f'Loading NISTADS datasets from {DATA_PATH}')
    serializer = DataSerializer(CONFIG)
    experiments, guest_data, host_data = serializer.load_datasets()

    sanitizer = DataSanitizer(CONFIG)       
    guest_data = sanitizer.convert_string_to_series(guest_data) 
    host_data = sanitizer.convert_string_to_series(host_data)    
     
    # 3. [PREPARE COLLECTED EXPERIMENTS DATA]
    #-------------------------------------------------------------------------   
    properties = MolecularProperties(CONFIG)  
    # process guest (adsorbed species) data by adding molecular properties
    logger.info('Retrieving molecular properties for sorbate species')
    guest_data = properties.fetch_guest_properties(experiments, guest_data)   
    # process host (adsorbent materials) data by adding molecular properties
    logger.info('Retrieving molecular properties for adsorbent materials') 
    host_data = properties.fetch_host_properties(experiments, host_data)    
  
    # save the final version of the materials dataset    
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')

    
   

    