# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':     
    
    # 1. [COLLECT EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    dataserializer = DataSerializer(CONFIG)
    experiments, _, _ = dataserializer.load_datasets(get_materials=False) 

    # 1. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get guest and host indexes invoking API
    logger.info('Collect guest and host indexes from NIST DB')
    webworker = GuestHostDataFetch(CONFIG)
    guest_index, host_index = webworker.get_guest_host_index()     

    # 2. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data from relative indexes')
    guest_data, host_data = webworker.get_guest_host_data(guest_index, host_index)   
     
    # 3. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    sanitizer = DataSanitizer(CONFIG)      
    guest_data = sanitizer.convert_series_to_string(guest_data) 
    host_data = sanitizer.convert_series_to_string(host_data)  
  
    # save the final version of the materials dataset
    serializer = DataSerializer(CONFIG)
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')

    
   

    