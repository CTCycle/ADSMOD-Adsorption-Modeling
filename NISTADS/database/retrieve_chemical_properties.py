# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.datamaker.datasets import BuildMaterialsDataset
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':     
    
    # 1. [COLLECT EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info(f'Loading NISTADS datasets from {DATA_PATH}')
    dataserializer = DataSerializer(CONFIG)
    adsorption_data, guest_data, host_data = dataserializer.load_datasets()    
     
    # 3. [PREPARE COLLECTED EXPERIMENTS DATA]
    #-------------------------------------------------------------------------   
    builder = BuildMaterialsDataset(CONFIG)      
    # process guest (adsorbed species) data by adding molecular properties
    logger.info('Retrieving molecular properties for sorbate species')
    guest_data = builder.add_guest_properties(guest_data)   
    # process host (adsorbent materials) data by adding molecular properties
    logger.info('Retrieving molecular properties for adsorbent materials') 
    host_data = builder.add_host_properties(host_data)    
  
    # save the final version of the materials dataset
    serializer = DataSerializer(CONFIG)
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')

    
   

    