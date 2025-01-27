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
    #-------------------------------------------------------------------------   
    builder = BuildMaterialsDataset(CONFIG)      
    guest_data, host_data = builder.retrieve_materials_from_experiments(experiments, guest_data, host_data) 
    
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

    
   

    