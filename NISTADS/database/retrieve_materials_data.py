# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import save_materials_datasets
from NISTADS.commons.utils.datamaker.datasets import BuildMaterialsDataset
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':    
   
    # 1. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get guest and host indexes invoking API
    logger.info('Collect guest and host indexes from NIST DB')
    webworker = GuestHostDataFetch(CONFIG)
    guest_index, host_index = webworker.get_guest_host_index()     

    # 2. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data from relative indexes')
    df_guest, df_host = webworker.get_guest_host_data(guest_index, host_index)
     
    # 6. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    builder = BuildMaterialsDataset()

    # remove excluded columns from the dataframe    
    df_guest = builder.drop_excluded_columns(df_guest)
    # process guest (adsorbed species) data by adding molecular properties
    guest_data = builder.add_guest_properties(df_guest)   
    # process host (adsorbent materials) data by adding molecular properties 
    host_data = builder.add_host_properties(df_host)    
  
    # save the final version of the materials dataset
    save_materials_datasets(guest_data, host_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')

    
   

    