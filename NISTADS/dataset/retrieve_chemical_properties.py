# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
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
     
    # 2. [FETCH MOLECULAR PROPERTIES]
    #--------------------------------------------------------------------------   
    properties = MolecularProperties(CONFIG)  
    # process guest (adsorbed species) data by adding molecular properties
    logger.info('Retrieving molecular properties for sorbate species using PubChem API')
    guest_data = properties.fetch_guest_properties(experiments, guest_data)   
    # process host (adsorbent materials) data by adding molecular properties   
    logger.info('Retrieving molecular properties for adsorbent materials using PubChem API') 
    host_data = properties.fetch_host_properties(experiments, host_data)   
  
    # 3. [SAVE MATERIALS DATASET]
    #--------------------------------------------------------------------------
    # save the final version of the materials dataset    
    serializer.save_materials_datasets(guest_data, host_data)
    logger.info('Data collection is concluded')

    
   

    