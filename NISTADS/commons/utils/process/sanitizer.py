import pandas as pd
from tqdm import tqdm
tqdm.pandas()      

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
class AdsorptionDataSanitizer:

    def __init__(self, configuration):

        self.P_TARGET_COL = 'pressure'
        self.Q_TARGET_COL = 'adsorbed_amount'
        self.T_TARGET_COL = 'temperature'
        self.max_pressure = configuration['dataset']['MAX_PRESSURE']
        self.max_uptake = configuration['dataset']['MAX_UPTAKE']
        self.configuration = configuration  
    
    #--------------------------------------------------------------------------
    def exclude_outside_boundary(self, dataset : pd.DataFrame):        
        dataset = dataset[dataset[self.T_TARGET_COL].astype(int) > 0]
        dataset = dataset[dataset[self.P_TARGET_COL].astype(float).between(0.0, self.max_pressure)]
        dataset = dataset[dataset[self.Q_TARGET_COL].astype(float).between(0.0, self.max_uptake)]
        
        return dataset
    
    #--------------------------------------------------------------------------
    def reduce_dataset_features(self, dataset : pd.DataFrame):    
        pass
    
   
    