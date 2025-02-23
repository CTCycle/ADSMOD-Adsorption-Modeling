import numpy as np
import pandas as pd
from tqdm import tqdm

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
class DataSanitizer:

    def __init__(self, configuration):

        self.separator = ' - '
        self.P_TARGET_COL = 'pressure'
        self.Q_TARGET_COL = 'adsorbed_amount'
        self.T_TARGET_COL = 'temperature'
        self.max_pressure = configuration['dataset']['MAX_PRESSURE']
        self.max_uptake = configuration['dataset']['MAX_UPTAKE']
        self.configuration = configuration  

        self.included_cols = ['temperature', 'pressure', 'adsorbed_amount', 'encoded_adsorbent',
                              'adsorbate_molecular_weight', 'adsorbate_encoded_SMILE']

    #--------------------------------------------------------------------------
    def is_convertible_to_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    #--------------------------------------------------------------------------
    def exclude_outside_boundary(self, dataset : pd.DataFrame):        
        dataset = dataset[dataset[self.T_TARGET_COL].astype(int) > 0]
        dataset[self.P_TARGET_COL] = dataset[self.P_TARGET_COL].apply(
            lambda x: [float(v) for v in x if 0.0 <= float(v) <= self.max_pressure])
        dataset[self.Q_TARGET_COL] = dataset[self.Q_TARGET_COL].apply(
            lambda x: [float(v) for v in x if 0.0 <= float(v) <= self.max_uptake])
    
        return dataset
    
    #--------------------------------------------------------------------------
    def isolate_preprocessed_features(self, dataset : pd.DataFrame): 
        return dataset[self.included_cols]
    
    #--------------------------------------------------------------------------
    def convert_series_to_string(self, dataset: pd.DataFrame):        
        dataset = dataset.applymap(
            lambda x: self.separator.join(map(str, x)) if isinstance(x, list) else x)
        return dataset

    #--------------------------------------------------------------------------
    def convert_string_to_series(self, dataset: pd.DataFrame):  
        dataset = dataset.applymap(
            lambda x : (
            [np.float32(f) for f in x.split(self.separator) if self.is_convertible_to_float(f)]
            if isinstance(x, str) and self.separator in x else x) if pd.notna(x) else x)
        
        return dataset
                    
      
    
   
    