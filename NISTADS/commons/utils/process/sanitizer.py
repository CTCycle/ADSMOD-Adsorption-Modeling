import numpy as np
import pandas as pd

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
class DataSanitizer:

    def __init__(self, configuration):

        self.separator = ' AND '
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
    def filter_elements_outside_boundaries(self, row):        
        p_list = row[self.P_TARGET_COL]
        q_list = row[self.Q_TARGET_COL]      
                
        filtered_p = []
        filtered_q = []
        final_p = []
        final_q = []

        for p, q in zip(p_list, q_list):
            if 0.0 <= p <= self.max_pressure:
                filtered_p.append(p)
                filtered_q.append(q)        
        
        for p, q in zip(filtered_p, filtered_q):
            if 0.0 <= q <= self.max_uptake:
                final_p.append(p)
                final_q.append(q)
        
        return pd.Series({self.P_TARGET_COL: final_p,
                          self.Q_TARGET_COL: final_q})
    
    #--------------------------------------------------------------------------
    def exclude_OOB_values(self, dataset : pd.DataFrame):        
        dataset = dataset[dataset[self.T_TARGET_COL].astype(int) > 0]
        filtered_series = dataset.apply(
            self.filter_elements_outside_boundaries, axis=1)
        dataset[self.P_TARGET_COL] = filtered_series[self.P_TARGET_COL]
        dataset[self.Q_TARGET_COL] = filtered_series[self.Q_TARGET_COL]
           
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
            [float(f) for f in x.split(self.separator) if self.is_convertible_to_float(f)]
            if isinstance(x, str) and self.separator in x else x) if pd.notna(x) else x)
        
        return dataset
                    
      
    
   
    