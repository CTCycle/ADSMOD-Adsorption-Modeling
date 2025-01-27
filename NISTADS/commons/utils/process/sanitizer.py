import pandas as pd
from tqdm import tqdm
tqdm.pandas()      

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

        self.drop_cols = ['adsorbate_SMILE', 'adsorbent_SMILE', 
                          'adsorbate_tokenized_SMILE', 'adsorbent_tokenized_SMILE']
    
    #--------------------------------------------------------------------------
    def exclude_outside_boundary(self, dataset : pd.DataFrame):        
        dataset = dataset[dataset[self.T_TARGET_COL].astype(int) > 0]
        dataset = dataset[dataset[self.P_TARGET_COL].astype(float).between(0.0, self.max_pressure)]
        dataset = dataset[dataset[self.Q_TARGET_COL].astype(float).between(0.0, self.max_uptake)]
        
        return dataset
    
    #--------------------------------------------------------------------------
    def reduce_dataset_features(self, dataset : pd.DataFrame): 
        dataset.drop(self.drop_cols, axis=1, inplace=True)   

    #--------------------------------------------------------------------------
    def convert_series_to_string(self, dataset: pd.DataFrame):        
        dataset = dataset.applymap(lambda x: self.separator.join(map(str, x)) if isinstance(x, list) else x)
        return dataset

    #--------------------------------------------------------------------------
    def convert_string_to_series(self, dataset: pd.DataFrame):  
        dataset = dataset.applymap(
            lambda x: x.split() if isinstance(x, str) and self.separator in x else x)
        return dataset
                    
      
    
   
    