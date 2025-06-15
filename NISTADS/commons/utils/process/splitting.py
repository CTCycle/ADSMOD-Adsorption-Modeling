import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataset):
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'
        self.adsorbate_col = 'adsorbate_name'
        self.adsorbent_col = 'adsorbent_name'     

        # Set the sizes for the train and validation datasets        
        self.validation_size = configuration["dataset"]["VALIDATION_SIZE"]
        self.seed = configuration["dataset"]["SPLIT_SEED"]
        self.train_size = 1.0 - self.validation_size
        self.dataset = dataset  

        self.splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.validation_size, random_state=self.seed)

    #--------------------------------------------------------------------------
    def remove_underpopulated_classes(self, dataset):        
        dataset['combination'] = (dataset[self.adsorbate_col].astype(str) 
                                  + "_" + dataset[self.adsorbent_col].astype(str))
        combo_counts = dataset['combination'].value_counts()
        valid_combinations = combo_counts[combo_counts >= 2].index    
        dataset = dataset[dataset['combination'].isin(valid_combinations)]              
           
        return dataset
            
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):   
        dataset = self.remove_underpopulated_classes(self.dataset) 
        combination_classes = dataset['combination']      
        # Get the train and validation indices, returns a generator with a single split      
        train_idx, valid_idx = next(self.splitter.split(dataset, combination_classes))        
        # Select rows based on indices and drop the helper column
        train_data = dataset.iloc[train_idx].drop(columns=['combination'])
        validation_data = dataset.iloc[valid_idx].drop(columns=['combination'])       
        
        return train_data, validation_data
    
    
    
   
