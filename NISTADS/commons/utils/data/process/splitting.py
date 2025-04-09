import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataframe: pd.DataFrame):
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'
        self.adsorbate_col = 'adsorbate_name'
        self.adsorbent_col = 'adsorbent_name'     

        # Set the sizes for the train and validation datasets        
        self.validation_size = configuration["dataset"]["VALIDATION_SIZE"]
        self.seed = configuration["dataset"]["SPLIT_SEED"]
        self.train_size = 1.0 - self.validation_size
        self.dataframe = dataframe        
        
        total_samples = len(dataframe)
        self.train_size = int(total_samples * self.train_size)
        self.val_size = int(total_samples * self.validation_size)
            
    #--------------------------------------------------------------------------
    def split_train_and_validation(self):          
        train_data, validation_data = train_test_split(
            self.dataframe, test_size=self.validation_size, random_state=self.seed,
            stratify=self.dataframe['combination']) 
        train_data = train_data.drop(columns=['combination'])
        validation_data = validation_data.drop(columns=['combination'])        
        
        return train_data, validation_data
    
    
    
   
