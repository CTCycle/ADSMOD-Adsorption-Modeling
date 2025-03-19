import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class TrainValidationSplit:

    def __init__(self, configuration, dataframe: pd.DataFrame):
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'     

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
        self.dataframe = shuffle(
            self.dataframe, random_state=self.seed).reset_index(drop=True) 
        train_data = self.dataframe.iloc[:self.train_size]
        validation_data = self.dataframe.iloc[self.train_size:self.train_size + self.val_size]
        
        return train_data, validation_data
    
    #--------------------------------------------------------------------------
    def get_normalization_parameters(self, train_data : pd.DataFrame):
        # concatenate all values together to obtain a flattened array     
        p_values = np.concatenate(train_data[self.P_COL].to_numpy())
        q_values = np.concatenate(train_data[self.Q_COL].to_numpy())
        # calculate mean and srandard deviation for pressure and uptake values        
        mean_p, std_p, max_p = p_values.mean(), p_values.std(), p_values.max()
        mean_q, std_q, max_q = q_values.mean(), q_values.std(), q_values.max()

        statistics = {self.P_COL: {
            "mean": round(mean_p, 3), "std": round(std_p, 3), "max": round(max_p, 3)},
                self.Q_COL: {
            "mean": round(mean_q, 3), "std": round(std_q, 3), "max": round(max_q, 3)}}

        return statistics
    
   
