import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


###############################################################################
class FeatureNormalizer:

    def __init__(self, configuration):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.norm_columns = ['temperature', 'molecular_weight', 
                             'heavy_atoms', 'H_acceptors','H_donors'] 
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def normalize_molecular_features(self, dataset : pd.DataFrame, train_dataset: pd.DataFrame):        
        # Fit the scaler on the training data, then normalize entire dataset 
        self.scaler.fit(train_dataset[self.norm_columns])       
        dataset[self.norm_columns] = self.scaler.transform(dataset[self.norm_columns]) 
       
        with open('normalizer.pkl', 'wb') as file:
            pickle.dump(self.scaler, file)

        return dataset

    
    