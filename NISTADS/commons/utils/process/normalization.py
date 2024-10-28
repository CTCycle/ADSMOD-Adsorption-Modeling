import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


###############################################################################
class DataNormalization:

    def __init__(self):
        pass

    #--------------------------------------------------------------------------
    def normalize_parameters(self, train_X, train_Y, test_X, test_Y):
       
        # cast float type for both the labels and the continuous features columns 
        norm_columns = ['temperature', 'mol_weight', 'complexity', 'heavy_atoms']       
        train_X[norm_columns] = train_X[norm_columns].astype(float)        
        test_X[norm_columns] = test_X[norm_columns].astype(float)
        
        # normalize the numerical features (temperature and physicochemical properties)      
        self.param_normalizer = MinMaxScaler(feature_range=(0, 1))
        train_X[norm_columns] = self.param_normalizer.fit_transform(train_X[norm_columns])
        test_X[norm_columns] = self.param_normalizer.transform(test_X[norm_columns])        

        return train_X, train_Y, test_X, test_Y     
    
    # normalize sequences using a RobustScaler: X = X - median(X)/IQR(X)
    # flatten and reshape array to make it compatible with the scaler
    #--------------------------------------------------------------------------  
    def normalize_sequences(self, train, test, column):        
        
        normalizer = MinMaxScaler(feature_range=(0,1))
        sequence_array = np.array([item for sublist in train[column] for item in sublist]).reshape(-1, 1)         
        normalizer.fit(sequence_array)
        train[column] = train[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())
        test[column] = test[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())

        return train, test, normalizer
