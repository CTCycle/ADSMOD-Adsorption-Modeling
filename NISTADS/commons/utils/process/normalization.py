import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
      
from NISTADS.commons.constants import CONFIG, DATA_PATH, PROCESSED_PATH
from NISTADS.commons.logger import logger





###############################################################################
class AdsorbentEncoder:

    def __init__(self, configuration):
        self.scaler = LabelEncoder()
        self.unknown_class_index = -1
        self.norm_columns = 'adsorbent_name' 
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def encode_adsorbents_by_name(self, dataset : pd.DataFrame, train_dataset: pd.DataFrame):        
        self.scaler.fit(train_dataset[self.norm_columns])         
        mapping = {label: idx for idx, label in enumerate(self.scaler.classes_)}           
        dataset[self.norm_columns] = dataset[self.norm_columns].map(
                mapping).fillna(self.unknown_class_index).astype(int)           

        return dataset, mapping


###############################################################################
class FeatureNormalizer:

    def __init__(self, configuration):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.norm_columns = ['temperature', 'adsorbate_molecular_weight'] 
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def normalize_molecular_features(self, dataset : pd.DataFrame, train_dataset: pd.DataFrame):        
        # Fit the scaler on the training data, then normalize entire dataset 
        self.scaler.fit(train_dataset[self.norm_columns])       
        dataset[self.norm_columns] = self.scaler.transform(dataset[self.norm_columns]) 
        normalizer_path = os.path.join(PROCESSED_PATH, 'normalizer.pkl')
        with open(normalizer_path, 'wb') as file:
            pickle.dump(self.scaler, file)

        return dataset

    
    