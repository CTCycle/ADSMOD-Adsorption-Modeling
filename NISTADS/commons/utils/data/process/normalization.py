import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
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
        dataset['encoded_adsorbent'] = dataset[self.norm_columns].map(
                mapping).fillna(self.unknown_class_index).astype(int)           

        return dataset, mapping

    #--------------------------------------------------------------------------
    def encode_adsorbents_from_vocabulary(self, dataset : pd.DataFrame, vocabulary: dict):              
        mapping = {label: idx for idx, label in vocabulary.items()}           
        dataset['encoded_adsorbent'] = dataset[self.norm_columns].map(
            vocabulary).fillna(self.unknown_class_index).astype(int)           

        return dataset, mapping
    

###############################################################################
class FeatureNormalizer:

    def __init__(self, configuration, train_dataset: pd.DataFrame, statistics=None): 
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'       
        self.norm_columns = ['temperature', 'adsorbate_molecular_weight']       
        self.configuration = configuration 

        self.statistics = self.get_normalization_parameters(
            train_dataset) if statistics is None and train_dataset is None else statistics    

    #--------------------------------------------------------------------------
    def get_normalization_parameters(self, train_data : pd.DataFrame):
        statistics = {}
        for col in self.norm_columns:
            statistics[col] = train_data[col].astype(float).max()

        # concatenate all values together to obtain a flattened array     
        p_values = np.concatenate(train_data[self.P_COL].to_numpy())
        q_values = np.concatenate(train_data[self.Q_COL].to_numpy())
        # calculate mean and srandard deviation for pressure and uptake values
        statistics[self.P_COL] = p_values.max()  
        statistics[self.Q_COL] = q_values.max()       
        
        return statistics

    #--------------------------------------------------------------------------
    def normalize_molecular_features(self, dataset : pd.DataFrame):        
        norm_cols_stats = {
            k : v for k, v in self.statistics.items() if k in self.norm_columns}
        for k, v in norm_cols_stats.items():
            dataset[k] = dataset[k].astype(float)/v

        return dataset
    
    #--------------------------------------------------------------------------  
    def PQ_series_normalization(self, dataset : pd.DataFrame):
        P_max = self.statistics[self.P_COL]
        Q_max = self.statistics[self.Q_COL]        
        dataset[self.P_COL] = dataset[self.P_COL].apply(
            lambda x : [(v/P_max) for v in x])                    
        dataset[self.Q_COL] = dataset[self.Q_COL].apply(
            lambda x : [(v/Q_max) for v in x])

        return dataset

    
    