import numpy as np
import tensorflow as tf
import keras

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator():

    def __init__(self, configuration):             
        self.normalization = configuration["dataset"]["NORMALIZE"]
        self.configuration = configuration    

    #--------------------------------------------------------------------------
    def separate_features_and_output(self, data): 

        parameters = data[['temperature', 'adsorbate_molecular_weight', 'adsorbent_molecular_weight']]
        smiles = data['adsorbate_SMILE', 'adsorbent_SMILE'].to_list()
        P_series = data['pressure'].to_list()        
        Q_series = data['adsorbed_amount'].to_list()     

    #--------------------------------------------------------------------------
    def process_data(self, path, text):        
        input_text = text[:-1]
        output_text = text[1:]      

        return input_text, output_text  


                     
    

    




            
