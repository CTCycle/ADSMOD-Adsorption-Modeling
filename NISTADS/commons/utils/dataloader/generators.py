import numpy as np
import tensorflow as tf
import keras
import torch

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator():

    def __init__(self, configuration):       
        self.configuration = configuration       

    #--------------------------------------------------------------------------
    def process_data(self, inputs, output):
        parameters, adsorbent, adsorbate, pressure = inputs
        # parameters = keras.ops.cast(parameters, dtype=torch.float32)        
        # adsorbent = keras.ops.cast(adsorbent, dtype=torch.float32)
        # adsorbate = keras.ops.cast(adsorbate, dtype=torch.float32)
        # pressure = keras.ops.cast(pressure, dtype=torch.float32)       
        # output = keras.ops.cast(output, dtype=torch.float32)

        return (parameters, adsorbent, adsorbate,pressure, ), output
                     
    

    




            
