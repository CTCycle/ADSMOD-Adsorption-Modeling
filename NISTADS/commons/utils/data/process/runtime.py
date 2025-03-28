import pandas as pd
import numpy as np
import tensorflow as tf

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataLoaderProcessor():

    def __init__(self, configuration):        
        self.configuration = configuration   
 
    # currently used as placeholder returning same input and output, additional
    # features may be implemented for image augmentation etc
    #--------------------------------------------------------------------------
    def process_data(self, inputs, output):       
        
        return inputs, output    

    








   


    