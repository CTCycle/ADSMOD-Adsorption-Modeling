import os
import numpy as np
import keras
from tqdm import tqdm

from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.constants import *
from NISTADS.commons.logger import logger


# [INFERENCE]
###############################################################################
class AdsorptionForecaster:
    
    def __init__(self, model : keras.Model, configuration):
       
        keras.utils.set_random_seed(configuration["SEED"])  
        





