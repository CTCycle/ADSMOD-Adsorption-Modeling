import os
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm

from NISTADS.commons.utils.data.loader import InferenceDataLoader

from NISTADS.commons.constants import CONFIG, INFERENCE_PATH
from NISTADS.commons.logger import logger


# [INFERENCE]
###############################################################################
class AdsorptionPredictions:
    
    def __init__(self, model : keras.Model, configuration : dict, metadata : dict, checkpoint_path : str):        
        keras.utils.set_random_seed(configuration["SEED"])  
        # initialize the inference data loader that prepares the data 
        # for using the model in inference mode (predictions)
        self.dataloader = InferenceDataLoader(configuration)                          
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.configuration = configuration
        self.metadata = metadata
        self.model = model     

    #--------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data : pd.DataFrame):       
        processed_inputs = self.dataloader.preprocess_inference_inputs(data)
        predictions = self.model.predict(processed_inputs, verbose=1)
        predictions = self.dataloader.postprocess_inference_output(processed_inputs, predictions)


        return predictions

  
        
      




