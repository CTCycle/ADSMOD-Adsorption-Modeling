import os
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm

from NISTADS.commons.utils.data.loader import InferenceDataLoader

from NISTADS.commons.constants import CONFIG, INFERENCE_PATH, PAD_VALUE
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

    # wrapper for the inference inputs preprocessing method called by the dataloader
    #--------------------------------------------------------------------------
    def process_inference_inputs(self, data : pd.DataFrame): 
        return self.dataloader.process_inference_inputs(data)

    #--------------------------------------------------------------------------
    def process_inference_output(self, inputs : dict, predictions : np.array):        
        # reshape predictions from (samples, measurements, 1) to (samples, measurements)        
        predictions = np.squeeze(predictions, axis=-1)      
        pressure_inputs = inputs['pressure_input']
        unpadded_predictions = []    
        
        # Reverse each row to get pad values as leading values
        # Create a boolean mask to set pad values to False
        flipped = np.flip(pressure_inputs, axis=1)      
        true_values_mask = flipped != PAD_VALUE
        
        # Find the index of the first true value, or set to the full length if all values are padded  
        trailing_counts = np.where(
            true_values_mask.any(axis=1), true_values_mask.argmax(axis=1), pressure_inputs.shape[1])
        
        # trim the predicted sequences based on true values number
        unpadded_length = pressure_inputs.shape[1] - trailing_counts       
        unpadded_predictions = [
            pred_row[:length] for pred_row, length in zip(predictions, unpadded_length)]
        
        return unpadded_predictions         

    #--------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data : pd.DataFrame):       
        processed_inputs = self.process_inference_inputs(data)
        predictions = self.model.predict(processed_inputs, verbose=1)
        predictions = self.process_inference_output(processed_inputs, predictions)

        return predictions
    
    #--------------------------------------------------------------------------
    def merge_predictions_to_dataset(self, data : pd.DataFrame, predictions : list):
        concat_predictions = np.concatenate(predictions)
        data['predicted_adsorbed_amount'] = concat_predictions

        return data




