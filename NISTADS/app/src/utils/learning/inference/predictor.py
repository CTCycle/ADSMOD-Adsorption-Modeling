import os
import numpy as np
from keras.utils import set_random_seed

from NISTADS.app.src.utils.learning.callbacks import LearningInterruptCallback
from NISTADS.app.src.interface.workers import check_thread_status, update_progress_callback
from NISTADS.app.src.utils.data.loader import InferenceDataLoader

from NISTADS.app.src.constants import INFERENCE_PATH, PAD_VALUE
from NISTADS.app.src.logger import logger


# [INFERENCE]
###############################################################################
class AdsorptionPredictions:
    
    def __init__(self, model, configuration : dict, checkpoint_path : str):        
        set_random_seed(configuration.get('train_seed', 42)) 
        # initialize the inference data loader that prepares the data 
        # for using the model in inference mode (predictions)
        self.dataloader = InferenceDataLoader(configuration)                          
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.configuration = configuration        
        self.model = model

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
            true_values_mask.any(axis=1), true_values_mask.argmax(axis=1), 
            pressure_inputs.shape[1])
        
        # trim the predicted sequences based on true values number
        unpadded_length = pressure_inputs.shape[1] - trailing_counts       
        unpadded_predictions = [
            pred_row[:length] for pred_row, length in zip(predictions, unpadded_length)]
        
        return unpadded_predictions         

    #--------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data, **kwargs):
        # add interruption callback to stop model predictions if requested
        callbacks_list = [LearningInterruptCallback(kwargs.get('worker', None))]     
        # preprocess inputs before feeding them to the pretrained model for inference
        # add padding, normalize data, encode categoricals
        processed_inputs = self.dataloader.process_inference_inputs(data)
        # perform prediction of adsorption isotherm sequences
        predictions = self.model.predict(processed_inputs, verbose=1, callbacks=callbacks_list) 
        # postprocess obtained outputs 
        # remove padding, rescale, decode categoricals
        predictions = self.process_inference_output(processed_inputs, predictions)

        return predictions
    
    #--------------------------------------------------------------------------
    def merge_predictions_to_dataset(self, data, predictions : list):
        concat_predictions = np.concatenate(predictions)
        data['predicted_adsorbed_amount'] = concat_predictions

        return data




