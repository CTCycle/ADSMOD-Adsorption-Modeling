import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Model

from NISTADS.app.utils.data.loader import SCADSDataLoader
from NISTADS.app.interface.workers import check_thread_status, update_progress_callback
from NISTADS.app.constants import EVALUATION_PATH, PAD_VALUE
from NISTADS.app.logger import logger

        

# [LOAD MODEL]
################################################################################
class AdsorptionPredictionsQuality:

    def __init__(self, model : Model, configuration : dict, metadata : dict, checkpoint_path : str, num_experiments=6): 
        self.save_images = configuration.get('save_images', True)  
        self.model = model            
        self.configuration = configuration 
        self.metadata = metadata      
        self.dataloader = SCADSDataLoader(configuration, metadata)       
        self.num_experiments = num_experiments
        self.cols = int(np.ceil(np.sqrt(self.num_experiments)))      
        self.rows = int(np.ceil(self.num_experiments/self.cols)) 
        self.DPI = configuration.get('image_resolution', 400)
        self.file_type = 'jpeg' 

        self.checkpoint = os.path.basename(checkpoint_path)        
        self.validation_path = os.path.join(
            EVALUATION_PATH, 'validation', self.checkpoint) 
        os.makedirs(self.validation_path, exist_ok=True)

    #--------------------------------------------------------------------------
    def save_image(self, fig, name):
        name = re.sub(r'[^0-9A-Za-z_]', '_', name)
        out_path = os.path.join(self.validation_path, name)
        fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)         

    #--------------------------------------------------------------------------
    def process_uptake_curves(self, inputs, output, predicted_output):
        pressures, uptakes, predictions = [], [], []
        for exp in range(self.num_experiments):
            pressure = inputs['pressure_input'][exp, :]
            true_y = output[exp, :]
            predicted_y = np.squeeze(predicted_output[exp, :]) 
            # calculate unpadded length to properly truncate series
            valid_pressure = pressure[pressure != PAD_VALUE]            
            pressure = pressure[:len(valid_pressure)]
            true_y = true_y[:len(valid_pressure)]
            predicted_y = predicted_y[:len(valid_pressure)]

            pressures.append(pressure * self.metadata['normalization']['pressure'])
            uptakes.append(true_y * self.metadata['normalization']['adsorbed_amount'])
            predictions.append(predicted_y * self.metadata['normalization']['adsorbed_amount'])

        return pressures, uptakes, predictions        

    #--------------------------------------------------------------------------
    def visualize_adsorption_isotherms(self, validation_data : pd.DataFrame, **kwargs):              
        sampled_data = validation_data.sample(n=self.num_experiments, random_state=42)
        sampled_X, sampled_Y = self.dataloader.separate_inputs_and_output(sampled_data)
        predictions = self.model.predict(sampled_X)      
        # process training uptake curves
        check_thread_status(kwargs.get('worker', None))
        pressures, uptakes, predictions = self.process_uptake_curves(
            sampled_X, sampled_Y, predictions)

        # Create the subplots (flatten axes to simplify iteration later)
        fig, axes = plt.subplots(
            self.rows, self.cols, figsize=(5 * self.cols, 4 * self.rows))
        axes = np.array(axes).flatten()        

        for i in range(self.num_experiments):
            axes[i].plot(pressures[i], uptakes[i], label='Adsorbed amount')
            axes[i].plot(pressures[i], predictions[i], label='Predicted adsorption')
            axes[i].set_title(f'Plot {i + 1}') 

            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i+1, self.num_experiments, kwargs.get('progress_callback', None))       

        plt.title('Comparison of validation adsorption isotherms', fontsize=16)
        plt.tight_layout()
        self.save_image(fig, 'validation_curves_comparison.jpeg')
        plt.close() 

        return fig 