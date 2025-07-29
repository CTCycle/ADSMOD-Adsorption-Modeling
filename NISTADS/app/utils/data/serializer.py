import os
import json

import pandas as pd
from keras.utils import plot_model
from keras.models import load_model
from datetime import datetime

from NISTADS.app.utils.data.database import AdsorptionDatabase
from NISTADS.app.utils.process.sanitizer import DataSanitizer
from NISTADS.app.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared
from NISTADS.app.utils.learning.training.scheduler import LinearDecayLRScheduler
from NISTADS.app.constants import PROCESS_METADATA_FILE, CHECKPOINT_PATH

from NISTADS.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration : dict):
        self.seed = configuration.get('seed', 42)
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'        
        self.series_cols = [self.P_COL, self.Q_COL, 'adsorbate_encoded_SMILE']
        self.configuration = configuration
        self.database = AdsorptionDatabase()
        
    #--------------------------------------------------------------------------
    def validate_metadata(self, metadata : dict, target_metadata : dict):        
        keys_to_compare = [k for k in metadata if k != "date"]
        meta_current = {k: metadata.get(k) for k in keys_to_compare}
        meta_target = {k: target_metadata.get(k) for k in keys_to_compare}        
        differences = {k: (meta_current[k], meta_target[k]) 
                       for k in keys_to_compare if meta_current[k] != meta_target[k]} 
        
        return False if differences else True
        
    #--------------------------------------------------------------------------
    def serialize_series(self, data : pd.DataFrame, columns):
        for col in columns:
            data[col] = data[col].apply(
                lambda x: ' '.join(map(str, x)) if isinstance(x, list)
                else [float(i) for i in x.split()] if isinstance(x, str)
                else x)
            
        return data
            
    #--------------------------------------------------------------------------
    def load_adsorption_datasets(self):          
        adsorption_data, guest_data, host_data = self.database.load_source_dataset()

        return adsorption_data, guest_data, host_data
    
    #--------------------------------------------------------------------------
    def load_inference_data(self):              
        return self.database.load_inference_data()   
    
    #--------------------------------------------------------------------------
    def load_train_and_validation_data(self, only_metadata=False): 
        with open(PROCESS_METADATA_FILE, 'r') as file:
            metadata = json.load(file)  

        if not only_metadata:
            # load preprocessed data from database and convert joint strings to list
            train_data, val_data = self.database.load_train_and_validation()
            train_data = self.serialize_series(train_data, self.series_cols) 
            val_data = self.serialize_series(val_data, self.series_cols) 

            return train_data, val_data, metadata   
        
        return metadata

    #--------------------------------------------------------------------------
    def save_train_and_validation_data(self, train_data, val_data, smile_vocabulary, 
                                       ads_vocabulary, normalization_stats={}):      

        # convert list to joint string and save preprocessed data to database
        train_data = self.serialize_series(train_data, self.series_cols) 
        val_data = self.serialize_series(val_data, self.series_cols)    
        self.database.save_train_and_validation(train_data, val_data)             
         
        metadata = {'seed' : self.seed, 
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'sample_size' : self.configuration.get('sample_size', 1.0),
                    'validation_size' : self.configuration.get('validation_size', 0.2),
                    'split_seed' : self.configuration.get('split_seed', 42),
                    'max_measurements' : self.configuration.get('max_measurements', 1000),
                    'SMILE_sequence_size' : self.configuration.get('SMILE_sequence_size', 30),
                    'SMILE_vocabulary_size' : len(smile_vocabulary),
                    'adsorbent_vocabulary_size' : len(ads_vocabulary), 
                    'normalization' : {
                        self.P_COL : float(normalization_stats[self.P_COL]),
                        self.Q_COL : float(normalization_stats[self.Q_COL]),
                        'temperature' : float(normalization_stats['temperature']),
                        'adsorbate_molecular_weight' : float(normalization_stats['adsorbate_molecular_weight'])},
                    'SMILE_vocabulary' : smile_vocabulary,
                    'adsorbent_vocabulary' : ads_vocabulary}  
               
        with open(PROCESS_METADATA_FILE, 'w') as file:
            json.dump(metadata, file, indent=4) 
   
    #--------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data=None, host_data=None):
        self.database.save_materials_datasets(guest_data, host_data)

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component, binary_mixture):
        self.database.save_adsorption_dataset(single_component, binary_mixture)  

    #--------------------------------------------------------------------------
    def save_predictions_dataset(self, data):
        self.database.save_predictions_dataset(data)  

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame):            
        self.database.save_checkpoints_summary(data)
    


    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'SCADS'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):           
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(
            CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_path, exist_ok=True) 
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model, path):
        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model {os.path.basename(path)} has been saved')

    #--------------------------------------------------------------------------
    def save_training_configuration(self, path, history : dict, configuration : dict, 
                                    metadata : dict):       
        os.makedirs(os.path.join(path, 'configuration'), exist_ok=True)         
        config_path = os.path.join(path, 'configuration', 'configuration.json')
        metadata_path = os.path.join(path, 'configuration', 'metadata.json')
        history_path = os.path.join(path, 'configuration', 'session_history.json')         
        
        # Save training and model configuration
        with open(config_path, 'w') as f:
            json.dump(configuration, f)
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration, session history and metadata saved for {os.path.basename(path)}')  

    #-------------------------------------------------------------------------- 
    def scan_checkpoints_folder(self):
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)
        
        return model_folders    

    #--------------------------------------------------------------------------
    def load_training_configuration(self, path):
        config_path = os.path.join(path, 'configuration', 'configuration.json')
        metadata_path = os.path.join(path, 'configuration', 'metadata.json')
        history_path = os.path.join(path, 'configuration', 'session_history.json')
        # Load training and model configuration
        with open(config_path, 'r') as f:
            configuration = json.load(f)         
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f) 
        # Load session history
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configuration, metadata, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):        
        logger.debug('Generating model architecture graph')
        plot_path = os.path.join(path, 'model_layout.png')       
        plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
            
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name : str):                             
        custom_objects = {
            'MaskedSparseCategoricalCrossentropy': MaskedMeanSquaredError,
            'MaskedAccuracy': MaskedRSquared,
            'LinearDecayLRScheduler': LinearDecayLRScheduler}             

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = load_model(model_path, custom_objects=custom_objects)
        configuration, metadata, session = self.load_training_configuration(checkpoint_path)        
            
        return model, configuration, metadata, session, checkpoint_path
     