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
from NISTADS.app.constants import METADATA_PATH, CHECKPOINT_PATH
from NISTADS.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration : dict):
        self.seed = configuration.get('general_seed', 42)
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'                
        self.configuration = configuration
        self.database = AdsorptionDatabase()

        self.metadata_path = os.path.join(
            METADATA_PATH, 'preprocessing_metadata.json') 
        self.smile_vocabulary_path = os.path.join(
            METADATA_PATH, 'SMILE_tokenization_index.json')
        self.ads_vocabulary_path = os.path.join(
            METADATA_PATH, 'adsorbents_index.json')
        
    #--------------------------------------------------------------------------
    def load_adsorption_datasets(self, sample_size=1.0):          
        adsorption_data, guest_data, host_data = self.database.load_source_dataset()
        adsorption_data = adsorption_data.sample(
            frac=sample_size, random_state=self.seed).reset_index(drop=True)

        return adsorption_data, guest_data, host_data
    
    #--------------------------------------------------------------------------
    def load_inference_data(self):              
        return self.database.load_inference_data_table()   
    
    #--------------------------------------------------------------------------
    def load_train_and_validation_data(self): 
        # load preprocessed data from database and convert joint strings to list
        sanitizer = DataSanitizer(self.configuration) 
        train_data, val_data = self.database.load_train_and_validation()
        train_data = sanitizer.convert_string_to_series(train_data) 
        val_data = sanitizer.convert_string_to_series(val_data) 

        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        with open(self.smile_vocabulary_path, 'r') as file:
            smile_vocabulary = json.load(file)
        with open(self.ads_vocabulary_path, 'r') as file:
            ads_vocabulary = json.load(file)  

        vocabularies = {'smile_vocab' : smile_vocabulary, 
                        'adsorbents_vocab' : ads_vocabulary}         
        
        return train_data, val_data, metadata, vocabularies  

    #--------------------------------------------------------------------------
    def save_train_and_validation_data(self, train_data, validation_data,
                                       smile_vocabulary, ads_vocabulary, normalization_stats={}):      

        # convert list to joint string and save preprocessed data to database
        sanitizer = DataSanitizer(self.configuration)    
        train_data = sanitizer.convert_series_to_string(train_data)   
        validation_data = sanitizer.convert_series_to_string(validation_data)      
        self.database.save_train_and_validation(train_data, validation_data) 
        
        with open(self.smile_vocabulary_path, 'w') as file:
            json.dump(smile_vocabulary, file, indent=4)    
        with open(self.ads_vocabulary_path, 'w') as file:
            json.dump(ads_vocabulary, file, indent=4)        
         
        metadata = {'seed' : self.seed, 
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'max_measurements' : self.configuration.get('max_measurements', 1000),
                    'SMILE_sequence_length' : self.configuration.get('SMILE_sequence_size', 30),
                    'SMILE_vocabulary_size' : len(smile_vocabulary),
                    'adsorbent_vocabulary_size' : len(ads_vocabulary), 
                    'normalization' : {
                        self.P_COL : float(normalization_stats[self.P_COL]),
                        self.Q_COL : float(normalization_stats[self.Q_COL]),
                        'temperature' : float(normalization_stats['temperature']),
                        'adsorbate_molecular_weight' : float(normalization_stats['adsorbate_molecular_weight'])}}  
               
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4) 
   
    #--------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data=None, host_data=None):
        sanitizer = DataSanitizer(self.configuration)                       
        if guest_data is not None:
            guest_data = pd.DataFrame.from_dict(guest_data)
            guest_data = sanitizer.convert_series_to_string(guest_data)            
        if host_data is not None:
            host_data = pd.DataFrame.from_dict(host_data)
            host_data = sanitizer.convert_series_to_string(host_data)    

        self.database.save_materials_table(guest_data, host_data)

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component, binary_mixture):
        self.database.save_experiments_table(single_component, binary_mixture)  

    #--------------------------------------------------------------------------
    def save_predictions_dataset(self, data):
        self.database.save_predictions_table(data)  


    
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
    def save_training_configuration(self, path, history, configuration, metadata):        
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

        logger.debug(f'Model configuration and session history saved for {os.path.basename(path)}')  

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
        configuration, session = self.load_training_configuration(checkpoint_path)        
            
        return model, configuration, session, checkpoint_path 
     