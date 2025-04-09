import os
import sys
import json
import pandas as pd
import keras
from datetime import datetime

from NISTADS.commons.utils.data.database import AdsorptionDatabase
from NISTADS.commons.utils.data.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared
from NISTADS.commons.utils.learning.scheduler import LinearDecayLRScheduler
from NISTADS.commons.constants import CONFIG, DATA_PATH, METADATA_PATH, CHECKPOINT_PATH
from NISTADS.commons.logger import logger


###############################################################################
def checkpoint_selection_menu(models_list):
    index_list = [idx + 1 for idx, item in enumerate(models_list)]     
    print('Currently available pretrained models:')             
    for i, directory in enumerate(models_list):
        print(f'{i + 1} - {directory}')                         
    while True:
        try:
            selection_index = int(input('\nSelect the pretrained model: '))
            print()
        except ValueError:
            logger.error('Invalid choice for the pretrained model, asking again')
            continue
        if selection_index in index_list:
            break
        else:
            logger.warning('Model does not exist, please select a valid index')

    return selection_index


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):  
        self.metadata_path = os.path.join(
            METADATA_PATH, 'preprocessing_metadata.json') 
        self.smile_vocabulary_path = os.path.join(
            METADATA_PATH, 'SMILE_tokenization_index.json')
        self.ads_vocabulary_path = os.path.join(
            METADATA_PATH, 'adsorbents_index.json')

        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'             
        self.parameters = configuration["dataset"]
        self.configuration = configuration 

        self.database = AdsorptionDatabase(configuration)
        self.sanitizer = DataSanitizer(configuration)         
        
    #--------------------------------------------------------------------------
    def load_datasets(self):                
        return self.database.load_source_data_table()
    
    #--------------------------------------------------------------------------
    def load_inference_data(self):              
        return self.database.load_inference_data_table()   
    
    #--------------------------------------------------------------------------
    def load_processed_data(self): 
        # load preprocessed data from database and convert joint strings to list 
        processed_data = self.database.load_processed_data_table()
        processed_data = self.sanitizer.convert_string_to_series(processed_data) 

        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        with open(self.smile_vocabulary_path, 'r') as file:
            smile_vocabulary = json.load(file)
        with open(self.ads_vocabulary_path, 'r') as file:
            ads_vocabulary = json.load(file)            
        
        return processed_data, metadata, smile_vocabulary, ads_vocabulary  

    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, processed_data : pd.DataFrame, smile_vocabulary={},
                               adsorbent_vocabulary={}, normalization_stats={}):
        # convert list to joint string and save preprocessed data to database
        processed_data = self.sanitizer.convert_series_to_string(processed_data)        
        self.database.save_processed_data_table(processed_data) 
        
        with open(self.smile_vocabulary_path, 'w') as file:
            json.dump(smile_vocabulary, file, indent=4)    
        with open(self.ads_vocabulary_path, 'w') as file:
            json.dump(adsorbent_vocabulary, file, indent=4)        
         
        metadata = {'seed' : self.configuration['SEED'], 
                    'dataset' : self.configuration['dataset'],
                    'date' : datetime.now().strftime("%Y-%m-%d"),
                    'SMILE_vocabulary_size' : len(smile_vocabulary),
                    'adsorbent_vocabulary_size' : len(adsorbent_vocabulary), 
                    'normalization' : {
                        self.P_COL : float(normalization_stats[self.P_COL]),
                        self.Q_COL : float(normalization_stats[self.Q_COL]),
                        'temperature' : float(normalization_stats['temperature']),
                        'adsorbate_molecular_weight' : float(normalization_stats['adsorbate_molecular_weight'])}}  
               
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4) 
   
    #--------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data=None, host_data=None):                   
        if guest_data is not None:
            guest_data = pd.DataFrame.from_dict(guest_data)
            guest_data = self.sanitizer.convert_series_to_string(guest_data)            
        if host_data is not None:
            host_data = pd.DataFrame.from_dict(host_data)
            host_data = self.sanitizer.convert_series_to_string(host_data)    

        self.database.save_materials_table(guest_data, host_data)

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component : pd.DataFrame, binary_mixture : pd.DataFrame):
        self.database.save_experiments_table(single_component, binary_mixture)  

    #--------------------------------------------------------------------------
    def save_predictions_dataset(self, data : pd.DataFrame):
        self.database.save_inference_data_table(data)  

    

          

    
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
        os.makedirs(os.path.join(checkpoint_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_path}')
        
        return checkpoint_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):
        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_session_configuration(self, path, history : dict, configurations : dict, metadata : dict):        
        os.makedirs(os.path.join(path, 'configurations'), exist_ok=True)         
        config_path = os.path.join(path, 'configurations', 'configurations.json')
        metadata_path = os.path.join(path, 'configurations', 'metadata.json')       
        history_path = os.path.join(path, 'configurations', 'session_history.json')         

        # Save training and model configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)

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
    def load_session_configuration(self, path):
        config_path = os.path.join(path, 'configurations', 'configurations.json')
        metadata_path = os.path.join(path, 'configurations', 'metadata.json') 
        history_path = os.path.join(path, 'configurations', 'session_history.json')
        # Load training and model configurations
        with open(config_path, 'r') as f:
            configurations = json.load(f)         
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)        
        # Load session history
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configurations, metadata, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):        
        logger.debug('Generating model architecture graph')
        plot_path = os.path.join(path, 'model_layout.png')       
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
            
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name):                     
        custom_objects = {
            'MaskedSparseCategoricalCrossentropy': MaskedMeanSquaredError,
            'MaskedAccuracy': MaskedRSquared,
            'LinearDecayLRScheduler': LinearDecayLRScheduler}             

        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
        model_path = os.path.join(checkpoint_path, 'saved_model.keras') 
        model = keras.models.load_model(model_path, custom_objects=custom_objects) 
        
        return model
            
    #-------------------------------------------------------------------------- 
    def select_and_load_checkpoint(self):         
        model_folders = self.scan_checkpoints_folder()
        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            selection_index = checkpoint_selection_menu(model_folders)                    
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[selection_index-1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, model_folders[0])
            logger.info(f'Since only checkpoint {os.path.basename(checkpoint_path)} is available, it will be loaded directly')
                          
        # effectively load the model using keras builtin method
        # load configuration data from .json file in checkpoint folder
        model = self.load_checkpoint(checkpoint_path)       
        configuration, metadata, history = self.load_session_configuration(checkpoint_path)           
            
        return model, configuration, metadata, history, checkpoint_path
