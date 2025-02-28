import os
import sys
import json
import pandas as pd
import keras
from datetime import datetime

from NISTADS.commons.utils.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared
from NISTADS.commons.utils.learning.scheduler import LRScheduler
from NISTADS.commons.constants import CONFIG, PROCESSED_PATH, DATA_PATH, CHECKPOINT_PATH
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
        self.SCADS_data_path = os.path.join(DATA_PATH, 'single_component_adsorption.csv') 
        self.BMADS_data_path = os.path.join(DATA_PATH, 'binary_mixture_adsorption.csv')
        self.guest_path = os.path.join(DATA_PATH, 'adsorbates_dataset.csv')  
        self.host_path = os.path.join(DATA_PATH, 'adsorbents_dataset.csv') 

        self.processed_SCADS_path = os.path.join(PROCESSED_PATH, 'SCADS_dataset.csv')
        self.metadata_path = os.path.join(PROCESSED_PATH, 'preprocessing_metadata.json') 
        self.smile_vocabulary_path = os.path.join(PROCESSED_PATH, 'SMILE_tokenization_index.json')
        self.ads_vocabulary_path = os.path.join(PROCESSED_PATH, 'adsorbents_index.json')

        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'
        self.adsorbate_SMILE_COL = 'adsorbate_encoded_SMILE'   
        self.adsorbent_SMILE_COL = 'adsorbent_encoded_SMILE'     
        self.parameters = configuration["dataset"]
        self.configuration = configuration 

        self.sanitizer = DataSanitizer(configuration)         
        
    #--------------------------------------------------------------------------
    def load_datasets(self, get_materials=True): 
        guest_properties, host_properties = None, None             
        adsorption_data = pd.read_csv(self.SCADS_data_path, encoding='utf-8', sep=';') 
        if get_materials:       
            guest_properties = pd.read_csv(self.guest_path, encoding='utf-8', sep=';')        
            host_properties = pd.read_csv(self.host_path, encoding='utf-8', sep=';')

        guest_properties = self.sanitizer.convert_string_to_series(guest_properties) 
        host_properties = self.sanitizer.convert_string_to_series(host_properties)                 

        return adsorption_data, guest_properties, host_properties 

    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, processed_data : pd.DataFrame, smile_vocabulary={},
                               adsorbent_vocabulary={}):
        
        metadata = self.configuration.copy()
        metadata['date'] = datetime.now().strftime("%Y-%m-%d")
        metadata['SMILE_vocabulary_size'] = len(smile_vocabulary)  
        metadata['adsorbent_vocabulary_size'] = len(adsorbent_vocabulary) 

        processed_data = self.sanitizer.convert_series_to_string(processed_data)         
        processed_data.to_csv(self.processed_SCADS_path, index=False, sep=';', encoding='utf-8')               
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)              
        with open(self.smile_vocabulary_path, 'w') as file:
            json.dump(smile_vocabulary, file, indent=4)    
        with open(self.ads_vocabulary_path, 'w') as file:
            json.dump(adsorbent_vocabulary, file, indent=4)             

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self): 
        processed_data = pd.read_csv(self.processed_SCADS_path, encoding='utf-8', sep=';') 
        processed_data = self.sanitizer.convert_string_to_series(processed_data)
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        with open(self.smile_vocabulary_path, 'r') as file:
            smile_vocabulary = json.load(file)
        with open(self.ads_vocabulary_path, 'r') as file:
            ads_vocabulary = json.load(file)            
        
        return processed_data, metadata, smile_vocabulary, ads_vocabulary         
    
    #--------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data=None, host_data=None):                   
        if guest_data is not None:
            guest_data = self.sanitizer.convert_series_to_string(guest_data) 
            dataframe = pd.DataFrame.from_dict(guest_data)          
            dataframe.to_csv(self.guest_path, index=False, sep=';', encoding='utf-8')
        if host_data is not None:
            host_data = self.sanitizer.convert_series_to_string(host_data)     
            dataframe = pd.DataFrame.from_dict(host_data)          
            dataframe.to_csv(self.host_path, index=False, sep=';', encoding='utf-8')    

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component : pd.DataFrame, binary_mixture : pd.DataFrame): 
        single_component = self.sanitizer.convert_series_to_string(single_component) 
        binary_mixture = self.sanitizer.convert_series_to_string(binary_mixture)         
        single_component.to_csv(self.SCADS_data_path, index=False, sep=';', encoding='utf-8')        
        binary_mixture.to_csv(self.BMADS_data_path, index=False, sep=';', encoding='utf-8')    

    #--------------------------------------------------------------------------
    def copy_data_to_checkpoint(self, checkpoint_path):        
        data_folder = os.path.join(checkpoint_path, 'data')        
        os.makedirs(data_folder, exist_ok=True)        
        os.system(f'cp {self.processed_SCADS_path} {data_folder}')
        os.system(f'cp {self.metadata_path} {data_folder}')
        

    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'SCADS'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):           
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_path = os.path.join(CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
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
    def save_session_configuration(self, path, history : dict, configurations : dict):
        
        os.makedirs(os.path.join(path, 'configurations'), exist_ok=True)         
        config_path = os.path.join(path, 'configurations', 'configurations.json')
        history_path = os.path.join(path, 'configurations', 'session_history.json')        

        # Save training and model configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)       

        # Save session history
        with open(history_path, 'w') as f:
            json.dump(history, f)

        logger.debug(f'Model configuration and session history have been saved at {path}')  

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
        with open(config_path, 'r') as f:
            configurations = json.load(f)        

        history_path = os.path.join(path, 'configurations', 'session_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)

        return configurations, history

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):        
        logger.debug('Generating model architecture graph')
        plot_path = os.path.join(path, 'model_layout.png')       
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                    show_layer_names=True, show_layer_activations=True, 
                    expand_nested=True, rankdir='TB', dpi=400)
            
    #--------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_name):                     
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedMeanSquaredError,
                          'MaskedAccuracy': MaskedRSquared,
                          'LRScheduler': LRScheduler}             

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
        configuration, history = self.load_session_configuration(checkpoint_path)           
            
        return model, configuration, history, checkpoint_path
