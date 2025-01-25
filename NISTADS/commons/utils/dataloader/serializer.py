import os 

import sys
import json
import pandas as pd
import keras
from datetime import datetime

from NISTADS.commons.constants import CONFIG, PROCESSED_PATH, DATA_PATH, CHECKPOINT_PATH
from NISTADS.commons.logger import logger



# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):        
        
        self.SCADS_data_path = os.path.join(DATA_PATH, 'single_component_adsorption.csv') 
        self.BMADS_data_path = os.path.join(DATA_PATH, 'binary_mixture_adsorption.csv')
        self.guest_path = os.path.join(DATA_PATH, 'guests_dataset.csv')  
        self.host_path = os.path.join(DATA_PATH, 'hosts_dataset.csv') 

        self.processed_SCADS_path = os.path.join(PROCESSED_PATH, 'SCADS_dataset.csv')
        self.metadata_path = os.path.join(PROCESSED_PATH, 'preprocessing_metadata.json') 
        self.vocabulary_path = os.path.join(PROCESSED_PATH, 'SMILE_tokenization_index.json')

        self.parameters = configuration["dataset"]
        self.configuration = configuration          
        
    #--------------------------------------------------------------------------
    def load_all_datasets(self):       
        adsorption_data = pd.read_csv(self.SCADS_data_path, encoding='utf-8', sep=';')        
        guest_properties = pd.read_csv(self.guest_path, encoding='utf-8', sep=';')
        host_path = os.path.join(DATA_PATH, 'hosts_dataset.csv') 
        host_properties = pd.read_csv(host_path, encoding='utf-8', sep=';')      

        return adsorption_data, guest_properties, host_properties 

    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, processed_data : pd.DataFrame, smile_vocabulary={}): 

        metadata = self.configuration.copy()
        metadata['date'] = datetime.now().strftime("%Y-%m-%d")
        metadata['vocabulary_size'] = len(smile_vocabulary)              
        processed_data.to_csv(self.processed_SCADS_path, index=False, sep=';', encoding='utf-8')        
        with open(self.metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)              
        with open(self.vocabulary_path, 'w') as file:
            json.dump(smile_vocabulary, file, indent=4)             

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self):                            
        processed_data = pd.read_csv(self.processed_SCADS_path, encoding='utf-8', sep=';', low_memory=False)        
        processed_data['tokens'] = processed_data['tokens'].apply(lambda x : [int(f) for f in x.split()])        
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)        
        with open(self.vocabulary_path, 'r') as file:
            vocabulary = json.load(file)
        
        return processed_data, metadata, vocabulary
    
    #--------------------------------------------------------------------------
    def save_materials_datasets(self, guest_data, host_data):
        dataframe = pd.DataFrame.from_dict(guest_data)          
        dataframe.to_csv(self.guest_path, index=False, sep=';', encoding='utf-8')
        dataframe = pd.DataFrame.from_dict(host_data)          
        dataframe.to_csv(self.host_path, index=False, sep=';', encoding='utf-8')    

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component : pd.DataFrame, binary_mixture : pd.DataFrame):        
        single_component.to_csv(self.SCADS_data_path, index=False, sep=';', encoding='utf-8')        
        binary_mixture.to_csv(self.BMADS_data_path, index=False, sep=';', encoding='utf-8')

    #--------------------------------------------------------------------------
    def copy_data_to_checkpoint(self, checkpoint_path):        
        data_folder = os.path.join(checkpoint_path, 'data')        
        os.makedirs(data_folder, exist_ok=True)        
        os.system(f'cp {self.processed_data_path} {data_folder}')
        os.system(f'cp {self.metadata_path} {data_folder}')
        

    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'SCADS'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):

        '''
        Creates a folder with the current date and time to save the model.

        Keyword arguments:
            None

        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
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

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            path (str): The directory path where the parameters will be saved.

        Returns:
            None  

        '''
        config_folder = os.path.join(path, 'configurations')
        os.makedirs(config_folder, exist_ok=True)

        # Paths to the JSON files
        config_path = os.path.join(config_folder, 'configurations.json')
        history_path = os.path.join(config_folder, 'session_history.json')

        # Function to merge dictionaries
        def merge_dicts(original, new_data):
            for key, value in new_data.items():
                if key in original:
                    if isinstance(value, dict) and isinstance(original[key], dict):
                        merge_dicts(original[key], value)
                    elif isinstance(value, list) and isinstance(original[key], list):
                        original[key].extend(value)
                    else:
                        original[key] = value
                else:
                    original[key] = value    

        # Save the merged configurations
        with open(config_path, 'w') as f:
            json.dump(configurations, f)

        # Load existing session history if the file exists and merge
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                existing_history = json.load(f)
            merge_dicts(existing_history, history)
        else:
            existing_history = history

        # Save the merged session history
        with open(history_path, 'w') as f:
            json.dump(existing_history, f)

        logger.debug(f'Model configuration and session history have been saved and merged at {path}')      

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

        if CONFIG["model"]["SAVE_MODEL_PLOT"]:
            logger.debug('Generating model architecture graph')
            plot_path = os.path.join(path, 'model_layout.png')       
            keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                       show_layer_names=True, show_layer_activations=True, 
                       expand_nested=True, rankdir='TB', dpi=400)
            
    #-------------------------------------------------------------------------- 
    def select_and_load_checkpoint(self): 

        '''
        Load a pretrained Keras model from the specified directory. If multiple model 
        directories are found, the user is prompted to select one. If only one model 
        directory is found, that model is loaded directly. If a 'model_parameters.json' 
        file is present in the selected directory, the function also loads the model 
        parameters.

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                            model parameters from a JSON file. 
                                            Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.
            configuration (dict): The loaded model parameters, or None if the parameters file is not found.

        '''  
        # look into checkpoint folder to get pretrained model names      
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)

        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Currently available pretrained models:')             
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')                         
            while True:
                try:
                    dir_index = int(input('\nSelect the pretrained model: '))
                    print()
                except ValueError:
                    logger.error('Invalid choice for the pretrained model, asking again')
                    continue
                if dir_index in index_list:
                    break
                else:
                    logger.warning('Model does not exist, please select a valid index')
                    
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[dir_index - 1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            logger.info('Loading pretrained model directly as only one is available')
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[0])                 
            
        # Set dictionary of custom objects     
        custom_objects = {'MaskedSparseCategoricalCrossentropy': MaskedSparseCategoricalCrossentropy,
                          'MaskedAccuracy': MaskedAccuracy, 
                          'LRScheduler': LRScheduler}          
        
        # effectively load the model using keras builtin method
        # Load the model with the custom objects 
        model_path = os.path.join(self.loaded_model_folder, 'saved_model.keras')         
        model = keras.models.load_model(model_path, custom_objects=custom_objects) 

        # load configuration data from .json file in checkpoint folder
        configuration, history = self.load_session_configuration(self.loaded_model_folder)          
            
        return model, configuration, history