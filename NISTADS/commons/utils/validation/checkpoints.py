import os
import shutil
import pandas as pd

from NISTADS.commons.utils.data.database import AdsorptionDatabase
from NISTADS.commons.utils.data.serializer import ModelSerializer
from NISTADS.commons.constants import CONFIG, CHECKPOINT_PATH, DATA_PATH
from NISTADS.commons.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration, remove_invalid=False):
        self.remove_invalid = remove_invalid
        self.serializer = ModelSerializer()       
        self.database = AdsorptionDatabase(configuration)
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)
                elif not os.path.isfile(pretrained_model_path) and self.remove_invalid:                    
                    shutil.rmtree(entry.path)

        return model_paths  

    #---------------------------------------------------------------------------
    def checkpoints_summary(self):       
        # look into checkpoint folder to get pretrained model names      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for model_path in model_paths:            
            model = self.serializer.load_checkpoint(model_path)
            configuration, metadata, history = self.serializer.load_training_configurationn(model_path)
            model_name = os.path.basename(model_path)            

            # Extract model name and training type                       
            device_config = configuration["device"]
            precision = 16 if device_config.get("MIXED_PRECISION", 'NA') == True else 32           

            chkp_config = {'Checkpoint name': model_name,                                                  
                           'Sample size': configuration["dataset"].get("SAMPLE_SIZE", 'NA'),
                           'Validation size': configuration["dataset"].get("VALIDATION_SIZE", 'NA'),
                           'Seed': configuration.get("SEED", 'NA'),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration["training"].get("EPOCHS", 'NA'),
                           'Additional Epochs': configuration["training"].get("ADDITIONAL_EPOCHS", 'NA'),
                           'Batch size': configuration["training"].get("BATCH_SIZE", 'NA'),           
                           'Split seed': configuration["dataset"].get("SPLIT_SEED", 'NA'),                                                    
                           'JIT Compile': configuration["model"].get("JIT_COMPILE", 'NA'),
                           'JIT Backend': configuration["model"].get("JIT_BACKEND", 'NA'),
                           'Device': configuration["device"].get("DEVICE", 'NA'),
                           'Device ID': configuration["device"].get("DEVICE_ID", 'NA'),                           
                           'Number of Processors': configuration["device"].get("NUM_PROCESSORS", 'NA'),
                           'Use TensorBoard': configuration["training"].get("USE_TENSORBOARD", 'NA'),                            
                           'LR Scheduler - Initial LR': configuration["training"]["LR_SCHEDULER"].get("INITIAL_LR", 'NA'),
                           'LR Scheduler - Constant steps': configuration["training"]["LR_SCHEDULER"].get("CONSTANT_STEPS", 'NA'),
                           'LR Scheduler -Decay steps': configuration["training"]["LR_SCHEDULER"].get("DECAY_STEPS", 'NA')}

            model_parameters.append(chkp_config)

        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary_table(dataframe)      
            
        return dataframe
    
    
