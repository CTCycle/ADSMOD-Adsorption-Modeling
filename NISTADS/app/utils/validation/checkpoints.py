import os
import shutil
import pandas as pd

from NISTADS.app.utils.data.serializer import ModelSerializer
from NISTADS.app.interface.workers import check_thread_status, update_progress_callback
from NISTADS.app.constants import CHECKPOINT_PATH
from NISTADS.app.logger import logger


# [LOAD MODEL]
################################################################################
class ModelEvaluationSummary:

    def __init__(self, configuration):
         
        self.configuration = configuration

    #---------------------------------------------------------------------------
    def scan_checkpoint_folder(self):
        model_paths = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():                
                pretrained_model_path = os.path.join(entry.path, 'saved_model.keras')                
                if os.path.isfile(pretrained_model_path):
                    model_paths.append(entry.path)
                

        return model_paths  

    #---------------------------------------------------------------------------
    def get_checkpoints_summary(self, **kwargs):       
        serializer = ModelSerializer()    
        # look into checkpoint folder to get pretrained model names      
        model_paths = self.scan_checkpoint_folder()
        model_parameters = []            
        for i, model_path in enumerate(model_paths):            
            model = serializer.load_checkpoint(model_path)
            configuration, history = serializer.load_training_configuration(model_path)
            model_name = os.path.basename(model_path)                   
            precision = 16 if configuration.get("use_mixed_precision", np.nan) else 32 
            chkp_config = {'Sample size': configuration.get("train_sample_size", np.nan),
                           'Validation size': configuration.get("validation_size", np.nan),
                           'Seed': configuration.get("train_seed", np.nan),                           
                           'Precision (bits)': precision,                      
                           'Epochs': configuration.get("epochs", np.nan),
                           'Additional Epochs': configuration.get("additional_epochs", np.nan),
                           'Batch size': configuration.get("batch_size", np.nan),           
                           'Split seed': configuration.get("split_seed", np.nan),
                           'Image augmentation': configuration.get("img_augmentation", np.nan),
                           'Image height': 224,
                           'Image width': 224,
                           'Image channels': 3,                          
                           'JIT Compile': configuration.get("jit_compile", np.nan),                           
                           'Device': configuration.get("device", np.nan),                                                      
                           'Number workers': configuration.get("num_workers", np.nan),
                           'LR Scheduler': configuration.get("use_scheduler", np.nan),                            
                           'LR Scheduler - Post Warmup LR': configuration.get("post_warmup_LR", np.nan),
                           'LR Scheduler - Warmup Steps': configuration.get("warmup_steps", np.nan),
                           'Temperature': configuration.get("train_temperature", np.nan),                            
                           'Tokenizer': configuration["dataset"].get("TOKENIZER", np.nan),                            
                           'Max report size': configuration["dataset"].get("MAX_REPORT_SIZE", np.nan),
                           'Number of heads': configuration["model"].get("ATTENTION_HEADS", np.nan),
                           'Number of encoders': configuration["model"].get("NUM_ENCODERS", np.nan),
                           'Number of decoders': configuration["model"].get("NUM_DECODERS", np.nan),
                           'Embedding dimensions': configuration["model"].get("EMBEDDING_DIMS", np.nan),
                           'Frozen image encoder': configuration["model"].get("FREEZE_IMG_ENCODER", np.nan)}

            model_parameters.append(chkp_config)

            # check for thread status and progress bar update   
            check_thread_status(kwargs.get('worker', None))         
            update_progress_callback(
                i+1, len(model_paths), kwargs.get('progress_callback', None)) 

        dataframe = pd.DataFrame(model_parameters)
        self.database.save_checkpoints_summary(dataframe)      
            
        return dataframe
    
    #--------------------------------------------------------------------------
    def get_evaluation_report(self, model, validation_dataset):     
        validation = model.evaluate(validation_dataset, verbose=1)    
        logger.info(
            f'Mean Square Error Loss {validation[0]:.3f} - R square metrics {validation[1]:.3f}')
        
    
