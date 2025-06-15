# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.loader import InferenceDataLoader
from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.validation.reports import evaluation_report
from NISTADS.commons.utils.validation.checkpoints import ModelEvaluationSummary
from NISTADS.commons.utils.validation.experiments import AdsorptionPredictionsQuality
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':    

    # 1. [CREATE CHECKPOINTS SUMMARY]
    #--------------------------------------------------------------------------  
    summarizer = ModelEvaluationSummary(self.configuration)    
    checkpoints_summary = summarizer.checkpoints_summary()     
    logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')  
    
    # 2. [LOAD MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, metadata, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)   

    # 3. [LOAD DATA]
    #--------------------------------------------------------------------------  
    # load preprocessed data and associated metadata
    dataserializer = DataSerializer(configuration)
    _, val_data, metadata, vocabularies = dataserializer.load_train_and_validation_data()
    logger.info(f'Validation data has been loaded: {val_data.shape[0]} samples')    
   
    loader = InferenceDataLoader(configuration)      
    validation_dataset = loader.build_inference_dataloader(val_data)

    # 4. [EVALUATE ON VALIDATION]
    #-------------------------------------------------------------------------- 
    logger.info('Calculating model evaluation loss and metrics')    
    evaluation_report(model, validation_dataset)  

    # 5. [COMPARE RECONTRUCTED ISOTHERMS]
    #--------------------------------------------------------------------------      
    validator = AdsorptionPredictionsQuality(
        model, CONFIG, metadata, checkpoint_path)      
    validator.visualize_adsorption_isotherms(val_data)       

