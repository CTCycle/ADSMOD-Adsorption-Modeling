# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.data.loader import TrainingDataLoader
from NISTADS.commons.utils.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.learning.training import ModelTraining
from NISTADS.commons.utils.validation.reports import log_training_report
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------    
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()      
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)  
    
    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    # initialize training device, allows changing device prior to initializing the generators
    #--------------------------------------------------------------------------    
    # load saved tf.datasets from the proper folders in the checkpoint directory
    logger.info('Loading preprocessed data and building dataloaders')     
    dataserializer = DataSerializer(configuration) 
    processed_data, metadata, smile_vocabulary, ads_vocabulary = dataserializer.load_preprocessed_data()


