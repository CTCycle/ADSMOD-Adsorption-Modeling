# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.dataloader.tensordata import TensorDatasetBuilder
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
    
    # 2. [SPLIT DATA]
    #--------------------------------------------------------------------------
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(configuration, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()         

    # 3. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    logger.info('Building NISTADS model and data loaders')     
    trainer = ModelTraining(configuration) 
    trainer.set_device()    
       
    # create the tf.datasets using the previously initialized generators 
    builder = TensorDatasetBuilder(configuration)   
    train_dataset, validation_dataset = builder.build_model_dataloader(train_data, validation_data) 

    # 4. [TRAINING MODEL]  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/     
    #--------------------------------------------------------------------------    
    log_training_report(train_data, validation_data, configuration, 
                        from_checkpoint=True)    

    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                        from_checkpoint=True)



