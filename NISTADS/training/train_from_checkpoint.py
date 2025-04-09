# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.data.loader import TrainingDataLoader
from NISTADS.commons.utils.learning.training import ModelTraining
from NISTADS.commons.utils.validation.reports import log_training_report
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------    
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()      
    model, configuration, metadata, _, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True) 

    # setting device for training    
    trainer = ModelTraining(configuration, metadata)    
    trainer.set_device()     

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]  
    #--------------------------------------------------------------------------    
    logger.info('Loading preprocessed data and building dataloaders')     
    dataserializer = DataSerializer(configuration)
    train_data, val_data, metadata, vocabularies = dataserializer.load_train_and_validation_data()           

    # 3. [BUILD TRAINING DATALODER]
    #--------------------------------------------------------------------------   
    builder = TrainingDataLoader(CONFIG)   
    train_dataset, validation_dataset = builder.build_training_dataloader(
        train_data, val_data)

    # 4. [TRAIN MODEL]  
    # Setting callbacks and training routine for the machine learning model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/     
    #--------------------------------------------------------------------------        
    log_training_report(train_data, val_data, configuration, metadata)                       

    # resume training from pretrained model    
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                        from_checkpoint=True)



