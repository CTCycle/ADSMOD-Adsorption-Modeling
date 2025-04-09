# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.data.loader import TrainingDataLoader
from NISTADS.commons.utils.data.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.learning.models import SCADSModel
from NISTADS.commons.utils.learning.training import ModelTraining
from NISTADS.commons.utils.validation.reports import log_training_report
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------   
    dataserializer = DataSerializer(CONFIG)
    data, metadata, smile_vocab, ads_vocab = dataserializer.load_processed_data()         
  
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, data)     
    train_data, validation_data = splitter.split_train_and_validation()        

    # 2. [BUILD TRAINING DATALODER]
    #-------------------------------------------------------------------------- 
    logger.info('Building model data loaders with prefetching and parallel processing')   
    builder = TrainingDataLoader(CONFIG)   
    train_dataset, validation_dataset = builder.build_training_dataloader(
        train_data, validation_data) 
    
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder()
    
    # 3. [SET DEVICE]
    #-------------------------------------------------------------------------- 
    logger.info('Setting device for training operations based on user configurations') 
    trainer = ModelTraining(CONFIG, metadata) 
    trainer.set_device()  

    # 4. [TRAIN MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the machine learning model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    log_training_report(train_data, validation_data, CONFIG, metadata)

    # initialize and compile the captioning model  
    logger.info('Building SCADS model based on user configurations')  
    wrapper = SCADSModel(metadata, CONFIG)
    model = wrapper.get_model(model_summary=True) 

    # generate graphviz plot fo the model layout       
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)

    