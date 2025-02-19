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
from NISTADS.commons.utils.learning.models import SCADSModel
from NISTADS.commons.utils.learning.training import ModelTraining
from NISTADS.commons.utils.validation.reports import log_training_report
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PREPROCESSED DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, add paths to images 
    dataserializer = DataSerializer(CONFIG)
    processed_data, metadata, smile_vocabulary, ads_vocabulary = dataserializer.load_preprocessed_data()    
    
    # 2. [SPLIT DATA]
    #--------------------------------------------------------------------------
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()  

    # create subfolder for preprocessing data
    modelserializer = ModelSerializer()
    checkpoint_path = modelserializer.create_checkpoint_folder()       

    # 3. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    logger.info('Building NISTADS model and data loaders')     
    trainer = ModelTraining(CONFIG) 
    trainer.set_device()    
       
    # create the tf.datasets using the previously initialized generators 
    builder = TensorDatasetBuilder(CONFIG)   
    train_dataset, validation_dataset = builder.build_model_dataloader(train_data, validation_data)  

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    log_training_report(train_data, validation_data, CONFIG, metadata)

    # initialize and compile the captioning model    
    wrapper = SCADSModel(metadata, CONFIG)
    model = wrapper.get_model(model_summary=True) 

    # generate graphviz plot fo the model layout       
    modelserializer.save_model_plot(model, checkpoint_path)              

    # perform training and save model at the end
    trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path)

    