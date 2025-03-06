import random 

# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.validation.reports import DataAnalysisPDF
from NISTADS.commons.utils.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.dataloader.tensordata import TrainingDatasetBuilder
from NISTADS.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.validation.reports import evaluation_report
from NISTADS.commons.utils.validation.checkpoints import ModelEvaluationSummary
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    evaluation_batch_size = 64
    num_images_to_evaluate = 6

    # 1. [LOAD DATASET]
    #--------------------------------------------------------------------------  
    summarizer = ModelEvaluationSummary()    
    checkpoints_summary = summarizer.checkpoints_summary() 
    logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')  
    
    # 2. [LOAD MODEL]
    #--------------------------------------------------------------------------
    # selected and load the pretrained model, then print the summary 
    modelserializer = ModelSerializer()         
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()
    model.summary(expand_nested=True)   

    # load data from csv, add paths to images 
    dataserializer = DataSerializer(CONFIG)
    processed_data, metadata, smile_vocabulary, ads_vocabulary = dataserializer.load_preprocessed_data() 

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators
    splitter = TrainValidationSplit(configuration, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation()    

    # 3. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------
    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    builder = TrainingDatasetBuilder(configuration)  
    train_dataset, validation_dataset = builder.build_model_dataloader(
        train_data, validation_data, configuration)

    # 4. [EVALUATE ON TRAIN AND VALIDATION]
    #--------------------------------------------------------------------------   
    evaluation_report(model, train_dataset, validation_dataset)     

    # 2. [INITIALIZE PDF REPORT]
    #--------------------------------------------------------------------------
    report = DataAnalysisPDF()
