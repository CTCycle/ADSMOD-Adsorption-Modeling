# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer, load_all_datasets
from NISTADS.commons.utils.process.sanitizer import AdsorptionDataSanitizer
from NISTADS.commons.utils.process.sequences import PressureUptakeSeriesProcess
from NISTADS.commons.utils.process.splitting import DatasetSplit 
from NISTADS.commons.utils.process.aggregation import merge_all_datasets, aggregate_adsorption_measurements
from NISTADS.commons.utils.process.conversion import units_conversion
from NISTADS.commons.utils.process.sequences import PressureUptakeSeriesProcess
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, retrieve and merge molecular properties 
    logger.info(f'Loading NISTADS datasets from {DATA_PATH}')
    adsorption_data, guests_data, hosts_data = load_all_datasets() 
    logger.info(f'{adsorption_data.shape[0]} measurements detected in the dataset')
    logger.info(f'{guests_data.shape[0]} total guests (adsorbates species) detected in the dataset')
    logger.info(f'{hosts_data.shape[0]} total hosts (adsorbents materials) detected in the dataset')

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------     
    # exlude all data outside given boundaries:
    # negative temperature
    # pressure and uptake below or above the given boundaries
    sanitizer = AdsorptionDataSanitizer()
    processed_data = sanitizer.exclude_outside_boundary(adsorption_data)  
    # group data from single measurements based on experiment identity  
    # merge data from adsorption and materials datasets   
    processed_data = aggregate_adsorption_measurements(processed_data)
    aggregated_data = merge_all_datasets(processed_data, guests_data, hosts_data)   

    # rectify sequences of pressure/uptake points through following steps:
    # 1. remove repeated zero values at the beginning of the series
    sequencer = PressureUptakeSeriesProcess()
    aggregated_data = sequencer.remove_leading_zeros(aggregated_data) 

    # sanitize experiments removing those where measurements number is outside acceptable values 
    aggregated_data = sequencer.select_by_sequence_size(aggregated_data)    

    # convert and normalize units
    converted_data = units_conversion(aggregated_data)
    converted_data = sequencer.sequence_padding(converted_data) 

    # 3. [PROCESS MOLECULAR INPUTS]
    #--------------------------------------------------------------------------  
    tokenization = TokenWizard(CONFIG)    
    train_data, validation_data = tokenization.tokenize_text_corpus(train_data, validation_data)
    vocabulary_size = tokenization.vocabulary_size
   

    # 3. [PREPARE ML DATASET]
    #--------------------------------------------------------------------------     
    train_X, val_X, train_Y, val_Y = self.splitter.split_train_and_validation(aggregated_data)

    # train_exp, train_guest, train_host, train_pressure = self.splitter.isolate_inputs(train_X)
    # val_exp, val_guest, val_host, val_pressure = self.splitter.isolate_inputs(val_X)

    # processed_data = {'train inputs' : (train_exp, train_guest, train_host, train_pressure),
    #                   'train output' : train_Y,
    #                   'validation' : (val_exp, val_guest, val_host, val_pressure),
    #                   'validation output' :val_Y}

    # 3. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer
    dataserializer = DataSerializer()
    dataserializer.save_preprocessed_data(train_data, validation_data, vocabulary_size)












