# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.serializer import DataSerializer
from NISTADS.commons.utils.data.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.data.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.data.process.normalization import FeatureNormalizer, AdsorbentEncoder
from NISTADS.commons.utils.data.process.sequences import PressureUptakeSeriesProcess, SMILETokenization
from NISTADS.commons.utils.data.process.aggregation import AggregateDatasets
from NISTADS.commons.utils.data.process.conversion import PQ_units_conversion
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load source data from csv files
    logger.info(f'Loading NISTADS datasets from {DATA_PATH}')
    dataserializer = DataSerializer(CONFIG)
    adsorption_data, guest_data, host_data = dataserializer.load_datasets() 
    logger.info(f'{adsorption_data.shape[0]} measurements in the dataset')
    logger.info(f'{guest_data.shape[0]} total guests (adsorbates species) in the dataset')
    logger.info(f'{host_data.shape[0]} total hosts (adsorbent materials) in the dataset')

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # group single component data based on the experiment name 
    # merge adsorption data with retrieved materials properties (guest and host)
    aggregator = AggregateDatasets(CONFIG)
    processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)
    processed_data = aggregator.join_materials_properties(processed_data, guest_data, host_data)
    logger.info(f'Dataset has been aggregated for a total of {processed_data.shape[0]} experiments')         

    # convert pressure and uptake into standard units:
    # pressure to Pascal, uptake to mol/g
    sequencer = PressureUptakeSeriesProcess(CONFIG)
    logger.info('Converting pressure into Pascal and uptake into mol/g')   
    processed_data = PQ_units_conversion(processed_data) 

    # exlude all data outside given boundaries, such as negative temperature values 
    # and pressure and uptake values below zero or above upper limits
    sanitizer = DataSanitizer(CONFIG)  
    processed_data = sanitizer.exclude_OOB_values(processed_data)
    
    # remove repeated zero values at the beginning of pressure and uptake series  
    # then filter out experiments with not enough measurements 
    processed_data = sequencer.remove_leading_zeros(processed_data)   
    processed_data = sequencer.filter_by_sequence_size(processed_data)          

    # 3. [PROCESS MOLECULAR INPUTS]
    #--------------------------------------------------------------------------  
    tokenization = SMILETokenization(CONFIG) 
    logger.info('Tokenizing SMILE sequences for adsorbate species')   
    processed_data, smile_vocab = tokenization.process_SMILE_sequences(processed_data)    

    # 4. [SPLIT BASED NORMALIZATION AND ENCODING]
    #-------------------------------------------------------------------------- 
    # split data into train set and validation set
    logger.info('Preparing dataset of images and captions based on splitting size')  
    splitter = TrainValidationSplit(CONFIG, processed_data)     
    train_data, validation_data = splitter.split_train_and_validation() 

    # normalize pressure and uptake series using max values computed from 
    # the training set, then pad sequences to a fixed length
    normalizer = FeatureNormalizer(CONFIG, train_data)
    train_data = normalizer.normalize_molecular_features(train_data) 
    train_data = normalizer.PQ_series_normalization(train_data) 
    validation_data = normalizer.normalize_molecular_features(validation_data) 
    validation_data = normalizer.PQ_series_normalization(validation_data)      
   
    # add padding to pressure and uptake series to match max length
    train_data = sequencer.PQ_series_padding(train_data)     
    validation_data = sequencer.PQ_series_padding(validation_data)     

    encoding = AdsorbentEncoder(CONFIG, train_data)    
    train_data = encoding.encode_adsorbents_by_name(train_data)
    validation_data = encoding.encode_adsorbents_by_name(validation_data)    
    adsorbent_vocab = encoding.mapping 

    # 5. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer   
    train_data = sanitizer.isolate_preprocessed_features(train_data)
    validation_data = sanitizer.isolate_preprocessed_features(validation_data)           
    dataserializer.save_train_and_validation_data(
        train_data, validation_data, smile_vocab, 
        adsorbent_vocab, normalizer.statistics)












