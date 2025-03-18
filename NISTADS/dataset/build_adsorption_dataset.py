# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import DataSerializer
from NISTADS.commons.utils.process.sanitizer import DataSanitizer
from NISTADS.commons.utils.process.splitting import TrainValidationSplit
from NISTADS.commons.utils.process.normalization import FeatureNormalizer, AdsorbentEncoder
from NISTADS.commons.utils.process.sequences import PressureUptakeSeriesProcess, SMILETokenization
from NISTADS.commons.utils.process.aggregation import AggregateDatasets
from NISTADS.commons.utils.process.conversion import PQ_units_conversion
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
    dataserializer = DataSerializer(CONFIG)
    adsorption_data, guest_data, host_data = dataserializer.load_datasets() 
    logger.info(f'{adsorption_data.shape[0]} measurements in the dataset')
    logger.info(f'{guest_data.shape[0]} total guests (adsorbates species) in the dataset')
    logger.info(f'{host_data.shape[0]} total hosts (adsorbent materials) in the dataset')

    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # group data from single measurements based in the experiments  
    # merge adsorption data with materials properties (guest and host)
    aggregator = AggregateDatasets(CONFIG)
    processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)
    processed_data = aggregator.join_materials_properties(processed_data, guest_data, host_data)
    logger.info(f'Dataset has been aggregated for a total of {processed_data.shape[0]} experiments')         

    # convert and normalize pressure and uptake units:
    # pressure to Pascal, uptake to mol/g
    sequencer = PressureUptakeSeriesProcess(CONFIG)
    logger.info('Converting pressure into Pascal and uptake into mol/g')   
    processed_data = PQ_units_conversion(processed_data) 

    # exlude all data outside given boundaries, such as negative temperature values 
    # and pressure and uptake values below or above the given boundaries
    sanitizer = DataSanitizer(CONFIG)  
    processed_data = sanitizer.exclude_OOB_values(processed_data)

    # rectify sequences of pressure/uptake points through following steps:
    # remove repeated zero values at the beginning of the series  
    # sanitize experiments removing those where measurements number is outside acceptable values 
    processed_data = sequencer.remove_leading_zeros(processed_data)   
    processed_data = sequencer.filter_by_sequence_size(processed_data)          

    # 3. [PROCESS MOLECULAR INPUTS]
    #--------------------------------------------------------------------------  
    tokenization = SMILETokenization(CONFIG) 
    logger.info('Tokenizing SMILE sequences for adsorbate species')   
    processed_data, smile_vocab = tokenization.process_SMILE_sequences(processed_data)    

    # 4. [SPLIT BASED NORMALIZATION AND ENCODING]
    #-------------------------------------------------------------------------- 
    splitter = TrainValidationSplit(CONFIG, processed_data)         
    train_data, validation_data = splitter.split_train_and_validation()
    Z_scores = splitter.get_normalization_parameters(train_data)

    # normalize pressure and uptake series using z-scores precomputed from 
    # the training set and pad sequences to a fixed length
    processed_data = sequencer.PQ_series_normalization(processed_data, Z_scores)  
    processed_data = sequencer.PQ_series_padding(processed_data) 

    normalizer = FeatureNormalizer(CONFIG)
    processed_data = normalizer.normalize_molecular_features(
        processed_data, train_data)

    encoding = AdsorbentEncoder(CONFIG)    
    processed_data, adsorbent_vocab = encoding.encode_adsorbents_by_name(
        processed_data, train_data)     

    # 5. [SAVE PREPROCESSED DATA]
    #--------------------------------------------------------------------------
    # save preprocessed data using data serializer   
    processed_data = sanitizer.isolate_preprocessed_features(processed_data)          
    dataserializer.save_preprocessed_data(
        processed_data, smile_vocab, adsorbent_vocab, Z_scores)












