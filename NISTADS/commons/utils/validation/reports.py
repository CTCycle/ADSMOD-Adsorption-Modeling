import keras

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


###############################################################################
def evaluation_report(model : keras.Model, train_dataset, validation_dataset):    
    training = model.evaluate(train_dataset, verbose=1)
    validation = model.evaluate(validation_dataset, verbose=1)
    logger.info(
        f'Training loss {training[0]:.3f} - Training R square {training[1]:.3f}')    
    logger.info(
        f'Validation loss {validation[0]:.3f} - Validation R square {validation[1]:.3f}')


###############################################################################
def log_training_report(train_data, validation_data, config : dict, metadata={}):
    smile_vocab_size = metadata.get('SMILE_vocabulary_size', 0)
    ads_vocab_size = metadata.get('adsorbent_vocabulary_size', 0)
    logger.info('--------------------------------------------------------------')
    logger.info('NISTADS training report')
    logger.info('--------------------------------------------------------------')    
    logger.info(f'Number of train samples:       {len(train_data)}')
    logger.info(f'Number of validation samples:  {len(validation_data)}')
    logger.info(f'SMILE vocabulary size:         {smile_vocab_size}')
    logger.info(f'Adsorbents vocabulary size:    {ads_vocab_size}')    
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key == 'ADDITIONAL_EPOCHS':
                    sub_value = CONFIG['training']['ADDITIONAL_EPOCHS']                
                if isinstance(sub_value, dict):
                    for inner_key, inner_value in sub_value.items():
                        logger.info(f'{key}.{sub_key}.{inner_key}: {inner_value}')
                else:
                    logger.info(f'{key}.{sub_key}: {sub_value}')
        else:
            logger.info(f'{key}: {value}')

    logger.info('--------------------------------------------------------------\n')



