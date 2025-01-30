import keras

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


###############################################################################
def evaluation_report(model : keras.Model, train_dataset, validation_dataset):
    
    train_eval = model.evaluate(train_dataset, verbose=1)
    validation_eval = model.evaluate(validation_dataset, verbose=1)
    logger.info('Train dataset:')
    logger.info(f'Loss: {train_eval[0]}')    
    logger.info(f'Metric: {train_eval[1]}')  
    logger.info('Test dataset:')
    logger.info(f'Loss: {validation_eval[0]}')    
    logger.info(f'Metric: {validation_eval[1]}')


###############################################################################
def log_training_report(train_data, validation_data, config : dict, 
                        metadata={}, from_checkpoint=False):
    

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
                if sub_key == 'LEARNING_RATE' and from_checkpoint:               
                    sub_value = CONFIG['training']['LEARNING_RATE']
                if isinstance(sub_value, dict):
                    for inner_key, inner_value in sub_value.items():
                        logger.info(f'{key}.{sub_key}.{inner_key}: {inner_value}')
                else:
                    logger.info(f'{key}.{sub_key}: {sub_value}')
        else:
            logger.info(f'{key}: {value}')

    logger.info('--------------------------------------------------------------\n')




        

              
        
        
            
        
