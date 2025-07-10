from torch import cuda, device
from keras.utils import set_random_seed
from keras.mixed_precision import set_global_policy

from NISTADS.app.src.utils.learning.callbacks import initialize_callbacks_handler
from NISTADS.app.src.utils.data.serializer import ModelSerializer
from NISTADS.app.src.logger import logger


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
###############################################################################
class ModelTraining:    
       
    def __init__(self, configuration):        
        set_random_seed(configuration.get('training_seed', 42))    
        self.configuration = configuration 
    
    #--------------------------------------------------------------------------
    def train_model(self, model, train_data, validation_data, metadata, checkpoint_path, **kwargs):
        total_epochs = self.configuration.get('epochs', 100) 
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, 
            progress_callback=kwargs.get('progress_callback', None), 
            worker=kwargs.get('worker', None))       
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list)

        serializer = ModelSerializer()  
        serializer.save_pretrained_model(model, checkpoint_path)       
        serializer.save_training_configuration(
            checkpoint_path, session, self.configuration, metadata)
        
    #--------------------------------------------------------------------------
    def resume_training(self, model, train_data, validation_data, 
                        checkpoint_path, session=None, **kwargs):
        from_epoch = 0 if not session else session['epochs']     
        total_epochs = from_epoch + self.configuration.get('additional_epochs', 100)            
        # add all callbacks to the callback list
        callbacks_list = initialize_callbacks_handler(
            self.configuration, checkpoint_path, session, 
            kwargs.get('progress_callback', None), kwargs.get('worker', None))       
        
        # run model fit using keras API method.             
        session = model.fit(
            train_data, epochs=total_epochs, validation_data=validation_data, 
            callbacks=callbacks_list, initial_epoch=from_epoch)

        serializer = ModelSerializer()  
        serializer.save_pretrained_model(model, checkpoint_path)       
        serializer.save_training_configuration(
            checkpoint_path, session, self.configuration)

        


      