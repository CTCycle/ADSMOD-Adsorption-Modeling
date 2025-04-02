# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.learning.training import ModelTraining

from NISTADS.commons.utils.inference.predictor import AdsorptionPredictions
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD PRETRAINED MODEL]
    #--------------------------------------------------------------------------    
    # selected and load the pretrained model, then print the summary     
    logger.info('Loading specific checkpoint from pretrained models') 
    modelserializer = ModelSerializer()      
    model, configuration, history, checkpoint_path = modelserializer.select_and_load_checkpoint()    
    model.summary(expand_nested=True)  
    
    # setting device for training    
    trainer = ModelTraining(configuration)    
    trainer.set_device()

    # 2. [LOAD AND PROCEPROCESS DATA]
    #--------------------------------------------------------------------------  
    dataserializer = DataSerializer(configuration)    
    inference_data = dataserializer.load_inference_data()       

    # 3. [PREDICT ADSORPTION] 
    #--------------------------------------------------------------------------
    logger.info('Preprocessing inference input data according to model configurations')
    predictor = AdsorptionPredictions(model, configuration, checkpoint_path)
    predictions = predictor.predict_adsorption_isotherm(inference_data)



