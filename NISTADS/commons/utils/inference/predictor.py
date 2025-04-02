import os
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm

from NISTADS.commons.utils.data.loader import InferenceDataLoader

from NISTADS.commons.constants import CONFIG, INFERENCE_PATH
from NISTADS.commons.logger import logger


# [INFERENCE]
###############################################################################
class AdsorptionPredictions:
    
    def __init__(self, model : keras.Model, configuration : dict, checkpoint_path : str):        
        keras.utils.set_random_seed(configuration["SEED"])  
        # initialize the inference data loader that prepares the data 
        # for using the model in inference mode (predictions)
        self.dataloader = InferenceDataLoader(configuration)
                          
        self.checkpoint_name = os.path.basename(checkpoint_path)        
        self.configuration = configuration
        self.model = model 

    #--------------------------------------------------------------------------
    def predict_adsorption_isotherm(self, data : pd.DataFrame):
        processed_inputs = self.dataloader.preprocess_inference_inputs(data)

        pass
        
        
        # for pt in tqdm(images_paths, desc='Encoding images', total=len(images_paths)):
        #     image_name = os.path.basename(pt)
        #     try:
        #         image = self.dataloader.load_image_as_array(pt)
        #         image = np.expand_dims(image, axis=0)
        #         extracted_features = self.encoder_model.predict(image, verbose=0)
        #         features[pt] = extracted_features
        #     except Exception as e:
        #         features[pt] = f'Error during encoding: {str(e)}'
        #         logger.error(f'Could not encode image {image_name}: {str(e)}')

        # # combine extracted features with images name and save them in numpy arrays    
        # structured_data = np.array(
        #     [(image, features[image]) for image in features], dtype=object)
        # file_loc = os.path.join(
        #     INFERENCE_PATH, f'encoded_images_{self.checkpoint_name}.npy')
        # np.save(file_loc, structured_data)
        
        #return features

    
    
  
        





