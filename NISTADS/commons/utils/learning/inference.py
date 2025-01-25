import os
import numpy as np
import keras
from tqdm import tqdm

from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.constants import ENCODED_PATH
from FEXT.commons.logger import logger


# [INFERENCE]
###############################################################################
class ImageEncoding:
    
    def __init__(self, model : keras.Model, configuration):
       
        keras.utils.set_random_seed(configuration["SEED"])  
        self.dataserializer = DataSerializer(configuration)  
        self.img_shape = configuration["model"]["IMG_SHAPE"]
        self.configuration = configuration
        self.model = model 

        # isolate the encoder submodel from the autoencoder model, and use it for inference             
        encoder_output = model.get_layer('compression_layer').output 
        self.encoder_model = keras.Model(inputs=model.input, outputs=encoder_output)              

    #--------------------------------------------------------------------------
    def encoder_images(self, images_paths):
        
        features = {}
        for pt in tqdm(images_paths):
            try:
                image = self.dataserializer.load_image(pt)
                image = keras.ops.expand_dims(image, axis=0)
                extracted_features = self.encoder_model.predict(image, verbose=0)
                features[pt] = extracted_features
            except Exception as e:
                features[pt] = f'Could not extract features: {str(e)}'
                logger.error(f'Could not extract features from image at {pt}: {str(e)}')

        # combine extracted features with images name and save them in numpy arrays    
        structured_data = np.array([(image, features[image]) for image in features], dtype=object)
        file_loc = os.path.join(ENCODED_PATH, 'extracted_features.npy')
        np.save(file_loc, structured_data)

        logger.debug(f'Extracted img features saved as numpy array at {file_loc}')

        return features





