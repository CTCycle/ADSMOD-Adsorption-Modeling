import pandas as pd
import numpy as np
import tensorflow as tf

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger   

        

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TensorDatasetBuilder:

    def __init__(self, configuration):                
        self.configuration = configuration
        self.features = ['temperature', 'pressure', 'encoded_adsorbent',
                         'adsorbate_encoded_SMILE', 'adsorbate_molecular_weight']
        self.output = 'adsorbed_amount'

    #--------------------------------------------------------------------------
    def _define_IO_features(self, data : pd.DataFrame):
        inputs = {'state_input': np.column_stack([data['temperature'].values, data['adsorbate_molecular_weight'].values]),
                  'adsorbent_input': data['encoded_adsorbent'].values,
                  'adsorbate_input': np.vstack(data['adsorbate_encoded_SMILE'].values),
                  'pressure_input': np.vstack(data['pressure'].values)}

        # the outout is reshaped to match the expected shape of the model (batch size, pressure points, 1)  
        output = np.reshape(np.vstack(data[self.output].values), newshape=(data.shape[0], -1, 1))   

        return inputs, output

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def build_tensor_dataset(self, data : pd.DataFrame, batch_size, buffer_size=tf.data.AUTOTUNE):    
        data = data.dropna(subset=self.features.append(self.output))
        num_samples = data.shape[0]                
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size
        inputs, output = self._define_IO_features(data)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))                      
               
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples)       

        return dataset
        
    #--------------------------------------------------------------------------
    def build_model_dataloader(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                               batch_size=None):            
        
        train_dataset = self.build_tensor_dataset(train_data, batch_size)
        validation_dataset = self.build_tensor_dataset(validation_data, batch_size)         

        return train_dataset, validation_dataset






   


    