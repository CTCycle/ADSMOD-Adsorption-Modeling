import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from NISTADS.app.utils.data.serializer import DataSerializer
from NISTADS.app.utils.processing.sanitizer import AggregateDatasets
from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.logger import logger   


# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class DataLoaderProcessor():

    def __init__(self, configuration : dict, metadata : dict):
        # load source datasets to obtain the guest and host data references
        # then load the metadata from the processed dataset. At any time, 
        # only a single instance of the processed dataset may exist, therefor
        # the user should be careful about loading a model trained on a different dataset
        self.normalization_config = metadata.get('normalization', {})
        self.series_length = metadata.get('max_measurements', 30)
        self.smile_length = metadata.get('SMILE_sequence_size', 30) 
        self.SMILE_vocab = metadata.get('SMILE_vocabulary', {}) 
        self.adsorbent_vocab = metadata.get('adsorbent_vocabulary', {}) 
        self.serializer = DataSerializer()   
        self.configuration = configuration   
         
  
    # this method is tailored on the inference input dataset, which is provided
    # with pressure already converted to Pascal and fewer columns compared to source data
    #-------------------------------------------------------------------------
    def aggregate_inference_data(self, dataset : pd.DataFrame):
        aggregate_dict = {'temperature' : 'first',                  
                          'adsorbent_name' : 'first',
                          'adsorbate_name' : 'first',                                                
                          'pressure' : lambda x: [float(v) for v in x]}   

        grouped_data = dataset.groupby(by='filename').agg(aggregate_dict).reset_index()
        grouped_data.drop(columns=['filename'], inplace=True)        

        return grouped_data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def add_properties_to_inference_inputs(self, data : pd.DataFrame) -> pd.DataFrame:        
        _, guest_data, host_data = self.serializer.load_adsorption_datasets() 
        aggregator = AggregateDatasets(self.configuration) 
        processed_data = aggregator.join_materials_properties(data, guest_data, host_data) 
        
        return processed_data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def remove_invalid_measurements(self, data : pd.DataFrame) -> pd.DataFrame: 
        data = data[data['temperature'] >= 0] 
        data = data[(data['pressure'] >= 0) & 
                    (data['pressure'] <= self.normalization_config['pressure'])]  
        
        return data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def normalize_from_references(self, data : pd.DataFrame) -> pd.DataFrame: 
        data['temperature'] = data['temperature']/self.normalization_config['temperature']
        data['adsorbate_molecular_weight'] = data['adsorbate_molecular_weight']/self.normalization_config['adsorbate_molecular_weight']
        data['pressure'] = data['pressure'].apply(
            lambda x : [s/self.normalization_config['pressure'] for s in x])

        return data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def encode_SMILE_from_vocabulary(self, smile : str):               
        encoded_tokens = []
        i = 0
        # Sort tokens by descending length to prioritize multi-character tokens
        sorted_tokens = sorted(self.SMILE_vocab.keys(), key=len, reverse=True)
        while i < len(smile):
            matched = False
            for token in sorted_tokens:
                if smile[i:i+len(token)] == token:
                    encoded_tokens.append(self.SMILE_vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                logger.warning(f"SMILE Tokenization error: no valid token found in '{smile}' at position {i}")

        return encoded_tokens    
    
    #-------------------------------------------------------------------------
    def encode_from_references(self, data : pd.DataFrame) -> pd.DataFrame: 
        data['adsorbate_encoded_SMILE'] = data['adsorbate_SMILE'].apply(
            lambda x : self.encode_SMILE_from_vocabulary(x))
        data['encoded_adsorbent'] = data['adsorbent_name'].str.lower().map(self.adsorbent_vocab)

        return data
    
    #-------------------------------------------------------------------------
    def apply_padding(self, data : pd.DataFrame) -> pd.DataFrame: 
        data['pressure'] = pad_sequences(
            data['pressure'], maxlen=self.series_length, value=PAD_VALUE, 
            dtype='float32', padding='post').tolist() 
        
        data['adsorbate_encoded_SMILE'] = pad_sequences(
            data['adsorbate_encoded_SMILE'], maxlen=self.smile_length, value=PAD_VALUE, 
            dtype='float32', padding='post').tolist() 

        return data
    

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class SCADSDataLoader:

    def __init__(self, configuration : dict, metadata : dict, shuffle=True): 
        self.processor = DataLoaderProcessor(configuration, metadata) 
        self.batch_size = configuration.get('batch_size', 32)
        self.inference_batch_size = configuration.get('inference_batch_size', 32)
        self.shuffle_samples = configuration.get('shuffle_size', 1024)
        self.buffer_size = tf.data.AUTOTUNE   
        self.metadata = metadata
        self.configuration = configuration
        self.shuffle = shuffle  
        self.output = 'adsorbed_amount'

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def separate_inputs_and_output(self, data : pd.DataFrame):
        state = np.array(data['temperature'].values, dtype=np.float32)
        chemo = np.array(data['adsorbate_molecular_weight'].values, dtype=np.float32)
        adsorbent = np.array(data['encoded_adsorbent'].values, dtype=np.float32)
        adsorbate = np.vstack(data['adsorbate_encoded_SMILE'].values, dtype=np.float32)
        pressure = np.vstack(data['pressure'].values, dtype=np.float32)
        inputs_dict = {
            'state_input': state, 'chemo_input': chemo, 'adsorbent_input': adsorbent,
            'adsorbate_input': adsorbate, 'pressure_input': pressure}

        # output is reshaped to match the expected shape of the model 
        # (batch size, pressure points, 1)
        output = None
        if self.output in data.columns:
            output = data[self.output]  
            output = np.vstack(output.values, dtype=np.float32) 
    
        return inputs_dict, output  
    
    #-------------------------------------------------------------------------
    def process_inference_inputs(self, data):
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(processed_data)
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        # add padding to pressure and uptake series to match max length
        processed_data = self.processor.apply_padding(processed_data) 
        inference_inputs, _ = self.separate_inputs_and_output(processed_data)

        return inference_inputs       

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def build_training_dataloader(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):           
        batch_size = self.batch_size if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))  
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=self.shuffle_samples) if self.shuffle else dataset

        return dataset
        
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #-------------------------------------------------------------------------
    def build_inference_dataloader(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):           
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        processed_data = self.process_inference_inputs(data)
        inputs, output = self.separate_inputs_and_output(processed_data)        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)        

        return dataset        

    
   




      
    
  
    