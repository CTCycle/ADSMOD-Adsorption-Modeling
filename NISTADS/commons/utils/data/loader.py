import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.api.preprocessing import sequence


from NISTADS.commons.utils.data.serializer import DataSerializer
from NISTADS.commons.utils.data.process.aggregation import AggregateDatasets
from NISTADS.commons.constants import CONFIG, PAD_VALUE
from NISTADS.commons.logger import logger   


# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
class TrainingDataLoaderProcessor():

    def __init__(self, configuration):        
        self.configuration = configuration   
 
    # currently used as placeholder returning same input and output, additional
    # features may be implemented for image augmentation etc
    #--------------------------------------------------------------------------
    def process_data(self, inputs, output):       
        
        return inputs, output    



# [INFERENCE]
###############################################################################
class InferenceDataLoaderProcessor:
    
    def __init__(self, configuration : dict):        
        keras.utils.set_random_seed(configuration["SEED"])        
        self.configuration = configuration              
        self.aggregator = AggregateDatasets(configuration)        

        # load source datasets to obtain the guest and host data references
        # then load the metadata from the processed dataset. At any time, 
        # only a single instance of the processed dataset may exist, therefor
        # the user should be careful about loading a model trained on a different
        # version of said dataset
        dataserializer = DataSerializer(configuration)  
        _, self.guest_data, self.host_data = dataserializer.load_datasets() 
        _, self.metadata, self.smile_vocab, self.ads_vocab = dataserializer.load_preprocessed_data() 

        self.pressure_padding = int(self.metadata['dataset']['MAX_PQ_POINTS'])
        self.smile_padding = int(self.metadata['dataset']['SMILE_PADDING'])  
  
    # get processing metadata as fetched from the processed dataset metadata in resources
    #--------------------------------------------------------------------------
    def get_processing_metadata(self):   
        return self.metadata, self.smile_vocab, self.ads_vocab       

    # this method is tailored on the inference input dataset, which is provided
    # with pressure already converted to Pascal and fewer columns compared to source data
    #--------------------------------------------------------------------------
    def aggregate_inference_data(self, dataset : pd.DataFrame):
        aggregate_dict = {'temperature' : 'first',                  
                          'adsorbent_name' : 'first',
                          'adsorbate_name' : 'first',                                                
                          'pressure' : lambda x: [float(v) for v in x]}   

        grouped_data = dataset.groupby(by='filename').agg(aggregate_dict).reset_index()
        grouped_data.drop(columns=['filename'], inplace=True)        

        return grouped_data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def add_properties_to_inference_inputs(self, data : pd.DataFrame):
        processed_data = self.aggregator.join_materials_properties(
            data, self.guest_data, self.host_data) 
        
        return processed_data

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def remove_invalid_measurements(self, data : pd.DataFrame):
        data = data[data['temperature'] >= 0] 
        data = data[(data['pressure'] >= 0) & 
                    (data['pressure'] <= self.metadata["Pressure_max"])]  
        
        return data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def normalize_from_references(self, data : pd.DataFrame):
        data['temperature'] = data['temperature']/self.metadata['Temperature_max']
        data['adsorbate_molecular_weight'] = data['adsorbate_molecular_weight']/self.metadata['Molecular_weight_max']
        data['pressure'] = data['pressure'].apply(
            lambda x : [s/self.metadata['Pressure_max'] for s in x])

        return data
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def encode_SMILE_from_vocabulary(self, smile):
        encoded_tokens = []
        i = 0
        # Sort tokens by descending length to prioritize multi-character tokens
        sorted_tokens = sorted(self.smile_vocab.keys(), key=len, reverse=True)
        while i < len(smile):
            matched = False
            for token in sorted_tokens:
                if smile[i:i+len(token)] == token:
                    encoded_tokens.append(self.smile_vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                logger.warning(f"SMILE Tokenization error: no valid token found in '{smile}' at position {i}")

        return encoded_tokens    
    
    #--------------------------------------------------------------------------
    def encode_from_references(self, data : pd.DataFrame):
        data['adsorbate_encoded_SMILE'] = data['adsorbate_SMILE'].apply(
            lambda x : self.encode_SMILE_from_vocabulary(x))
        data['encoded_adsorbent'] = data['adsorbent_name'].str.lower().map(self.ads_vocab)

        return data
    
    #--------------------------------------------------------------------------
    def apply_padding(self, data : pd.DataFrame):
        pressure_padding = int(self.metadata['dataset']['MAX_PQ_POINTS'])
        smile_padding = int(self.metadata['dataset']['SMILE_PADDING'])

        data['pressure'] = sequence.pad_sequences(
            data['pressure'], maxlen=pressure_padding, value=PAD_VALUE, 
            dtype='float32', padding='post').tolist() 
        
        data['adsorbate_encoded_SMILE'] = sequence.pad_sequences(
            data['adsorbate_encoded_SMILE'], maxlen=smile_padding, value=PAD_VALUE, 
            dtype='float32', padding='post').tolist() 

        return data
   
   

# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class TrainingDataLoader:

    def __init__(self, configuration, shuffle=True): 
        self.generator = TrainingDataLoaderProcessor(configuration)                
        self.configuration = configuration
        self.shuffle = shuffle
        self.output = 'adsorbed_amount'        

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def separate_inputs_and_output(self, data : pd.DataFrame):
        data = data.dropna(how='any')                 
        state = np.array(data['temperature'].values, dtype=np.float32)
        chemo = np.array(data['adsorbate_molecular_weight'].values, dtype=np.float32)
        adsorbent = np.array(data['encoded_adsorbent'].values, dtype=np.float32)
        adsorbate = np.vstack(data['adsorbate_encoded_SMILE'].values, dtype=np.float32)
        pressure = np.vstack(data['pressure'].values, dtype=np.float32)

        inputs_dict = {'state_input': state,
                       'chemo_input': chemo,
                       'adsorbent_input': adsorbent,
                       'adsorbate_input': adsorbate,
                       'pressure_input': pressure}

        # output is reshaped to match the expected shape of the model 
        # (batch size, pressure points, 1)
        output = data[self.output]  
        output = np.vstack(output.values, dtype=np.float32)        
  
        return inputs_dict, output   

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data : pd.DataFrame, batch_size=None, buffer_size=tf.data.AUTOTUNE):      
        num_samples = data.shape[0]                
        batch_size = self.configuration["training"]["BATCH_SIZE"] if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))  
        dataset = dataset.map(self.generator.process_data, num_parallel_calls=buffer_size)                
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=num_samples) if self.shuffle else dataset       

        return dataset
        
    #--------------------------------------------------------------------------
    def build_training_dataloader(self, train_data : pd.DataFrame, validation_data : pd.DataFrame, 
                                  batch_size=None):       
        train_dataset = self.compose_tensor_dataset(train_data, batch_size)
        validation_dataset = self.compose_tensor_dataset(validation_data, batch_size)         

        return train_dataset, validation_dataset
    



# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
class InferenceDataLoader:

    def __init__(self, configuration):        
        self.batch_size = configuration['validation']["BATCH_SIZE"] 
        self.processor = InferenceDataLoaderProcessor(configuration)
        self.configuration = configuration
        self.output = 'adsorbed_amount'        
               

    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def separate_inputs(self, data : pd.DataFrame):  
        data = data.dropna(how='any')               
        state = np.array(data['temperature'].values, dtype=np.float32)
        chemo = np.array(data['adsorbate_molecular_weight'].values, dtype=np.float32)
        adsorbent = np.array(data['encoded_adsorbent'].values, dtype=np.float32)
        adsorbate = np.vstack(data['adsorbate_encoded_SMILE'].values, dtype=np.float32)
        pressure = np.vstack(data['pressure'].values, dtype=np.float32)
        inputs_dict = {'state_input': state,
                       'chemo_input': chemo,
                       'adsorbent_input': adsorbent,
                       'adsorbate_input': adsorbate,
                       'pressure_input': pressure}      
  
        return inputs_dict
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def separate_inputs_and_output(self, data : pd.DataFrame):
        data = data.dropna(how='any')            
        state = np.array(data['temperature'].values, dtype=np.float32)
        chemo = np.array(data['adsorbate_molecular_weight'].values, dtype=np.float32)
        adsorbent = np.array(data['encoded_adsorbent'].values, dtype=np.float32)
        adsorbate = np.vstack(data['adsorbate_encoded_SMILE'].values, dtype=np.float32)
        pressure = np.vstack(data['pressure'].values, dtype=np.float32)
        inputs_dict = {'state_input': state,
                       'chemo_input': chemo,
                       'adsorbent_input': adsorbent,
                       'adsorbate_input': adsorbate,
                       'pressure_input': pressure}

        # output is reshaped to match the expected shape of the model 
        # (batch size, pressure points, 1)
        output = data[self.output]  
        output = np.vstack(output.values, dtype=np.float32)        
  
        return inputs_dict, output   

    #--------------------------------------------------------------------------
    def preprocess_inference_inputs(self, data : pd.DataFrame):
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(processed_data)
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        # add padding to pressure and uptake series to match max length
        processed_data = self.processor.apply_padding(processed_data) 
        inference_inputs = self.separate_inputs(processed_data)

        return inference_inputs   
    
    #--------------------------------------------------------------------------
    def postprocess_inference_output(self, inputs : dict, predictions : np.array):
        
        metadata, smile_vocab, ads_vocab = self.processor.get_processing_metadata()
        # reshape predictions from (samples, measurements, 1) to (samples, measurements)
        predictions = np.squeeze(predictions, axis=-1)
        
        return predictions
    
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #--------------------------------------------------------------------------
    def compose_tensor_dataset(self, data : pd.DataFrame, batch_size=None, buffer_size=tf.data.AUTOTUNE):                             
        batch_size = self.configuration["validation"]["BATCH_SIZE"] if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)       
        dataset = tf.data.Dataset.from_tensor_slices((inputs, output))    
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
       
        return dataset      
    
    #--------------------------------------------------------------------------
    def build_inference_dataloader(self, train_data, batch_size=None):       
        dataset = self.compose_tensor_dataset(train_data, batch_size)             

        return dataset  
      
    
  
    