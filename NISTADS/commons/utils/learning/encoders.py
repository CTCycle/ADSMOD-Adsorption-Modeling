import keras
from keras import layers, activations
import torch
torch.autograd.set_detect_anomaly(True)

from NISTADS.commons.utils.learning.layers import AddNorm
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger   



# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='encoder', name='StateEncoder')
class StateEncoder(keras.layers.Layer):
    def __init__(self, dropout_rate, seed, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)        
        self.dropout_rate = dropout_rate        
        self.dense1 = layers.Dense(32, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(48, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(64, kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)        
        self.seed = seed

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(StateEncoder, self).build(input_shape)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, x, training=None):
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x) 
        x = activations.relu(x)
        x = self.dense3(x) 
        x = activations.relu(x)        
        output = self.dropout(x, training=training) 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StateEncoder, self).get_config()
        config.update({'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='encoder', name='PressureSerierEncoder')
class PressureSerierEncoder(keras.layers.Layer):
    def __init__(self, units, dropout_rate, seed, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)        
        self.units = units
        self.dropout_rate = dropout_rate       
        self.context_addnorm = AddNorm()
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.conv1 = layers.Conv1D(self.units, 4, padding='same', kernel_initializer='he_uniform')
        self.conv2 = layers.Conv1D(self.units, 2, padding='same', kernel_initializer='he_uniform')
        self.conv3 = layers.Conv1D(self.units, 1, padding='same', kernel_initializer='he_uniform')
        self.context_dense = layers.Dense(self.units, kernel_initializer='he_uniform')
        self.dense1 = layers.Dense(self.units, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(self.units, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(self.units, kernel_initializer='he_uniform')
        self.outdense = layers.Dense(1, kernel_initializer='he_uniform') 
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.seed = seed
        self.supports_masking = True

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(PressureSerierEncoder, self).build(input_shape)

    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, -1)  
        mask = keras.ops.expand_dims(keras.ops.cast(mask, torch.float32), axis=-1)       
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, pressure, context, mask=None, training=None):
        series_length = keras.ops.shape(pressure)[1]
        expanded_context = keras.ops.expand_dims(context, axis=1)
        tiled_context = keras.ops.tile(expanded_context, [1, series_length, 1])        

        inputs = keras.ops.expand_dims(pressure, axis=-1)        
        added_context = self.context_addnorm([inputs, tiled_context])
        added_context = self.context_dense(added_context)                      
        
        series = self.conv1(added_context)
        series = activations.relu(series)
        series = self.dense1(series)
        series = activations.relu(series)
        addnorm = self.addnorm1([added_context, series])
        series = self.conv2(addnorm)
        series = activations.relu(series)
        series = self.dense2(series)
        series = activations.relu(series)
        addnorm = self.addnorm2([added_context, series])
        series = self.conv3(addnorm)
        series = activations.relu(series)
        series = self.dense3(series)
        series = activations.relu(series)
        output = self.addnorm3([added_context, series])   

        mask = self.compute_mask(pressure) if mask is None else mask               
        output = output * mask 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PressureSerierEncoder, self).get_config()
        config.update({'units' : self.units,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)  
    
    
# [TRANSFORMER ENCODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='MolecularEncoder')
class MolecularEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, seed, **kwargs):
        super(MolecularEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims                
        self.seed = seed  
        self.dense1 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')  
        self.flatten = layers.Flatten()      
        self.supports_masking = True       

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(MolecularEncoder, self).build(input_shape)    

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, smiles, mask=None, training=None):
        
        if mask is not None:       
            mask = keras.ops.expand_dims(mask, axis=-1)           

        smiles = self.dense1(smiles)
        smiles = activations.relu(smiles)      
        smiles = self.dense2(smiles)
        smiles = activations.relu(smiles)
        smiles = self.dense1(smiles)
        smiles = activations.relu(smiles) 

        output = smiles * mask if mask is not None else smiles
        output = self.flatten(output)      
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(MolecularEncoder, self).get_config()
        config.update({'embedding_dims': self.embedding_dims,                       
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
      

# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='decoder', name='QDecoder')
class QDecoder(keras.layers.Layer):
    def __init__(self, units, dropout_rate, seed, **kwargs):
        super(QDecoder, self).__init__(**kwargs)        
        self.units = units
        self.dropout_rate = dropout_rate       
        self.Q1 = layers.Dense(units, kernel_initializer='he_uniform')        
        self.Q2 = layers.Dense(units//2, kernel_initializer='he_uniform')      
        self.Q3 = layers.Dense(units//4, kernel_initializer='he_uniform')
        self.Q_output = layers.Dense(1, kernel_initializer='he_uniform')
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.seed = seed
        self.supports_masking = True

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(QDecoder, self).build(input_shape)

    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, -1)   
        mask = keras.ops.expand_dims(keras.ops.cast(mask, torch.float32), axis=-1)     
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, pressure, mask=None, training=None):
        logits = self.Q1(pressure)
        logits = activations.relu(logits)
        logits = self.Q2(logits)
        logits = activations.relu(logits)
        logits = self.Q3(logits)
        logits = activations.relu(logits)
        logits = self.Q_output(logits)       
        output = activations.relu(logits)

        mask = self.compute_mask(pressure) if mask is None else mask
        output = output * mask

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(QDecoder, self).get_config()
        config.update({'units' : self.units,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)    

