import keras
from keras import layers, activations, Model, optimizers
import torch

from NISTADS.commons.utils.learning.transformers import AddNorm
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger   



# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='encoder', name='StateEncoder')
class StateEncoder(keras.layers.Layer):
    def __init__(self, dropout, pressure_points, seed, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)        
        self.dropout_rate = dropout
        self.pressure_points = pressure_points
        self.dense1 = layers.Dense(64, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(128, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(256, kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(rate=dropout, seed=seed)
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

        x = keras.ops.expand_dims(x, axis=1)
        tiled_output = keras.ops.tile(x, [1, self.pressure_points, 1])
        output = self.dropout(tiled_output, training=training) 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StateEncoder, self).get_config()
        config.update({'pressure_points' : self.pressure_points,
                       'dropout_rate' : self.dropout_rate,
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
    def __init__(self, units, dropout, seed, mask_values=True, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)        
        self.units = units
        self.dropout_rate = dropout
        self.pressure_kernel = 4
        self.mask_values = mask_values

        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()

        self.conv1 = layers.Conv1D(
            self.units//4, self.pressure_kernel, padding='same', kernel_initializer='he_uniform')
        self.conv2 = layers.Conv1D(
            self.units//2, self.pressure_kernel, padding='same', kernel_initializer='he_uniform')
        self.conv3 = layers.Conv1D(
            self.units, self.pressure_kernel, padding='same', kernel_initializer='he_uniform')

        self.dense1 = layers.Dense(self.units//4, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(self.units//2, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(self.units, kernel_initializer='he_uniform')        
        self.dropout = layers.Dropout(rate=dropout, seed=seed)
        self.seed = seed

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(PressureSerierEncoder, self).build(input_shape)

    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, -1)        
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, x, training=None):
        
        inputs = keras.ops.expand_dims(x, axis=-1)

        series = self.conv1(inputs)
        series = activations.relu(series)
        series = self.dense1(series)
        series = activations.relu(series)
        addnorm = self.addnorm1([inputs, series])

        series = self.conv2(addnorm)
        series = activations.relu(series)
        series = self.dense2(series)
        series = activations.relu(series)
        addnorm = self.addnorm2([inputs, series])

        series = self.conv3(addnorm)
        series = activations.relu(series)
        series = self.dense3(series)
        series = activations.relu(series)
        output = self.addnorm3([inputs, series])        

        if self.mask_values:
            mask = keras.ops.not_equal(inputs, -1)            
            output *= mask 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PressureSerierEncoder, self).get_config()
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
@keras.utils.register_keras_serializable(package='decoder', name='QDecoder')
class QDecoder(keras.layers.Layer):
    def __init__(self, embedding_dims, series_lenght, dropout, depth=3, seed=42, **kwargs):
        super(QDecoder, self).__init__(**kwargs)        
        self.dropout_rate = dropout
        self.series_lenght = series_lenght
        self.embedding_dims = embedding_dims
        self.depth = depth
        self.dense_layers = [layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')
                             for x in range(depth)]
        self.addnorm_layers = [AddNorm() for x in range(depth)]
        
        self.addnorm = AddNorm()
        self.concat = layers.Concatenate(axis=-1)
        
        self.pool1 = layers.GlobalMaxPooling1D()
        self.pool2 = layers.GlobalMaxPooling1D()
        self.dense1 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform') 
        self.dense2 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')
        self.dense3 = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')  

        self.dropout = layers.Dropout(rate=dropout, seed=seed)
        self.Q1_dense = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')
        self.Q2_dense = layers.Dense(self.series_lenght, kernel_initializer='he_uniform') 
        self.seed = seed

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(QDecoder, self).build(input_shape)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, molecules, pressure, states, training=None):

        molecules = self.dense1(molecules)
        molecules = activations.relu(molecules)
        pressure = self.dense2(pressure)
        pressure = activations.relu(pressure)
        states = self.dense3(states)
        states = activations.relu(states)

        addnorm = self.addnorm([pressure, states])
        P_pooling = self.pool1(addnorm)
        mol_pooling = self.pool2(molecules)

        concat = self.concat([P_pooling, mol_pooling])
        uptake_scores = self.Q1_dense(concat)

        output = uptake_scores
        for dense, an in zip(self.dense_layers, self.addnorm_layers):
            output = dense(output)
            output = activations.relu(output)
            output = an([output, uptake_scores]) 

        output = self.Q2_dense(output)    
        output = activations.relu(output)
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(QDecoder, self).get_config()
        config.update({'embedding_dims' : self.embedding_dims,
                       'series_lenght' : self.series_lenght,
                       'depth' : self.depth,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)