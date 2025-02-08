import keras
from keras import layers, activations

from NISTADS.commons.utils.learning.transformers import AddNorm
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger   



# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='encoder', name='StateEncoder')
class StateEncoder(keras.layers.Layer):
    def __init__(self, dropout, seed, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)        
        self.dropout_rate = dropout        
        self.dense1 = layers.Dense(32, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(48, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(64, kernel_initializer='he_uniform')        
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
    def __init__(self, units, dropout, seed, mask_values=True, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)        
        self.units = units
        self.dropout_rate = dropout        
        self.mask_values = mask_values
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
    def call(self, x, context, training=None):
        series_length = keras.ops.shape(x)[1]
        expanded_context = keras.ops.expand_dims(context, axis=1)
        tiled_context = keras.ops.tile(expanded_context, [1, series_length, 1])
        context = self.context_dense(tiled_context)
        context = activations.relu(context)

        inputs = keras.ops.expand_dims(x, axis=-1)
        added_context = self.context_addnorm([inputs, context])       
        
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

        if self.mask_values:
            mask = keras.ops.not_equal(inputs, -1)            
            output *= mask 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PressureSerierEncoder, self).get_config()
        config.update({'units' : self.units,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed,
                       'mask_values' : self.mask_values})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)    

