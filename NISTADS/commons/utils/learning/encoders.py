import keras
from keras import layers, activations, Model, optimizers
from transformers import AutoFeatureExtractor, AutoModel
import torch

from NISTADS.commons.utils.learning.transformers import TransformerEncoder
from NISTADS.commons.utils.learning.embeddings import PositionalEmbedding




# [FEED FORWARD]
###############################################################################
@keras.utils.register_keras_serializable(package='encoder', name='StateEncoder')
class StateEncoder(keras.layers.Layer):
    def __init__(self, dense_units, dropout, seed, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(dense_units, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(dense_units, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(dense_units, kernel_initializer='he_uniform')        
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
        x = self.dense2(x) 
        x = activations.relu(x)
        output = self.dropout(x, training=training) 
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(StateEncoder, self).get_config()
        config.update({'dense_units' : self.dense_units,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

