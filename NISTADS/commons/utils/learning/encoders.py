import keras
from keras import layers, activations
import torch

from NISTADS.commons.utils.learning.transformers import AddNorm, FeedForward
from NISTADS.commons.constants import CONFIG, PAD_VALUE
from NISTADS.commons.logger import logger   


# [STATE ENCODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='StateEncoder')
class StateEncoder(keras.layers.Layer):
    def __init__(self, dropout_rate, seed, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)        
        self.dropout_rate = dropout_rate        
        self.dense1 = layers.Dense(64, kernel_initializer='he_uniform')
        self.dense2 = layers.Dense(96, kernel_initializer='he_uniform') 
        self.dense3 = layers.Dense(128, kernel_initializer='he_uniform') 
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()  
        self.batch_norm3 = layers.BatchNormalization()       
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)        
        self.seed = seed

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(StateEncoder, self).build(input_shape)
      
    #--------------------------------------------------------------------------    
    def call(self, x, training=None):
        x = keras.ops.expand_dims(x, axis=-1)        
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.batch_norm1(x, training=training)
        x = self.dense2(x) 
        x = activations.relu(x)
        x = self.batch_norm2(x, training=training)
        x = self.dense3(x) 
        x = activations.relu(x) 
        x = self.batch_norm3(x, training=training)       
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
@keras.utils.register_keras_serializable(package='Encoders', name='PressureSerierEncoder')
class PressureSerierEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, dropout_rate, num_heads, seed, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)        
        self.embedding_dims = embedding_dims
        self.dropout_rate = dropout_rate  
        self.num_heads = num_heads  
        self.seed = seed       
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dims)
        self.addnorm1 = AddNorm()        
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()        
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed) 
        self.ffn2 = FeedForward(self.embedding_dims, 0.3, seed)   
        self.P_dense = layers.Dense(
            self.embedding_dims, kernel_initializer='he_uniform') 
        self.state_dense = layers.Dense(
            self.embedding_dims, kernel_initializer='he_uniform') 
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        
        self.supports_masking = True

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(PressureSerierEncoder, self).build(input_shape)    

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, pressure, context, state, key_mask=None, training=None):
        # compute query mask as the masked pressure series
        query_mask = self.compute_mask(pressure)
        # project the pressure series into the embedding space using 
        pressure = keras.ops.expand_dims(pressure, axis=-1)        
        pressure = self.P_dense(pressure)         

        # cross-attention between the pressure series and the molecular context
        # the latter being generated from self-attention of the enriched SMILE sequences
        attention_output = self.attention(
            query=pressure, key=context, value=context, 
            query_mask=query_mask, key_mask=key_mask, value_mask=key_mask,
            training=training)
         
        addnorm = self.addnorm1([pressure, attention_output])

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        ffn_out = self.addnorm2([addnorm, ffn_out])  

        # ideally, higher temperature should decrease the adsorbed amount, therefor
        # temperature is used to compute an inverse scaling factor for the output        
        state = self.state_dense(state)
        state = activations.relu(state)        
        temperature_scaling = keras.ops.expand_dims(keras.ops.exp(-state), axis=1)                
        output = ffn_out * temperature_scaling
        
        return output
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, PAD_VALUE)        
        mask = keras.ops.cast(mask, torch.float32)       
        
        return mask
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(PressureSerierEncoder, self).get_config()
        config.update({'embedding_dims' : self.embedding_dims,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)     
    


# [UPTAKE DECODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Decoders', name='QDecoder')
class QDecoder(keras.layers.Layer):
    def __init__(self, num_layers, dropout_rate, embedding_dims, seed, **kwargs):
        super(QDecoder, self).__init__(**kwargs)        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embedding_dims = embedding_dims 
        self.seed = seed  
        self.batch_norm = [layers.BatchNormalization() for _ in range(num_layers)]
        self.dropouts = [layers.Dropout(rate=dropout_rate, seed=seed) for _ in range(num_layers)]       
        self.dense = [layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')
                      for _ in range(num_layers)]        
        
        self.Q_output = layers.Dense(1, kernel_initializer='he_uniform')        
        self.seed = seed
        self.supports_masking = True

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(QDecoder, self).build(input_shape)

    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, PAD_VALUE) 
        mask = keras.ops.expand_dims(mask, axis=-1)  
        mask = keras.ops.cast(mask, torch.float32)  
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, P_logits, pressure, mask=None, training=None):        
        mask = self.compute_mask(pressure) if mask is None else mask
        layer = P_logits * mask if mask is not None else P_logits        
        for dense, bn, dp in zip(self.dense, self.batch_norm, self.dropouts):
            layer = dense(layer)
            layer = activations.relu(layer)
            layer = bn(layer, training=training)
            layer = dp(layer, training=training)
        
        logits = self.Q_output(layer)         
        output = keras.ops.squeeze(logits, axis=-1)           

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(QDecoder, self).get_config()
        config.update({'num_layers' : self.num_layers,
                       'dropout_rate' : self.dropout_rate,
                       'embedding_dims' : self.embedding_dims,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)    

