import keras
from keras import layers, activations
import torch

from NISTADS.commons.utils.learning.transformers import AddNorm, FeedForward, TransformerEncoder
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
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()  
        self.batch_norm3 = layers.BatchNormalization()       
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)        
        self.seed = seed

    # build method for the custom layer 
    #--------------------------------------------------------------------------
    def build(self, input_shape):        
        super(StateEncoder, self).build(input_shape)

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, x, training=None):
        x = keras.ops.expand_dims(x, axis=1)
        x = self.dense1(x)
        x = activations.elu(x)
        x = self.batch_norm1(x, training=training)
        x = self.dense2(x) 
        x = activations.elu(x)
        x = self.batch_norm2(x, training=training)
        x = self.dense3(x) 
        x = activations.elu(x) 
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
@keras.utils.register_keras_serializable(package='encoder', name='PressureSerierEncoder')
class PressureSerierEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, dropout_rate, seed, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)        
        self.embedding_dims = embedding_dims
        self.dropout_rate = dropout_rate          
        self.attention = layers.MultiHeadAttention(
            num_heads=3, key_dim=self.embedding_dims)
        self.addnorm1 = AddNorm()        
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)   
        self.P_dense = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform') 
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
        mask = keras.ops.cast(mask, torch.float32)       
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, pressure, context, state, mask=None, training=None):
        pressure = keras.ops.expand_dims(pressure, axis=-1)
        pressure = self.P_dense(pressure)
        attention_output = self.attention(
            query=pressure, key=context, value=context, attention_mask=None, 
            training=training)
         
        addnorm = self.addnorm1([pressure, attention_output])

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])          

        mask = self.compute_mask(pressure) if mask is None else mask               
        output = output * mask 
        
        return output
    
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
    
    
# [TRANSFORMER ENCODER]
###############################################################################
@keras.utils.register_keras_serializable(package='Encoders', name='MolecularEncoder')
class MolecularEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, num_heads=4, num_encoders=2, seed=42, **kwargs):
        super(MolecularEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims                
        self.seed = seed
        self.num_heads = num_heads
        self.num_encoders = num_encoders  
        self.encoders = [TransformerEncoder(
            self.embedding_dims, num_heads, self.seed) for _ in range(num_encoders)]
        
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
    def call(self, inputs, mask=None, training=None):         
        encoder_output = inputs    
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)

        output = encoder_output * mask if mask is not None else encoder_output            
        
        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(MolecularEncoder, self).get_config()
        config.update({'num_heads' : self.num_heads,
                       'num_encoders' : self.num_encoders,
                       'embedding_dims': self.embedding_dims,                       
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
    def __init__(self, num_layers, dropout_rate, seed, **kwargs):
        super(QDecoder, self).__init__(**kwargs)        
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate 
        self.seed = seed  
        self.batch_norm = [layers.BatchNormalization() for _ in range(num_layers)]       
        self.dense = [layers.Dense(256, kernel_initializer='he_uniform')
                      for _ in range(num_layers)]        
        
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
        mask = keras.ops.cast(mask, torch.float32)  
        
        return mask

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, P_logits, pressure, mask=None, training=None):

        pressure = keras.ops.expand_dims(pressure, axis=-1)
        mask = self.compute_mask(pressure) if mask is None else mask
        layer = P_logits * mask if mask is not None else P_logits
        
        for dense, bn in zip(self.dense, self.batch_norm):
            layer = dense(layer)
            layer = activations.relu(layer)
            layer = bn(layer, training=training)

        logits = self.Q_output(layer)       
        output = activations.relu(logits)            

        return output
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(QDecoder, self).get_config()
        config.update({'num_layers' : self.num_layers,
                       'dropout_rate' : self.dropout_rate,
                       'seed' : self.seed})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)    

