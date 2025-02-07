import torch
import keras
from keras import layers 

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger
      

# [POSITIONAL EMBEDDING]
###############################################################################
@keras.utils.register_keras_serializable(package='CustomLayers', name='MolecularEmbedding')
class MolecularEmbedding(keras.layers.Layer):
    def __init__(self, smile_vocab_size, ads_vocab_size, embedding_dims, sequence_length, 
                 mask_values=True, **kwargs):
        super(MolecularEmbedding, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.sequence_length = sequence_length 
        self.smile_vocab_size = smile_vocab_size
        self.ads_vocab_size = ads_vocab_size
        self.mask_values = mask_values
        self.ads_embeddings = layers.Embedding(
            input_dim=ads_vocab_size, output_dim=self.embedding_dims, mask_zero=mask_values)
        self.smile_embeddings = layers.Embedding(
            input_dim=smile_vocab_size, output_dim=self.embedding_dims, mask_zero=mask_values)
        self.position_embeddings = layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.embedding_dims)
        self.embedding_scale = keras.ops.sqrt(keras.ops.cast(self.embedding_dims, torch.float32))       
    
    # implement positional embedding through call method  
    #--------------------------------------------------------------------------    
    def call(self, smiles, adsorbent, training=False):

        ads_embeddings = self.ads_embeddings(adsorbent)          
        ads_embeddings = keras.ops.expand_dims(ads_embeddings, axis=1)
        ads_repeated_embeddings = keras.ops.tile(
            ads_embeddings, [1, self.sequence_length, 1])
        ads_embeddings *= self.embedding_scale

        length = keras.ops.shape(smiles)[-1] 
        positions = keras.ops.arange(start=0, stop=length, step=1)
        positions = keras.ops.cast(positions, dtype=smiles.dtype)        
        embedded_smile = self.smile_embeddings(smiles)
        embedded_smile *= self.embedding_scale        
        embedded_positions = self.position_embeddings(positions)        
        full_embedding = embedded_smile + embedded_positions + ads_repeated_embeddings
        
        if self.mask_values:
            mask = keras.ops.not_equal(smiles, 0)
            mask = keras.ops.expand_dims(keras.ops.cast(mask, torch.float32), axis=-1)
            full_embedding *= mask

        return full_embedding
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):        
        mask = keras.ops.not_equal(inputs, 0)        
        
        return mask
    
    # serialize layer for saving  
    #--------------------------------------------------------------------------
    def get_config(self):
        config = super(MolecularEmbedding, self).get_config()
        config.update({'smile_vocabulary_size': self.smile_vocab_size,
                       'ads_vocabulary_size': self.ads_vocab_size,
                       'sequence_length': self.sequence_length,                       
                       'embedding_dims': self.embedding_dims,                       
                       'mask_values': self.mask_values})
        return config

    # deserialization method 
    #--------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

