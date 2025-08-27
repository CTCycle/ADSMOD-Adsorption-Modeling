import keras
from keras import layers 

from NISTADS.app.constants import PAD_VALUE
      

# [POSITIONAL EMBEDDING]
###############################################################################
@keras.saving.register_keras_serializable(package='Embeddings', name='MolecularEmbedding')
class MolecularEmbedding(layers.Layer):
    def __init__(self, smile_vocab_size, ads_vocab_size, embedding_dims, sequence_length, 
                 mask_values=True, **kwargs):
        super(MolecularEmbedding, self).__init__(**kwargs)
        self.smile_vocab_size = smile_vocab_size
        self.ads_vocab_size = ads_vocab_size
        self.embedding_dims = embedding_dims
        self.sequence_length = sequence_length         
        self.mask_values = mask_values        
        
        self.adsorbent_embeddings = layers.Embedding(
            mask_zero=False, input_dim=self.ads_vocab_size, 
            output_dim=self.embedding_dims)
        self.chemo_embeddings = layers.Dense(
            self.embedding_dims, kernel_initializer='he_uniform')
        self.smile_embeddings = layers.Embedding(
            mask_zero=False, input_dim=smile_vocab_size, 
            output_dim=self.embedding_dims)
        self.position_embeddings = layers.Embedding(
            mask_zero=False, input_dim=self.sequence_length, 
            output_dim=self.embedding_dims)
        self.embedding_scale = keras.ops.sqrt(
            keras.ops.cast(self.embedding_dims, keras.config.floatx()))       
    
    # implement positional embedding through call method  
    #-------------------------------------------------------------------------    
    def call(self, smiles, adsorbent, chemometrics, training=False):
        # compute the positional embeddings for the SMILE sequences
        # and add it to the SMILE embeddings upon casting to same dtype
        # SMILE embeddings are scaled by the square root of the embedding dimensions
        length = keras.ops.shape(smiles)[-1] 
        positions = keras.ops.arange(start=0, stop=length, step=1)
        positions = keras.ops.cast(positions, dtype=smiles.dtype)        
        embedded_smile = self.smile_embeddings(smiles)
        embedded_smile *= self.embedding_scale        
        embedded_positions = self.position_embeddings(positions)
        embedded_smile = embedded_smile + embedded_positions
        # project adsorbents embeddings from their vocabulary 
        # tile them to match the sequence length
        # Embeddings are scaled by the square root of their dimensions
        adsorbent = keras.ops.expand_dims(adsorbent, axis=1)              
        ads_embeddings = self.adsorbent_embeddings(adsorbent)         
        ads_embeddings = keras.ops.tile(ads_embeddings, [1, self.sequence_length, 1])
        ads_embeddings *= self.embedding_scale        
        # finally, project the chemometrics embeddings and process them
        # similarly to the other embeddings prior to addition
        chemometrics = keras.ops.expand_dims(chemometrics, axis=-1)
        chemo_embeddings = self.chemo_embeddings(chemometrics) 
        chemo_embeddings = keras.ops.expand_dims(chemo_embeddings, axis=1)          
        chemo_embeddings = keras.ops.tile(chemo_embeddings, [1, self.sequence_length, 1])
        chemo_embeddings *= self.embedding_scale
        # add all embeddings together to obtain the full embeddings 
        full_embedding = embedded_smile + ads_embeddings + chemo_embeddings
        # mask SMILE padding values if selected
        if self.mask_values:
            mask = keras.ops.not_equal(smiles, PAD_VALUE) 
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask = keras.ops.cast(mask, keras.config.floatx()) 
            full_embedding *= mask

        return full_embedding
    
    # build method for the custom layer 
    #-------------------------------------------------------------------------
    def build(self, input_shape):        
        super(MolecularEmbedding, self).build(input_shape)
    
    # compute the mask for padded sequences  
    #-------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):
        if mask is None:        
            mask = keras.ops.not_equal(inputs, PAD_VALUE) 
            mask = keras.ops.cast(mask, keras.config.floatx())                  
        
        return mask    
    
    # serialize layer for saving  
    #-------------------------------------------------------------------------
    def get_config(self):
        config = super(MolecularEmbedding, self).get_config()
        config.update({'smile_vocab_size': self.smile_vocab_size,
                       'ads_vocab_size': self.ads_vocab_size,                                              
                       'embedding_dims': self.embedding_dims,             
                       'sequence_length': self.sequence_length,          
                       'mask_values': self.mask_values})
        return config

    # deserialization method 
    #-------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

