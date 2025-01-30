import torch
from keras import layers, Model, optimizers
from transformers import AutoFeatureExtractor, AutoModel
import tensorflow as tf

from NISTADS.commons.utils.learning.transformers import TransformerEncoder
from NISTADS.commons.utils.learning.embeddings import PositionalEmbedding
from NISTADS.commons.utils.learning.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy


# [XREP CAPTIONING MODEL]
###############################################################################
class SCADSModel: 

    def __init__(self, metadata, configuration):

        self.smile_vocab_size = metadata.get('SMILE_vocabulary_size', 0)
        self.ads_vocab_size = metadata.get('adsorbent_vocabulary_size', 0)        
        
        self.seed = configuration["SEED"]
        self.smile_length = configuration["dataset"]["SMILE_PADDING"]
        self.series_length = configuration["dataset"]["MAX_PQ_POINTS"]       
        self.smile_embedding_dims = configuration["model"]["GUEST_EMBEDDING_DIMS"] 
        self.ads_embedding_dims = configuration["model"]["HOST_EMBEDDING_DIMS"] 
        self.num_heads = configuration["model"]["ATTENTION_HEADS"]  
        self.num_encoders = configuration["model"]["NUM_ENCODERS"]         
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]             
        self.learning_rate = configuration["training"]["LEARNING_RATE"]        
        self.temperature = configuration["training"]["TEMPERATURE"]
        self.configuration = configuration
        
        # initialize the image encoder and the transformers encoders and decoders
        self.parameters_input = layers.Input(shape=(2,), name='parameters_input')
        self.adsorbents_input = layers.Input(shape=(), name='adsorbent_input')
        self.adsorbates_input = layers.Input(shape=(self.smile_length), name='adsorbate_input')
        self.pressure_input = layers.Input(shape=(self.series_length), name='pressure_input')       

        self.encoders = [TransformerEncoder(self.smile_embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.smile_embeddings = PositionalEmbedding(
            self.smile_vocab_size, self.smile_embedding_dims, self.smile_length) 
        self.adsorbent_embeddings = PositionalEmbedding(
            self.ads_vocab_size, self.ads_embedding_dims, self.series_length)   

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
        # encode images and extract their features using the convolutional 
        # image encoder or a selected pretrained model
             
        smile_embeddings = self.smile_embeddings(self.adsorbates_input)
        smile_padding_mask = self.smile_embeddings.compute_mask(self.adsorbates_input)         
                
        encoder_output = smile_embeddings
           
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=False)       

        # wrap the model and compile it with AdamW optimizer
        model = Model(inputs=[self.parameters_input, self.adsorbents_input, 
                      self.adsorbates_input, self.pressure_input], 
                      outputs=encoder_output)       
        
        loss = MaskedSparseCategoricalCrossentropy()  
        metric = [MaskedAccuracy()]
        opt = optimizers.AdamW(learning_rate=self.learning_rate)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False) 

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')       

        if model_summary:
            model.summary(expand_nested=True)

        return model     
    
    

                 
