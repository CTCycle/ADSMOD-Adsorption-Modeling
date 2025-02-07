import torch
import keras
from keras import layers, Model, optimizers

from NISTADS.commons.utils.learning.transformers import TranSMILEncoder, AddNorm
from NISTADS.commons.utils.learning.embeddings import MolecularEmbedding
from NISTADS.commons.utils.learning.encoders import StateEncoder, PressureSerierEncoder, QDecoder
from NISTADS.commons.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared


# [XREP CAPTIONING MODEL]
###############################################################################
class SCADSModel: 

    def __init__(self, metadata, configuration):

        self.smile_vocab_size = metadata.get('SMILE_vocabulary_size', 0)
        self.ads_vocab_size = metadata.get('adsorbent_vocabulary_size', 0)        
        
        self.seed = configuration["SEED"]
        self.smile_length = configuration["dataset"]["SMILE_PADDING"]
        self.series_length = configuration["dataset"]["MAX_PQ_POINTS"]       
        self.embedding_dims = configuration["model"]["MOLECULAR_EMBEDDING"]        
        self.num_heads = configuration["model"]["ATTENTION_HEADS"]  
        self.num_encoders = configuration["model"]["NUM_ENCODERS"]         
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]             
        self.learning_rate = configuration["training"]["LEARNING_RATE"]           
        self.configuration = configuration
        
        # initialize the image encoder and the transformers encoders and decoders
        self.state_input = layers.Input(shape=(2,), name='state_input')
        self.adsorbents_input = layers.Input(shape=(), name='adsorbent_input')
        self.adsorbates_input = layers.Input(shape=(self.smile_length,), name='adsorbate_input')
        self.pressure_input = layers.Input(shape=(self.series_length,), name='pressure_input')       

        self.state_encoder = StateEncoder(0.2, self.series_length, seed=self.seed)
        self.smile_encoders = [TranSMILEncoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size, self.ads_vocab_size, self.embedding_dims, self.smile_length)         
        self.pressure_encoder = PressureSerierEncoder(256, dropout=0.2, seed=self.seed) 
        self.Q_decoder = QDecoder(
            self.embedding_dims, self.series_length, dropout=0.2, depth=4, seed=self.seed)

        self.addnorm = AddNorm()

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):         

        molecular_embeddings = self.molecular_embeddings(self.adsorbates_input, self.adsorbents_input)
        encoder_output = molecular_embeddings          
        for encoder in self.smile_encoders:
            encoder_output = encoder(encoder_output, training=False)

        encoded_states = self.state_encoder(self.state_input, training=False)
        encoded_pressure = self.pressure_encoder(self.pressure_input, training=False)
        output = self.Q_decoder(encoder_output, encoded_pressure, encoded_states, training=False)

        # wrap the model and compile it with AdamW optimizer
        model = Model(inputs=[self.state_input, self.adsorbents_input, 
                      self.adsorbates_input, self.pressure_input], 
                      outputs=output)       
        
        loss = MaskedMeanSquaredError()  
        metric = [MaskedRSquared()]
        opt = optimizers.AdamW(learning_rate=self.learning_rate)          
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False) 

        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')       

        if model_summary:
            model.summary(expand_nested=True)

        return model     
    
    

                 
