import torch
import keras
from keras import layers, activations, Model, optimizers

from NISTADS.commons.utils.learning.transformers import TransMolecularEncoder, AddNorm
from NISTADS.commons.utils.learning.embeddings import MolecularEmbedding
from NISTADS.commons.utils.learning.encoders import StateEncoder, PressureSerierEncoder
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

        self.state_encoder = StateEncoder(0.2,seed=self.seed)
        self.smile_encoders = [TransMolecularEncoder(
            self.embedding_dims, self.num_heads, self.seed) for _ in range(self.num_encoders)]
        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size, self.ads_vocab_size, self.embedding_dims, self.smile_length)         
        self.pressure_encoder = PressureSerierEncoder(self.embedding_dims, dropout=0.2, seed=self.seed) 

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):         

        # create combined embeddings of both the adsorbates and adsorbents 
        # molecular representations, where the adsorbate is embedded as a SMILE sequence
        # to which we add the adsorbent and positional contribution
        molecular_embeddings = self.molecular_embeddings(self.adsorbates_input, self.adsorbents_input)

        # pass the molecular embeddings through the transformer encoders
        encoder_output = molecular_embeddings          
        for encoder in self.smile_encoders:
            encoder_output = encoder(encoder_output, training=False)

        mean_encoder_output = keras.ops.mean(encoder_output, axis=1)

        # encode temperature and molecular weight of the adsorbate as a single vector
        # and tile it to match the SMILE sequence length
        encoded_states = self.state_encoder(self.state_input, training=False)

        # create a molecular context by concatenating the mean molecular embeddings
        # together with the encoded states (temperature and molecular weight)
        molecular_context = layers.Concatenate()([mean_encoder_output, encoded_states])        

        # encode the pressure series and add information from the molecular context
        encoded_pressure = self.pressure_encoder(self.pressure_input, molecular_context, training=False)        

        Q_logits = layers.Dense(self.embedding_dims, kernel_initializer='he_uniform')(encoded_pressure)
        Q_logits = activations.relu(Q_logits)
        Q_logits = layers.Dense(self.embedding_dims//2, kernel_initializer='he_uniform')(Q_logits)
        Q_logits = activations.relu(Q_logits)
        Q_logits = layers.Dense(self.embedding_dims//4, kernel_initializer='he_uniform')(Q_logits)
        Q_logits = activations.relu(Q_logits)
        Q_logits = layers.Dense(1, kernel_initializer='he_uniform')(Q_logits)        
        output = activations.relu(Q_logits)

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
    
    

                 
