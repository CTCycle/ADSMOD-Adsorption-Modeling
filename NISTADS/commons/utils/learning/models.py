import torch
from keras import layers, Model, optimizers

from NISTADS.commons.utils.learning.scheduler import LRScheduler
from NISTADS.commons.utils.learning.embeddings import MolecularEmbedding
from NISTADS.commons.utils.learning.encoders import StateEncoder, PressureSerierEncoder, MolecularEncoder, QDecoder
from NISTADS.commons.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared


# [XREP CAPTIONING MODEL]
###############################################################################
class SCADSModel: 

    def __init__(self, metadata, configuration):
        self.smile_vocab_size = metadata.get('SMILE_vocabulary_size', 0)
        self.ads_vocab_size = metadata.get('adsorbent_vocabulary_size', 0)        
        self.seed = configuration["SEED"]
        self.smile_length = metadata["dataset"]["SMILE_PADDING"]
        self.series_length = metadata["dataset"]["MAX_PQ_POINTS"]       
        self.embedding_dims = configuration["model"]["MOLECULAR_EMBEDDING"]            
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]

        self.scheduler_config = configuration["training"]["LR_SCHEDULER"] 
        self.initial_lr = self.scheduler_config["INITIAL_LR"]
        self.constant_lr_steps = self.scheduler_config["CONSTANT_STEPS"]       
        self.decay_steps = self.scheduler_config["DECAY_STEPS"]         
        self.configuration = configuration       

        self.state_encoder = StateEncoder(0.2, seed=self.seed)        
        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size, self.ads_vocab_size, self.embedding_dims, self.smile_length, mask_values=True)   
        self.smile_encoders = MolecularEncoder(self.embedding_dims, self.seed)       
        self.pressure_encoder = PressureSerierEncoder(
            self.embedding_dims, dropout_rate=0.2, seed=self.seed) 
        self.Qdecoder = QDecoder(self.embedding_dims, dropout_rate=0.2, seed=self.seed)

        self.state_input = layers.Input(shape=(), name='state_input')
        self.chemo_input = layers.Input(shape=(), name='chemo_input')
        self.adsorbents_input = layers.Input(shape=(), name='adsorbent_input')
        self.adsorbates_input = layers.Input(shape=(self.smile_length,), name='adsorbate_input')
        self.pressure_input = layers.Input(shape=(self.series_length,), name='pressure_input')    

    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True): 
        # create combined embeddings of both the adsorbates and adsorbents 
        # molecular representations, where the adsorbate is embedded as a SMILE sequence
        # to which we add the adsorbent and positional contribution together with the chemometrics
        molecular_embeddings = self.molecular_embeddings(
            self.adsorbates_input, self.adsorbents_input, self.chemo_input) 
        smile_mask = self.molecular_embeddings.compute_mask(self.adsorbates_input)   

        # pass the molecular embeddings through the transformer encoders               
        encoder_output = self.smile_encoders(
            molecular_embeddings, smile_mask, training=False)

        # encode temperature and molecular weight of the adsorbate as a single vector
        # and tile it to match the SMILE sequence length
        encoded_states = self.state_encoder(self.state_input, training=False)           

        # encode the pressure series and add information from the molecular context
        encoded_pressure = self.pressure_encoder(
            self.pressure_input, encoder_output, encoded_states, training=False) 
        
        output = self.Qdecoder(encoded_pressure, self.pressure_input)        

        # wrap the model and compile it with Adam optimizer
        model = Model(
            inputs=[self.state_input, self.adsorbents_input, self.adsorbates_input, 
                    self.pressure_input], outputs=output, name='SCADS_model')       
        
        lr_schedule = LRScheduler(self.initial_lr, self.constant_lr_steps, self.decay_steps)
        opt = optimizers.Adam(learning_rate=lr_schedule)  
        loss = MaskedMeanSquaredError()  
        metric = [MaskedRSquared()]                
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False) 

        if model_summary:
            model.summary(expand_nested=True)
    
        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')

        return model 
    
        

        
    

                 
