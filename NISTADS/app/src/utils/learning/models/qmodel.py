import torch
from keras import layers, Model, optimizers

from NISTADS.app.src.utils.learning.training.scheduler import LinearDecayLRScheduler
from NISTADS.app.src.utils.learning.models.embeddings import MolecularEmbedding
from NISTADS.app.src.utils.learning.models.transformers import TransformerEncoder
from NISTADS.app.src.utils.learning.models.encoders import StateEncoder, PressureSerierEncoder, QDecoder
from NISTADS.app.src.utils.learning.metrics import MaskedMeanSquaredError, MaskedRSquared


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
        self.num_heads = configuration["model"]["ATTENTION_HEADS"]   
        self.num_encoders = configuration["model"]["NUM_ENCODERS"]            
        self.jit_compile = configuration["model"]["JIT_COMPILE"]
        self.jit_backend = configuration["model"]["JIT_BACKEND"]

        self.scheduler_config = configuration["training"]["LR_SCHEDULER"] 
        self.initial_lr = self.scheduler_config["INITIAL_LR"]
        self.constant_lr_steps = self.scheduler_config["CONSTANT_STEPS"]       
        self.decay_steps = self.scheduler_config["DECAY_STEPS"]   
        self.final_lr = self.scheduler_config["FINAL_LR"]         
        self.configuration = configuration 
        
        self.state_encoder = StateEncoder(0.2, seed=self.seed)        
        self.molecular_embeddings = MolecularEmbedding(
            self.smile_vocab_size, self.ads_vocab_size, self.embedding_dims, 
            self.smile_length, mask_values=True) 
        self.encoders = [TransformerEncoder(
            self.embedding_dims, self.num_heads, self.seed)
            for _ in range(self.num_encoders)]               
        self.pressure_encoder = PressureSerierEncoder(
            self.embedding_dims, 0.2, self.num_heads, self.seed) 
        self.Qdecoder = QDecoder(self.embedding_dims, 0.2, self.seed)

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

        # pass the molecular embeddings through the stack of transformer encoders
        # apply SMILE mask to ignore padding values                   
        encoder_output = molecular_embeddings    
        for encoder in self.encoders:
            encoder_output = encoder(
                encoder_output, mask=smile_mask, training=False)

        # encode temperature and molecular weight of the adsorbate as a single vector
        # and tile it to match the SMILE sequence length
        encoded_states = self.state_encoder(self.state_input, training=False)           

        # encode the pressure series and add information from the molecular context
        encoded_pressure = self.pressure_encoder(
            self.pressure_input, encoder_output, smile_mask, training=False) 
        
        output = self.Qdecoder(encoded_pressure, self.pressure_input, encoded_states)        

        # wrap the model and compile it with Adam optimizer
        model = Model(inputs=[self.state_input, self.chemo_input, self.adsorbents_input, self.adsorbates_input, 
                      self.pressure_input], outputs=output, name='SCADS_model')       
        
        lr_schedule = LinearDecayLRScheduler(
            self.initial_lr, self.constant_lr_steps, self.decay_steps, self.final_lr)
        opt = optimizers.AdamW(learning_rate=lr_schedule)  
        loss = MaskedMeanSquaredError()  
        metric = [MaskedRSquared()]                
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False) 

        if model_summary:
            model.summary(expand_nested=True)
    
        if self.jit_compile:
            model = torch.compile(model, backend=self.jit_backend, mode='default')

        return model 
    
        

        
    

                 
