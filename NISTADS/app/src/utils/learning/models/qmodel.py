from torch import compile as torch_compile
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
        self.smile_length = metadata.get('SMILE_sequence_size', 20)
        self.series_length = metadata.get('max_measurements', 30)
        self.seed = configuration.get('train_seed', 42)
        self.embedding_dims = configuration.get('molecular_embedding_size', 64)
        self.num_heads = configuration.get('num_attention_heads', 2)
        self.num_encoders = configuration.get('num_encoders', 2)
        self.jit_compile = configuration.get('jit_compile', False)
        self.jit_backend = configuration.get('jit_backend', 'inductor')        
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

    #--------------------------------------------------------------------------
    def compile_model(self, model : Model, model_summary=True):
        initial_LR = self.configuration.get('initial_RL', 0.001)
        LR_schedule = initial_LR        
        if self.configuration.get('use_scheduler', False):          
            constant_lr_steps = self.configuration.get('constant_steps', 1000)   
            decay_steps = self.configuration.get('decay_steps', 500)  
            final_LR = self.configuration.get('final_LR', 0.0001)          
            LR_schedule = LinearDecayLRScheduler(
                initial_LR, constant_lr_steps, decay_steps, final_LR)  
        
        opt = optimizers.AdamW(learning_rate=LR_schedule)  
        loss = MaskedMeanSquaredError()  
        metric = [MaskedRSquared()]                
        model.compile(loss=loss, optimizer=opt, metrics=metric, jit_compile=False)            
        # print model summary on console and run torch.compile 
        # with triton compiler and selected backend
        model.summary(expand_nested=True) if model_summary else None
        if self.jit_compile:
            model = torch_compile(model, backend=self.jit_backend, mode='default')

        return model     

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
        model = Model(inputs=[self.state_input, self.chemo_input, 
                              self.adsorbents_input, self.adsorbates_input, 
                              self.pressure_input], outputs=output, name='SCADS_model')
        model = self.compile_model(model, model_summary=model_summary)  
             
        return model 
    
        