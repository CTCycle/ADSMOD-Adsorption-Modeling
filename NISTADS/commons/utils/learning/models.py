import os
import numpy as np
import json
from datetime import datetime
import tensorflow as tf
from keras import layers, activations

               




    

     
    

    
    
# [SCADS MODEL]
#====================================================================================================================================================
class SCADSModel:

    def __init__(self, learning_rate, num_features, sequence_length, pad_value, adsorbent_dims, 
                 adsorbates_dims, embedding_dims, seed=42, XLA_acceleration=False):

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.pad_value = pad_value
        self.adsorbent_dims = adsorbent_dims
        self.adsorbates_dims = adsorbates_dims 
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.XLA_state = XLA_acceleration
        self.parametrizer = Parametrizer(sequence_length, seed)
        self.embedder = GHEncoder(sequence_length, adsorbates_dims, adsorbent_dims, embedding_dims, seed)
        self.encoder = PressureEncoder(pad_value, seed)
        self.decoder = QDecoder(sequence_length, seed)
        
    # build model given the architecture
    #--------------------------------------------------------------------------
    def get_model(self, model_summary=True):       
       
        # define model inputs using input layers
        feat_inputs = layers.Input(shape = (self.num_features, ))
        host_inputs = layers.Input(shape = (1,))
        guest_inputs = layers.Input(shape = (1,))
        pressure_inputs = layers.Input(shape = (self.sequence_length, ))
               
        parametrizer = self.parametrizer(feat_inputs)
        GH_encoder = self.embedder(host_inputs, guest_inputs)        
        pressure_encoder = self.encoder(pressure_inputs)
        decoder = self.decoder(parametrizer, GH_encoder, pressure_encoder)        
        
        model = Model(inputs=[feat_inputs, host_inputs, guest_inputs, pressure_inputs],
                      outputs=decoder, name='SCADS')
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = MaskedMeanSquaredError(self.pad_value)
        metrics = MaskedMeanAbsoluteError(self.pad_value)
        model.compile(loss=loss, optimizer=opt, metrics=metrics, run_eagerly=False,
                      COMPILE=self.XLA_state)     
        if model_summary==True:
            model.summary(expand_nested=True)

        return model
                 
