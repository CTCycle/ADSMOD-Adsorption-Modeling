import keras

from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.logger import logger

# [LOSS FUNCTION]
###############################################################################
class MaskedMeanSquaredError(keras.losses.Loss):
    
    def __init__(self, name='MaskedMeanSquaredError', **kwargs):
        super(MaskedMeanSquaredError, self).__init__(name=name, **kwargs)        
        
    #--------------------------------------------------------------------------    
    def call(self, y_true, y_pred):
        mask = keras.ops.not_equal(y_true, PAD_VALUE)        
        mask = keras.ops.cast(mask, dtype=y_true.dtype)
        # squeeze output dimensions: (batch size, points, 1) --> (batch size, points)
        y_pred = keras.ops.squeeze(y_pred, axis=-1)  
        loss = keras.ops.square(y_true - y_pred)             
        loss *= mask
        loss = keras.ops.sum(loss)/(keras.ops.sum(mask) + keras.backend.epsilon())

        return loss
    
    #--------------------------------------------------------------------------    
    def get_config(self):
        base_config = super(MaskedMeanSquaredError, self).get_config()
        return {**base_config, 'name': self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
   
# [METRICS]
###############################################################################
class MaskedRSquared(keras.metrics.Metric):

    def __init__(self, name='MaskedR2', **kwargs):
        super(MaskedRSquared, self).__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name='ssr', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    #--------------------------------------------------------------------------  
    def update_state(self, y_true, y_pred, sample_weight=None):
        # squeeze output dimensions: (batch size, points, 1) --> (batch size, points)
        y_pred = keras.ops.squeeze(y_pred, axis=-1)  
        
        y_true = keras.ops.cast(y_true, dtype='float32')
        y_pred = keras.ops.cast(y_pred, dtype='float32')
        
        # Create a mask to ignore padding values 
        mask = keras.ops.not_equal(y_true, PAD_VALUE)
        mask = keras.ops.cast(mask, dtype='float32')
        
        # Compute residual sum of squares (SSR)
        residuals = keras.ops.square(y_true - y_pred)
        residuals = keras.ops.multiply(residuals, mask)
        
        # Compute total sum of squares (SST)
        mean_y_true = keras.ops.sum(y_true * mask) / (keras.ops.sum(mask) + keras.backend.epsilon())
        total_variance = keras.ops.square(y_true - mean_y_true)
        total_variance = keras.ops.multiply(total_variance, mask)
        
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype='float32')
            residuals = keras.ops.multiply(residuals, sample_weight)
            total_variance = keras.ops.multiply(total_variance, sample_weight)
            mask = keras.ops.multiply(mask, sample_weight)
        
        # Update the state variables
        self.ssr.assign_add(keras.ops.sum(residuals))
        self.sst.assign_add(keras.ops.sum(total_variance))
        self.count.assign_add(keras.ops.sum(mask))
    
    #--------------------------------------------------------------------------  
    def result(self):
        return 1 - (self.ssr / (self.sst + keras.backend.epsilon()))
    
    #--------------------------------------------------------------------------  
    def reset_states(self):
        self.ssr.assign(0)
        self.sst.assign(0)
        self.count.assign(0)
    
    #--------------------------------------------------------------------------  
    def get_config(self):
        base_config = super(MaskedRSquared, self).get_config()
        return {**base_config, 'name': self.name}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)







