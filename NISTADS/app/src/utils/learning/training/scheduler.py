import numpy as np
import keras


from NISTADS.app.src.logger import logger
           
# [LEARNING RATE SCHEDULER]
###############################################################################
@keras.saving.register_keras_serializable(package='LinearDecayLRScheduler')
class LinearDecayLRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, constant_steps, decay_steps, final_lr, **kwargs):
        super(LinearDecayLRScheduler, self).__init__(**kwargs)
        self.initial_lr = initial_lr
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        self.final_lr = final_lr

    #--------------------------------------------------------------------------
    def __call__(self, step):        
        global_step = keras.ops.cast(step, np.float32)
        constant_steps = keras.ops.cast(self.constant_steps, np.float32)
        decay_steps = keras.ops.cast(self.decay_steps, np.float32)
        initial_lr = keras.ops.cast(self.initial_lr, np.float32)
        final_lr = keras.ops.cast(self.final_lr, np.float32)

        # Compute the decayed learning rate (linear interpolation).
        # progress is 0.0 when global_step equals constant_steps,
        # and 1.0 when global_step equals constant_steps + decay_steps.
        progress = (global_step - constant_steps) / decay_steps

        # Compute linearly decayed lr: it decreases from initial_lr to final_lr.
        decayed_lr = initial_lr - (initial_lr - final_lr) * progress
        # Ensure the decayed lr does not drop below final_lr.
        decayed_lr = keras.ops.maximum(decayed_lr, final_lr)

        # Before constant_steps, use the initial constant lr.
        # After constant_steps, use the decayed lr.
        learning_rate = keras.ops.where(global_step < constant_steps, 
                                        initial_lr, 
                                        decayed_lr)

        return learning_rate

    #--------------------------------------------------------------------------
    def get_config(self):
        return {'initial_lr': self.initial_lr,
                'constant_steps': self.constant_steps,
                'decay_steps': self.decay_steps,
                'final_lr': self.final_lr}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
      
