import torch
import keras

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger
           
# [LEARNING RATE SCHEDULER]
###############################################################################
@keras.utils.register_keras_serializable(package='LRScheduler')
class LRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, constant_steps, decay_steps, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
        self.initial_lr = initial_lr
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps

    #--------------------------------------------------------------------------
    def __call__(self, step):
        # Cast step and the phase lengths to torch.float32 for computation
        global_step = keras.ops.cast(step, torch.float32)
        constant_steps = keras.ops.cast(self.constant_steps, torch.float32)
        decay_steps = keras.ops.cast(self.decay_steps, torch.float32)        
        # Constant phase: LR remains equal to post_warmup_lr
        constant_lr = self.initial_lr        
        # Decay phase: linear decay from post_warmup_lr to 0
        # Calculate progress in the decay phase: 0.0 at the start, 1.0 at the end.
        decay_progress = (global_step - constant_steps) / decay_steps
        decayed_lr = self.initial_lr * (1.0 - decay_progress)
        # Clamp to zero so that lr never goes negative
        decayed_lr = keras.ops.maximum(decayed_lr, torch.tensor(0.0, dtype=torch.float32))        
        
        learning_rate = keras.ops.cond(global_step < constant_steps,
                                       lambda: constant_lr,
                                       lambda: decayed_lr)
        
        return learning_rate

    #--------------------------------------------------------------------------
    def get_config(self):
        return {'initial_lr': self.initial_lr,
                'constant_steps': self.constant_steps,
                'decay_steps': self.decay_steps}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
      
