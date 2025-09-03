from __future__ import annotations

from typing import Any

import keras
import numpy as np


# [LEARNING RATE SCHEDULER]
###############################################################################
@keras.saving.register_keras_serializable(package="LinearDecayLRScheduler")
class LinearDecayLRScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_LR: float,
        constant_steps: int,
        decay_steps: int,
        target_LR: float,
        **kwargs,
    ) -> None:
        super(LinearDecayLRScheduler, self).__init__(**kwargs)
        self.initial_LR = initial_LR
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        self.target_LR = target_LR

    # -------------------------------------------------------------------------
    def __call__(self, step) -> Any:
        global_step = keras.ops.cast(step, np.float32)
        constant_steps = keras.ops.cast(self.constant_steps, np.float32)
        decay_steps = keras.ops.cast(self.decay_steps, np.float32)
        initial_LR = keras.ops.cast(self.initial_LR, np.float32)
        target_LR = keras.ops.cast(self.target_LR, np.float32)

        # Compute the decayed learning rate (linear interpolation).
        # progress is 0.0 when global_step equals constant_steps,
        # and 1.0 when global_step equals constant_steps + decay_steps.
        progress = (global_step - constant_steps) / decay_steps

        # Compute linearly decayed lr: it decreases from initial_LR to target_LR.
        decayed_LR = initial_LR - (initial_LR - target_LR) * progress
        # Ensure the decayed lr does not drop below target_LR.
        decayed_LR = keras.ops.maximum(decayed_LR, target_LR)

        # Before constant_steps, use the initial constant lr.
        # After constant_steps, use the decayed lr.
        learning_rate = keras.ops.where(
            global_step < constant_steps, initial_LR, decayed_LR
        )

        return learning_rate

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            "initial_LR": self.initial_LR,
            "constant_steps": self.constant_steps,
            "decay_steps": self.decay_steps,
            "target_LR": self.target_LR,
        }

    @classmethod
    def from_config(cls, config) -> "LinearDecayLRScheduler":
        return cls(**config)
