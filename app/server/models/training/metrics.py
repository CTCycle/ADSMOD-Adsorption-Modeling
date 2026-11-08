# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

import keras

from server.common.constants import PAD_VALUE


# [LOSS FUNCTION]

###############################################################################
@keras.saving.register_keras_serializable(
    package="CustomLoss", name="MaskedMeanSquaredError"
)
class MaskedMeanSquaredError(keras.losses.Loss):

    # -------------------------------------------------------------------------
    def __init__(self, name: str = "MaskedMeanSquaredError", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    # -------------------------------------------------------------------------
    def call(self, y_true: Any, y_pred: Any) -> Any:
        mask = keras.ops.not_equal(y_true, PAD_VALUE)
        mask = keras.ops.cast(mask, dtype=y_true.dtype)
        y_pred = keras.ops.squeeze(y_pred, axis=-1)
        loss = keras.ops.square(y_true - y_pred)
        loss *= mask
        loss = keras.ops.sum(loss) / (keras.ops.sum(mask) + keras.backend.epsilon())

        return loss

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "name": self.name}

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[MaskedMeanSquaredError], config: dict[str, Any]
    ) -> MaskedMeanSquaredError:
        return cls(**config)


# [METRICS]

###############################################################################
@keras.saving.register_keras_serializable(
    package="CustomMetrics", name="MaskedRSquared"
)
class MaskedRSquared(keras.metrics.Metric):

    # -------------------------------------------------------------------------
    def __init__(self, name: str = "MaskedR2", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.n = self.add_weight(name="n", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.sse = self.add_weight(name="sse", initializer="zeros")

    # -------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_pred = keras.ops.squeeze(y_pred, axis=-1)

        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        weights = keras.ops.cast(keras.ops.not_equal(y_true, PAD_VALUE), "float32")
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype="float32")
            weights = keras.ops.multiply(weights, sample_weight)

        self.n.assign_add(keras.ops.sum(weights))
        self.sum_y.assign_add(keras.ops.sum(weights * y_true))
        self.sum_y2.assign_add(keras.ops.sum(weights * keras.ops.square(y_true)))
        self.sse.assign_add(keras.ops.sum(weights * keras.ops.square(y_true - y_pred)))

    # -------------------------------------------------------------------------
    def result(self) -> Any:
        epsilon = keras.backend.epsilon()
        sst = self.sum_y2 - keras.ops.square(self.sum_y) / (self.n + epsilon)
        return 1 - self.sse / (sst + epsilon)

    # -------------------------------------------------------------------------
    def reset_states(self) -> None:
        self.n.assign(0)
        self.sum_y.assign(0)
        self.sum_y2.assign(0)
        self.sse.assign(0)

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "name": self.name}

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[MaskedRSquared], config: dict[str, Any]
    ) -> MaskedRSquared:
        return cls(**config)
