from __future__ import annotations

from typing import Any

import keras
from keras import activations, layers


# [ADD NORM LAYER]
###############################################################################
@keras.saving.register_keras_serializable(package="CustomLayers", name="AddNorm")
class AddNorm(keras.layers.Layer):
    def __init__(self, epsilon: float = 10e-10, **kwargs) -> None:
        super(AddNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization(epsilon=self.epsilon)

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super(AddNorm, self).build(input_shape)

    # implement transformer encoder through call method
    # -------------------------------------------------------------------------
    def call(self, inputs) -> Any:
        x1, x2 = inputs
        x_add = self.add([x1, x2])
        x_norm = self.layernorm(x_add)

        return x_norm

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(AddNorm, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> "AddNorm":
        return cls(**config)


# [FEED FORWARD]
###############################################################################
@keras.saving.register_keras_serializable(package="CustomLayers", name="FeedForward")
class FeedForward(keras.layers.Layer):
    def __init__(
        self, dense_units: int, dropout: float, seed: int = 42, **kwargs
    ) -> None:
        super(FeedForward, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(dense_units, kernel_initializer="he_uniform")
        self.dense2 = layers.Dense(dense_units, kernel_initializer="he_uniform")
        self.dropout = layers.Dropout(rate=dropout, seed=seed)
        self.seed = seed
        self.supports_masking = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super(FeedForward, self).build(input_shape)

    # implement transformer encoder through call method
    # -------------------------------------------------------------------------
    def call(self, x, training: bool | None = None) -> Any:
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        x = activations.relu(x)
        output = self.dropout(x, training=training)
        return output

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(FeedForward, self).get_config()
        config.update(
            {
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
            }
        )
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> "FeedForward":
        return cls(**config)


# [TRANSFORMER ENCODER]
###############################################################################
@keras.saving.register_keras_serializable(package="Encoders", name="TransformerEncoder")
class TransformerEncoder(keras.layers.Layer):
    def __init__(
        self, embedding_dims: int, num_heads: int, seed: int, **kwargs
    ) -> None:
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.seed = seed
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dims, seed=self.seed
        )
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)

        # set mask supports to True but mask propagation is handled
        # through the attention layer call
        self.supports_masking = True
        self.attention_scores = {}

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape) -> None:
        super(TransformerEncoder, self).build(input_shape)

    # implement transformer encoder through call method
    # -------------------------------------------------------------------------
    def call(self, inputs, mask=None, training: bool | None = None) -> Any:
        # self attention with causal masking, using the embedded captions as input
        # for query, value and key. The output of this attention layer is then summed
        # to the inputs and normalized
        attention_output, self_attention_scores = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            query_mask=mask,
            value_mask=mask,
            key_mask=mask,
            training=training,
            return_attention_scores=True,
        )
        addnorm = self.addnorm1([inputs, attention_output])
        # store self-attention scores for later retrieval
        self.attention_scores["self_attention_scores"] = self_attention_scores

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])

        return output

    # -------------------------------------------------------------------------
    def get_attention_scores(self) -> dict:
        return self.attention_scores

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        config = super(TransformerEncoder, self).get_config()
        config.update(
            {
                "embedding_dims": self.embedding_dims,
                "num_heads": self.num_heads,
                "seed": self.seed,
            }
        )
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config) -> "TransformerEncoder":
        return cls(**config)
