import keras
from keras import activations, layers

from NISTADS.app.constants import PAD_VALUE
from NISTADS.app.utils.learning.models.transformers import AddNorm, FeedForward


# [STATE ENCODER]
###############################################################################
@keras.saving.register_keras_serializable(package="Encoders", name="StateEncoder")
class StateEncoder(keras.layers.Layer):
    def __init__(self, dropout_rate=0.2, seed=42, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dense_units = [8, 16, 32, 64]
        self.dense_layers = [
            layers.Dense(units, kernel_initializer="he_uniform")
            for units in self.dense_units
        ]
        self.bn_layers = [layers.BatchNormalization() for _ in self.dense_units]
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.seed = seed

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape):
        super(StateEncoder, self).build(input_shape)

    # -------------------------------------------------------------------------
    def call(self, x, training : bool | None = None):
        layer = keras.ops.expand_dims(x, axis=-1)
        for dense, bn in zip(self.dense_layers, self.bn_layers):
            layer = dense(layer)
            layer = bn(layer, training=training)
            layer = activations.relu(layer)

        output = self.dropout(layer, training=training)

        return output

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super(StateEncoder, self).get_config()
        config.update({"dropout_rate": self.dropout_rate, "seed": self.seed})
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# [FEED FORWARD]
###############################################################################
@keras.saving.register_keras_serializable(
    package="Encoders", name="PressureSerierEncoder"
)
class PressureSerierEncoder(keras.layers.Layer):
    def __init__(self, embedding_dims, dropout_rate, num_heads, seed=42, **kwargs):
        super(PressureSerierEncoder, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.seed = seed
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.embedding_dims
        )
        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()
        self.ffn1 = FeedForward(self.embedding_dims, 0.2, seed)
        self.ffn2 = FeedForward(self.embedding_dims, 0.3, seed)
        self.P_dense = layers.Dense(
            self.embedding_dims, kernel_initializer="he_uniform"
        )
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)

        self.supports_masking = True
        self.attention_scores = {}

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape):
        super(PressureSerierEncoder, self).build(input_shape)

    # implement transformer encoder through call method
    # -------------------------------------------------------------------------
    def call(self, pressure, context, key_mask=None, training : bool | None = None):
        # compute query mask as the masked pressure series
        query_mask = self.compute_mask(pressure)
        # project the pressure series into the embedding space using
        pressure = keras.ops.expand_dims(pressure, axis=-1)
        pressure = self.P_dense(pressure)

        # cross-attention between the pressure series and the molecular context
        # the latter being generated from self-attention of the enriched SMILE sequences
        attention_output, cross_attention_scores = self.attention(
            query=pressure,
            key=context,
            value=context,
            query_mask=query_mask,
            key_mask=key_mask,
            value_mask=key_mask,
            training=training,
            return_attention_scores=True,
        )
        addnorm = self.addnorm1([pressure, attention_output])
        # store cross-attention scores for later retrieval
        self.attention_scores["cross_attention_scores"] = cross_attention_scores

        # feed forward network with ReLU activation to further process the output
        # addition and layer normalization of inputs and outputs
        ffn_out = self.ffn1(addnorm, training=training)
        output = self.addnorm2([addnorm, ffn_out])

        return output

    # -------------------------------------------------------------------------
    def get_attention_scores(self):
        return self.attention_scores

    # compute the mask for padded sequences
    # -------------------------------------------------------------------------
    def compute_mask(self, inputs, previous_mask = None):
        mask = keras.ops.not_equal(inputs, PAD_VALUE)
        mask = keras.ops.cast(mask, keras.config.floatx())

        return mask

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super(PressureSerierEncoder, self).get_config()
        config.update(
            {
                "embedding_dims": self.embedding_dims,
                "dropout_rate": self.dropout_rate,
                "num_heads": self.num_heads,
                "seed": self.seed,
            }
        )
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# [UPTAKE DECODER]
###############################################################################
@keras.saving.register_keras_serializable(package="Decoders", name="QDecoder")
class QDecoder(keras.layers.Layer):
    def __init__(self, embedding_dims=128, dropout_rate=0.2, seed=42, **kwargs):
        super(QDecoder, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.state_dense = layers.Dense(
            self.embedding_dims, kernel_initializer="he_uniform"
        )
        self.dense = [
            layers.Dense(self.embedding_dims, kernel_initializer="he_uniform")
            for x in range(4)
        ]
        self.batch_norm = [layers.BatchNormalization() for _ in range(4)]
        self.dropout = layers.Dropout(rate=dropout_rate, seed=seed)
        self.Q_output = layers.Dense(1, kernel_initializer="he_uniform")
        self.seed = seed
        self.supports_masking = True

    # build method for the custom layer
    # -------------------------------------------------------------------------
    def build(self, input_shape):
        super(QDecoder, self).build(input_shape)

    # compute the mask for padded sequences
    # -------------------------------------------------------------------------
    def compute_mask(self, inputs, previous_mask = None):
        mask = keras.ops.not_equal(inputs, PAD_VALUE)
        mask = keras.ops.expand_dims(mask, axis=-1)
        mask = keras.ops.cast(mask, keras.config.floatx())

        return mask

    # implement transformer encoder through call method
    # -------------------------------------------------------------------------
    def call(self, P_logits, pressure, state, mask=None, training : bool | None = None):
        mask = self.compute_mask(pressure) if mask is None else mask
        layer = P_logits * mask if mask is not None else P_logits

        # ideally, higher temperature should decrease the adsorbed amount, therefor
        # temperature is used to compute an inverse scaling factor for the output
        state = self.state_dense(state)
        state = activations.relu(state)
        T_scale = keras.ops.expand_dims(keras.ops.exp(-state), axis=1)

        for dense, bn in zip(self.dense, self.batch_norm):
            layer = dense(layer)
            layer = bn(layer, training=training)
            layer = activations.relu(layer)
            layer = layer * T_scale

        output = self.Q_output(layer)
        output = activations.relu(output)

        return output

    # serialize layer for saving
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super(QDecoder, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "embedding_dims": self.embedding_dims,
                "seed": self.seed,
            }
        )
        return config

    # deserialization method
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        return cls(**config)
