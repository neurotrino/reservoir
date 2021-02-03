"""TODO: module docs"""

# external
from typing import Any

import tensorflow as tf

# local
from models.common import BaseModel, SpikeRegularization, exp_convolve
from models.neurons.lif import *
from models.neurons.adex import *

class SinusoidSlayer(BaseModel):
    """Model for the sinusoid-matching example."""

    def __init__(self,
        uni: Any,
        spike_identity: str, # silly thing for demo purposes
        cell: LIF
    ):
        super().__init__()
        # Training parameters needed for all models
        self.uni = uni

        # Model-specific attributes
        # (this particular value is just here for demo purposes)
        self.spike_identity = spike_identity

        # Sub-networks and layers
        self.cell = cell

    def build(self, cfg):
        """TODO: method docs"""
        # TODO
        uni = self.uni
        cell = self.cell

        # is there a mismatch happening here for input dimensions?
        # below it seems like input has 0th dimension of batch size
        # whereas right here it seems like the 0th dimension is time
        inputs = tf.keras.Input(shape=(uni.seq_len, uni.n_input))

        rnn = tf.keras.layers.RNN(cell, return_sequences=True)

        initial_state = cell.zero_state(cfg['train'].batch_size)
        rnn_output = rnn(inputs, initial_state=initial_state)
        regularization_layer = SpikeRegularization(
            cell,
            uni.target_rate,
            uni.rate_cost
        )
        voltages, spikes = regularization_layer(rnn_output)
        voltages = tf.identity(voltages, name='voltages')
        spikes = tf.identity(spikes, name=self.spike_identity)

        weighted_out_projection = tf.keras.layers.Dense(1)
        weighted_out = weighted_out_projection(spikes)

        prediction = exp_convolve(weighted_out, axis=1)
        prediction = tf.identity(prediction, name='output')

        return tf.keras.Model(
            inputs=inputs,
            outputs=[voltages, spikes, prediction]
        )
