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
        target_rate,
        rate_cost,
        cell: LIF
    ):
        super().__init__()

        self.target_rate = target_rate
        self.rate_cost = rate_cost

        # Sub-networks and layers
        self.cell = cell

    def build(self, cfg):
        """TODO: method docs"""
        # TODO
        cell = self.cell

        inputs = tf.keras.Input(shape=(cfg['data'].seq_len, cfg['data'].n_input))

        rnn = tf.keras.layers.RNN(cell, return_sequences=True)

        initial_state = cell.zero_state(cfg['train'].batch_size)
        rnn_output = rnn(inputs, initial_state=initial_state)
        regularization_layer = SpikeRegularization(
            cell,
            self.target_rate,
            self.rate_cost
        )
        voltages, spikes = regularization_layer(rnn_output)
        voltages = tf.identity(voltages, name='voltages')
        spikes = tf.identity(spikes, name='spikes')

        weighted_out_projection = tf.keras.layers.Dense(1)
        weighted_out = weighted_out_projection(spikes)

        prediction = exp_convolve(weighted_out, axis=1)
        prediction = tf.identity(prediction, name='output')

        return tf.keras.Model(
            inputs=inputs,
            outputs=[voltages, spikes, prediction]
        )
