"""TODO: module docs"""

# external
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

# local
from models.common import BaseModel, SpikeRegularization, SpikeVoltageRegularization, SynchronyRateVoltageRegularization, exp_convolve
from models.neurons.lif import *
from models.neurons.adex import *

class SinusoidSlayer(BaseModel):
    """Model for the sinusoid-matching example."""

    def __init__(self,
        target_rate,
        rate_cost,
        voltage_cost,
        target_synch,
        synch_cost,
        cell: LIF
    ):
        super().__init__()

        self.target_rate = target_rate
        self.rate_cost = rate_cost
        self.voltage_cost = voltage_cost
        self.target_synch = target_synch
        self.synch_cost = synch_cost

        # Sub-networks and layers
        self.cell = cell

    def build(self, cfg):
        cell = self.cell

        inputs = tf.keras.Input(
            shape=(cfg['data'].seq_len, cfg['data'].n_input)
        )

        rnn = tf.keras.layers.RNN(cell, return_sequences=True)

        initial_state = cell.zero_state(cfg['train'].batch_size)
        rnn_output = rnn(inputs, initial_state=initial_state)
        regularization_layer = SynchronyRateVoltageRegularization(
            cell,
            self.target_synch,
            self.synch_cost,
            self.target_rate,
            self.rate_cost,
            self.voltage_cost
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
