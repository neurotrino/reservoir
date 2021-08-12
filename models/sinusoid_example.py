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

    def __init__(self, cfg, cell):
        """ ... """
        super().__init__()

        # define cell
        self.cell = cell  # [!] holdover from old HSJON init

        # Layer definitions
        self.rnn1 = tf.keras.layers.RNN(cell, return_sequences=True)
        self.dense1 = tf.keras.layers.Dense(1)


    def call(self, inputs, training=False):
        """ ... """

        # [!] is okay that I got rid of tf.identity for the outputs?
        # [!] is it a problem that I'm putting cell.initial_state here?
        voltages, spikes = self.rnn1(
            inputs,
            initial_state=self.cell.zero_state(cfg['train'].batch_size)
        )
        prediction = self.dense1(spikes)
        prediction = exp_convolve(prediction, axis=1)

        return voltages, spikes, prediction
