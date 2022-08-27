"""Adaptive Purpose Model, Type II"""

# external
import tensorflow as tf

from models.common import exp_convolve

# from models.neurons.adex import *
from models.neurons.lif import *


class Gemini(tf.keras.Model):
    """SNN model designed for multi-task learning."""

    def __init__(self, cfg):
        """Create a new Gemini model."""
        super(Gemini, self).__init__()
        self.cfg = cfg

        # Setup
        cell_type = eval(cfg["model"].cell_type)
        self.cell = cell_type(cfg)

        # Layer definitions
        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.weighted_projection_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """Complete a pass through the model."""

        # Begin forward pass
        x = self.rnn_layer(
            inputs,
            initial_state=self.cell.zero_state(self.cfg["train"].batch_size),
        )

        # Mid-processing outputs
        voltages = tf.identity(
            x[0], name="voltages"
        )  # is it right to make these three tf.identities?
        spikes = tf.identity(x[1], name="spikes")

        # Continue forward pass
        x = self.weighted_projection_layer(spikes)

        # Final outputs
        prediction = tf.identity(exp_convolve(x, axis=1), name="output")

        return voltages, spikes, prediction
