"""TODO: module docs

Backprop
"""

# external ----
import logging
import tensorflow as tf
import tensorflow_probability as tfp

# internal ----
from models.common import *
#from models.neurons.adex import *
from models.neurons.lif import *
from utils.config import subconfig
from utils.misc import SwitchedDecorator

DEBUG_MODE = True

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE

class Model(BaseModel):
    """Generic prototyping model designed to test new features and
    provide an example to people learning the research infrastructure.
    """

    def __init__(self, cfg):
        """ ... """
        super().__init__()

        # Attribute assignments
        self.cfg = cfg

        cell_type = eval(cfg['model'].cell.type)  # neuron (class)
        self.cell = cell_type(subconfig(          # neuron (object)
            cfg,
            cfg['model'].cell,
            old_label='model',
            new_label='cell'
        ))
        logging.info(f"cell type set to {cell_type.__name__}")

        if self.cfg["model"].likelihood_output:
            n_out = 2
        else:
            n_out = 1

        # Layer definitions
        self.rnn1 = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.dense1 = tf.keras.layers.Dense(self.n_out)
        self.dense1.trainable = self.cfg['train'].output_trainable

        self.layers = [  # gather in a list for later convenience
            self.rnn1,
            self.dense1
        ]


    @switched_tf_function
    def noise_weights(self, mean=1.0, stddev=0.1):
        """Add noise to the recurrent weights."""
        weights = self.rnn1.get_weights()

        iweights = weights[0]
        rweights = weights[1]

        gain_matrix = tf.clip_by_value(
            tf.random.normal(rweights.shape, mean, stddev), -1, 1
        )
        noised_weights = rweights * gain_matrix

        self.rnn1.set_weights([iweights, noised_weights])


    @switched_tf_function
    def call(self, inputs, training=False):
        """ ... """

        # [!] is it okay that I got rid of tf.identity for the outputs?
        # [!] is it a problem that I'm putting cell.initial_state here?
        voltages, spikes = self.rnn1(
            inputs,
            initial_state=self.cell.zero_state(self.cfg['train'].batch_size)
        )
        prediction = self.dense1(spikes)
        prediction = exp_convolve(prediction, axis=1)

        return voltages, spikes, prediction
