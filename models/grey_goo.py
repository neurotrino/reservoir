"""TODO: module docs"""

# external
import tensorflow as tf
import tensorflow_probability as tfp

# internal
from models.common import *
from models.neurons.adex import *
from models.neurons.lif import *
from utils.config import subconfig

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

        # Layer definitions
        self.rnn1 = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.dense1 = tf.keras.layers.Dense(1)


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
