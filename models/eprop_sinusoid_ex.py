"""TODO: module docs"""

# external ----
from typing import Any

import tensorflow as tf

# internal ----
from models.common import BaseModel, SpikeRegularization, exp_convolve
from models.neurons.lif import *
#from models.neurons.adex import *
from utils.config import subconfig
from utils.misc import SwitchedDecorator

DEBUG_MODE = False

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE

@tf.custom_gradient  # Taken from Bellec et al. (2020) code
def matmul_random_feedback(filtered_z, W_out_arg, B_out_arg):
    """Document this"""
    logits = tf.einsum('btj,jk->btk', filtered_z, W_out_arg)
    def grad(dy):
        dloss_dW_out = tf.einsum('bij,bik->jk', filtered_z, dy)
        dloss_dfiltered_z = tf.einsum('bik,jk->bij', dy, B_out_arg)
        dloss_db_out = tf.zeros_like(B_out_arg)
        return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]
    return logits, grad


class Model(BaseModel):
    """Model for the sinusoid-matching example."""

    def __init__(self, cfg):
        """ ... """
        super().__init__()

        # Attribute assignments
        self.cfg = cfg

        self.target_rate = cfg['train'].target_rate
        self.rate_cost = cfg['train'].rate_cost

        cell_type = eval(cfg['model'].cell.type)  # neuron (class)
        self.cell = cell_type(subconfig(          # neuron (object)
            cfg,
            cfg['model'].cell,
            old_label='model',
            new_label='cell'
        ))
        logging.info(f"cell type set to {cell_type.__name__}")

        # Layer definitions
        self.rnn1 = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.reg1 = SpikeRegularization(
            self.cell,
            self.target_rate,
            self.rate_cost
        )
        self.layers = [  # gather in a list for later convenience
            self.rnn1,
            self.reg1
        ]


    @switched_tf_function
    def call(self, inputs, training=False):
        """ ... """

        # Basic forward pass:
        x = self.rnn1(
            inputs,
            initial_state=self.cell.zero_state(self.cfg['train'].batch_size)
        )
        voltages, spikes = self.reg1(x)

        # Additional step(s):
        out_initializer = tf.keras.initializers.GlorotUniform()
        out_values = out_initializer(shape=(self.cfg['model'].cell.units, 1))
        W_out = out_values#tf.Variable(out_values, name='out_weight', trainable=True)

        # decay ~0.95 as in Bellec et al. (2020) where they use
        # np.exp(-dt/tau_out) with tau_out is between 15 and 30 ms
        filtered_Z = exp_convolve(spikes, decay=np.exp(-1/20))

        B_out = tf.constant(
            np.random.standard_normal((self.cfg['model'].cell.units, 1)),
            dtype=tf.float32,
            name='feedback_weights'
        )

        # Assuming the shape of the spikes (and filtered_Z) is
        # batch_size x seq_len x cfg['model'].cell.units

        # Prediction should have the shape batch_size x seq_len x 1
        # Note: prediction and loss computation is different than how
        # Bellec et al. do it but it should work for this task
        prediction = matmul_random_feedback(filtered_Z, W_out, B_out)
        prediction = exp_convolve(prediction, axis=1)

        return voltages, spikes, prediction
