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
        cell: EligAdEx
    ):
        super().__init__()

        self.target_rate = target_rate
        self.rate_cost = rate_cost

        # Sub-networks and layers
        self.cell = cell

    def build(self, cfg):
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

        cell_cfg = cfg['model'].cell
        n_neurons = cell_cfg.units
        out_initializer = tf.keras.initializers.GlorotUniform()
        out_values = out_initializer(shape=(n_neurons, 1))
        W_out = tf.Variable(out_values, name='out_weight', trainable=True)
        filtered_Z = exp_convolve(spikes, decay=np.exp(-1/20))  # decay ~0.95 as in Bellec et al. (2020) where they use np.exp(-dt/tau_out) with tau_out is between 15 and 30 ms

        @tf.custom_gradient  # Taken from Bellec et al. (2020) code
        def matmul_random_feedback(filtered_z, W_out_arg, B_out_arg):
       	    logits = tf.einsum('btj,jk->btk', filtered_z, W_out_arg)
            def grad(dy):
       	        dloss_dW_out = tf.einsum('bij,bik->jk', filtered_z, dy)
               	dloss_dfiltered_z = tf.einsum('bik,jk->bij', dy, B_out_arg)
                dloss_db_out = tf.zeros_like(B_out_arg)
                return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]
            return logits, grad
        
        B_out = tf.constant(np.random.standard_normal((n_neurons, 1)), dtype=tf.float32, name='feedback_weights')
        # Assuming the shape of the spikes (and filtered_Z) is batch_size x seq_len x n_neurons
        prediction = matmul_random_feedback(filtered_Z, W_out, B_out)
        # Prediction should have the shape batch_size x seq_len x 1
        prediction = tf.identity(prediction, name='output')
        # Note: prediction and loss computation is different than how Bellec et al. do it but it should work for this task

        return tf.keras.Model(
            inputs=inputs,
            outputs=[voltages, spikes, prediction]
        )
