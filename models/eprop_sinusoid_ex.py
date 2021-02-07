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
        cell: EligExInAdEx
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

        # TODO Check if this gets the number of units from configs
        cell_cfg = cfg['model'].cell
        n_neurons = cell_cfg.units
        W_out = tf.Variable(name='out_weight', shape=(n_neurons, 2))
        filtered_Z = exp_convolve(spikes)  # they use for the decay dt/tau_out where tau_out is between 15 and 30 ms

        @tf.custom_gradient  # Taken from Bellec et al. (2020) code
        def matmul_random_feedback(filtered_z, W_out_arg, B_out_arg):
       	    # use this function to generate the random feedback path - the symmetric feedback W_out^T that would arise
            # from BPTT is replaced by a randomly generated matrix B_out
       	    logits = tf.einsum('btj,jk->btk', filtered_z, W_out_arg)
            def grad(dy):
       	        dloss_dW_out = tf.einsum('bij,bik->jk', filtered_z, dy)
               	dloss_dfiltered_z = tf.einsum('bik,jk->bij', dy, B_out_arg)
                dloss_db_out = tf.zeros_like(B_out_arg)
                return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]
            return logits, grad
        
        B_out = tf.constant(np.random.standard_normal((n_neurons, 2)), dtype=tf.float32, name='feedback_weights')
        out = matmul_random_feedback(filtered_Z, W_out, B_out)
        # TODO TAKE ONLY LAST OUTPUT FOR PREDICTION
        m = 5 # something, depends on task
        # TODO to figure it out, see shape of spikes --> shape of filtered_Z --> shape of out
        output_logits = out[:, -m:]
        
        # NEW PREDICTION
        prediction = tf.argmax(tf.reduce_mean(output_logits, axis=1), axis=1)
        prediction = tf.identity(prediction, name='output')

        # TODO I think for the sinusoid task, comparing the prediction to true should be enough
        # to calculat the loss in trainers.eprop_sinusoid_ex.py    
        # However, for classification tasks, we might want to consider doing like Bellect et al.
        # which is to take output the logits and in the loss function compute sparse softmax 
        # cross entropy between logits and labels then take the reduced mean as the loss.
        # Note: they seperate the classification loss from the regularization loss then sum them.

        return tf.keras.Model(
            inputs=inputs,
            outputs=[voltages, spikes, prediction]
        )
