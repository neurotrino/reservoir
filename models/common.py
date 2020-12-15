"""Common SNN components."""

from typing import Any

import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Base (Inherited) Model                                                    │
#┴───────────────────────────────────────────────────────────────────────────╯

# the base model inhereted by specific models in the models directory
# This value is currently just an alias
BaseModel = tf.keras.layers.Layer

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Regularization Layers                                                     │
#┴───────────────────────────────────────────────────────────────────────────╯

class SpikeRegularization(tf.keras.layers.Layer):
    """TODO: docs"""
    def __init__(self, cell, target_rate, rate_cost): # rate in spikes/ms for ease
        """TODO: docs"""
        super().__init__()
        self._rate_cost = rate_cost
        self._target_rate = target_rate
        self._cell = cell

    def call(self, inputs, **kwargs):
        """TODO: docs"""
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.thr

        rate = tf.reduce_mean(spike, axis=(0, 1))
        # av = Second * tf.reduce_mean(z, axis=(0, 1)) / flags.dt
        #regularization_coeff = tf.Variable(np.ones(flags.n_neurons) * flags.reg_fr, dtype=tf.float32, trainable=False)
        #loss_reg_fr = tf.reduce_sum(tf.square(rate - flags.target_rate) * regularization_coeff)
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        return inputs

class SpikeVoltageRegularization(tf.keras.layers.Layer):
    """TODO: docs"""
    def __init__(self, cell, rate_cost=.1, voltage_cost=.01, target_rate=.02): # rate in spikes/ms for ease
        """TODO: docs"""
        self._rate_cost = rate_cost
        self._voltage_cost = voltage_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        """TODO: docs"""
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.threshold

        rate = tf.reduce_mean(spike, axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        v_pos = tf.square(tf.clip_by_value(tf.nn.relu(voltage - upper_threshold), 0., 1.))
        v_neg = tf.square(tf.clip_by_value(tf.nn.relu(-voltage - self._cell.threshold), 0., 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        self.add_loss(voltage_loss)
        self.add_metric(voltage_loss, name='voltage_loss', aggregation='mean')
        return inputs

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ ...                                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯

def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    """TODO: docs"""
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered
