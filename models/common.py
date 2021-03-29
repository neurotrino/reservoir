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

class BaseModel(tf.keras.layers.Layer):
    """TODO: docs"""
    def __init__(self):
        super(BaseModel, self).__init__()


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Regularization Layers                                                     │
#┴───────────────────────────────────────────────────────────────────────────╯
"""
class BranchingRegularization(tf.keras.layers.Layer):
    # can be strung togther or later collapsed with other regularization layers
    # branching has previously been calculated on whole runs, but this layer acts in real time
    # input is all spikes for a particular timepoint in the rSNN
    # therefore it can be as simple as timept to timept comparison of spike numbers for instantaneous branching
    # or, rather than a layer here, it may be part of the loss equation - that is, if all spikes can be accessed.
    def __init__(self, cell, target_branch, branch_cost): # fano factor-like rapid synchrony measure
        super().__init__()
        self._branch_cost = branch_cost
        self._target_branch = target_branch
        self._cell = cell

    def call(self, inputs, **kwargs):
        spike = inputs[1]

        branching =
        self.add_metric(branching, name='branching', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(branching - self._target_branch)) * self._branch_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='branch_loss', aggregation='mean')

        return inputs

class SynchronyRegularization(tf.keras.layers.Layer):
    # can be strung togther or later collapsed with other regularization layers
    def __init__(self, cell, target_synch, synch_cost): # fano factor-like rapid synchrony measure
        super().__init__()
        self._synch_cost = synch_cost
        self._target_synch = target_synch
        self._cell = cell

    def call(self, inputs, **kwargs):
        spike = inputs[1]

        synchrony = fano_factor(self, spike)
        global_synchrony = tf.reduce_mean(synchrony)
        self.add_metric(global_synchrony, name='synchrony', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(synchrony - self._target_synch)) * self._synch_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='synch_loss', aggregation='mean')

        return inputs


def fano_factor(self, spike):

    #Calculate value similar to the Fano factor to estimate synchrony quickly
    #During each bin, calculate the variance of the number of spikes per neuron divided by the mean of the number of spikes per neuron
    #The Fano factor during one interval is equal to the mean of the values calculated for each bin in it
    #Spike should have dims of trial, neuron, time
    #Returned fano factor should have dims of trial


    # leaving this unfinished for now as accessible attributes and dimensions are unclear


    try:
        len_bins = 10
        n_fano = [0]*self.num_intervals
        len_interval = self.last_spiked/self.num_intervals
        n_bins = int(round(len_interval/len_bins))
        for m in range(0, self.num_intervals):
            fano_all = [0] * n_bins
            print(len_interval*m)
            for i in range(0, n_bins):
                spikes_per_neuron = [0]*(self.n_neurons-self.n_inhib)
                for j in range(0, self.n_neurons-self.n_inhib):
                    for k in self.spike_trains[j]:
                        if len_interval*m + len_bins*i <= k < len_interval*m + len_bins*(i+1):
                            spikes_per_neuron[j] += 1
                fano_all[i] = var(spikes_per_neuron)/mean(spikes_per_neuron)
            n_fano[m] = mean(fano_all)
        return n_fano


    try:
        len_bins = 10
        n_fano = [0]*(self.seq_len/len_bins)
        len_interval = self.seq_len/len_bins
        n_bins = int(round(len_interval/len_bins))
        fano_all = [0]*(self._cell.units)
            for i in range(0, self._cell.units):
                fano_all[i] = var(spike[i])/mean(spike[i])
        n_fano = mean(fano_all)
    return n_fano

    except Exception as e:
        print(e)
        return False

"""

class SpikeRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, target_rate, rate_cost):
        super().__init__()
        self._rate_cost = rate_cost
        self._target_rate = target_rate
        self._cell = cell

    def call(self, inputs, **kwargs):

        spike = inputs[1]

        # regularize for small durations of time for individual units

        
        unitwise_rates = tf.reduce_mean(spike, axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name = 'rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name = 'rate_loss', aggregation = 'mean')

        return inputs

class GlobalSpikeRegularization(tf.keras.layers.Layer):
    """TODO: docs"""
    def __init__(self, cell, target_rate, rate_cost): # rate in spikes/ms for ease
        """TODO: docs"""
        super().__init__()
        self._rate_cost = rate_cost
        self._target_rate = target_rate
        self._cell = cell

    def call(self, inputs, **kwargs):

        # it is unclear to me what self refers to here. is it cell? if so, does it
        # have the attributes of model: cell: in hjson? some of which conflict with/are redundant with uni?
        # for example, if I want to access the duration of a trial, am I able to call self.seq_len since
        # that is under uni, or is it not accessible?

        """TODO: docs"""
        # inputs should be in dims of trial (i.e. batch size) x neuron x time
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.thr

        rate = tf.reduce_mean(spike, axis=(0, 1))
        # av = Second * tf.reduce_mean(z, axis=(0, 1)) / flags.dt
        #regularization_coeff = tf.Variable(np.ones(flags.n_neurons) * flags.reg_fr, dtype=tf.float32, trainable=False)
        #loss_reg_fr = tf.reduce_sum(tf.square(rate - flags.target_rate) * regularization_coeff)
        global_rate = tf.reduce_mean(rate) # I think this is for the whole batch of trials?
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        return inputs # unchanged; facets of self however are changed

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
