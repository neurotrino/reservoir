"""TODO: module docs"""

from typing import Any, Dict, Union

import tensorflow as tf
import tensorflow.keras.initializers as kinits

import logging

import numpy as np

# local
from models.neurons.base import BaseNeuron
from utils.connmat import ConnectivityMatrixGenerator as CMG
from utils.connmat import ExInConnectivityMatrixGenerator as ExInCMG

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Core Properties                                                           │
#┴───────────────────────────────────────────────────────────────────────────╯

class _AdExCore(BaseNeuron):
    """TODO: docs"""

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg

        self.rewiring = cfg['cell'].rewiring
        self.units = cfg['cell'].units
        self.thr = cfg['cell'].thr
        self.EL = cfg['cell'].EL
        self.n_refrac = cfg['cell'].n_refrac
        self.dampening_factor = cfg['cell'].dampening_factor
        self.tauw = cfg['cell'].tauw
        self.a = cfg['cell'].a
        self.b = cfg['cell'].b
        self.gL = cfg['cell'].gL
        self.C = cfg['cell'].C
        self.deltaT = cfg['cell'].deltaT
        self.V_reset = cfg['cell'].V_reset

        if self.tauw is None:
            raise ValueError("time constant must be set for adaptive bias")
        if self.a is None:
            raise ValueError("parameter 'a' must be set for adaptive bias ")

        self._dt = float(cfg['misc'].dt)

        self.dt_gL__C = self._dt * self.gL / self.C
        self.dt_a__tauw = self._dt * self.a / self.tauw

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        #                  voltage,    refractory, adaptation, spikes (spiking or not)
        self.state_size = (self.units, self.units, self.units, self.units)
        #                  voltage,     spikes
        self.output_size = (self.units, self.units)


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape, connmat_generator, initializers):
        # Create the input weights which should be of the form:
        #   np.array([[input1toN1, ..., input1toNn], ..., [inputktoN1, ..., inputktoNn]], dtype=np.float32)
        # Not sure why this choice of distribution; included also uniform used in LIFCell model
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        '''
        self.input_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers['input_weights'],
            trainable=True,
            name='input_weights'
        )

        # Create the recurrent weights, their value here is not important
        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 trainable=True,
                                                 name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses

        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # Store neurons' signs
        #try:
        if self.rewiring:
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype = tf.float32) # +1 for excitatory and -1 for inhibitory
        else:
            self.rec_sign = tf.sign(self.recurrent_weights) # as above but 0 for zeros
        #except:
        #    self.rec_sign = tf.sign(self.recurrent_weights) # as above but 0 for zeros
        #    logging.warn(f'neuronal sign storage defaulted in {type(self)}')

        # Needed to disconnect self-connections if self.rewiring
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)),tf.bool)

        # Bias_currents; commented out because we are not using it and it might affect the way I am assigning the weights
        # self.bias_currents = self.add_weight(shape=(self.units,),
        #                                      initializer=tf.keras.initializers.Zeros(),
        #                                      name='bias_currents')

        super().build(input_shape)

    def call(self, inputs, state):
        old_v, old_r, old_w, old_z = state[:4]  # old states

        """
        if self.rewiring:  # I believe this has been moved to the trainer
            # Make sure all self-connections are 0
            self.recurrent_weights.assign(tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights))
            # If the sign of a weight changed, make it 0
            self.recurrent_weights.assign(tf.where(self.rec_sign * self.recurrent_weights > 0, self.recurrent_weights, 0))
        else:
            # If the sign of a weight changed or the weight is no longer 0, make the weight 0
            self.recurrent_weights.assign(tf.where(self.rec_sign * self.recurrent_weights > 0, self.recurrent_weights, 0))
        """

        # Calculate input current
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, self.recurrent_weights)
        # There is no reset current because we are setting new_V to V_reset if old_z > 0.5
        i_t = i_in + i_rec  # + self.bias_currents[None]

        # Update voltage
        # Note: put inside new_v because wasn't sure about stopping gradient: exp_terms = tf.clip_by_value(tf.exp((old_v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)  # These min and max values were taken from tf1
        new_v = old_v - (self.dt_gL__C * (old_v - self.EL)) + (self.dt_gL__C * self.deltaT * tf.clip_by_value(tf.exp((old_v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)) + ((i_t - old_w) * self._dt / self.C)
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)

        # Update adaptation term
        new_w = old_w - ((self._dt / self.tauw) * old_w) + (self.dt_a__tauw * (old_v - self.EL))
        new_w += self.b * old_z

        # Determine if the neuron is spiking
        is_refractory = tf.greater(old_r, 0)
        # v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.thr-new_v) / (self.thr-self.EL)
        new_z = self.spike_function(v_scaled, self.dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Determine if the neuron is in a refractory period (and remaining time if yes)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self.n_refrac, tf.int32), 0, self.n_refrac)

        # New state and output
        new_state = (new_v, new_r, new_w, new_z)
        output = (new_v, new_z)

        return output, new_state


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.float32):  # is this not in BaseNeuron? does it differ from LIF?
        # Voltage (all at EL)
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL  # Do we want to start with random V?
        # Refractory (all 0)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # Adaptation (all 0)
        w0 = tf.zeros((batch_size, self.units), tf.float32)
        # Spike (all not spiking)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return [v0, r0, w0, z_buf0]


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Adaptive Exponential Integrate-and-Fire (AdEx) Neuron                     │
#┴───────────────────────────────────────────────────────────────────────────╯

class AdEx(_AdExCore):
    def build(self, input_shape):
        misc_cfg = self.cfg['misc']
        super().build(
            input_shape,
            CMG(self.units, self.p, misc_cfg.mu, misc_cfg.sigma),
            initializers={
                'input_weights': kinits.RandomUniform(minval=0.0, maxval=0.4)
            }
        )


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory AdEx Neuron                                         │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInAdEx(_AdExCore):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_excite = int(self.cfg['cell'].frac_e * self.units)
        self.n_inhib = self.units - self.n_excite
        self.p_ee = self.cfg['cell'].p_ee
        self.p_ei = self.cfg['cell'].p_ei
        self.p_ie = self.cfg['cell'].p_ie
        self.p_ii = self.cfg['cell'].p_ii


    def build(self, input_shape):
        super().build(
            input_shape,
            connmat_generator=ExInCMG(
                self.n_excite, self.n_inhib,
                self.p_ee, self.p_ei, self.p_ie, self.p_ii,
                self.cfg['cell'].mu, self.cfg['cell'].sigma
            ),
            initializers={
                'input_weights': kinits.RandomUniform(minval=0.0, maxval=0.4)
            }
        )


class _EligAdexCore(BaseNeuron):  # how is this different than _AdexCore
    """TODO: docs"""

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        if tauw is None:
            raise ValueError("Time constant for adaptive bias must be set.")
        if a is None:
            raise ValueError("a parameter for adaptive bias must be set.")

        super().__init__()
        self.cfg = cfg
        self._dt = float(cfg['misc'].dt)

        self.units = units
        self.thr = thr
        self.n_refrac = n_refrac
        self.dampening_factor = dampening_factor
        self.tauw = tauw
        self.a = a
        self.b = b
        self.gL = gL
        self.EL = EL
        self.C = C
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.p = p
        self.dt_gL__C = self._dt * self.gL / self.C
        self.dt_a__tauw = self._dt * self.a / self.tauw

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None
        self.rewiring = rewiring

        #                  voltage,    refractory, adaptation, spikes (spiking or not)
        self.state_size = (self.units, self.units, self.units, self.units)
        #                  voltage,     spikes
        self.output_size = (self.units, self.units)


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape, connmat_generator, initializers):
        # Create the input weights which should be of the form:
        #   np.array([[input1toN1, ..., input1toNn], ..., [inputktoN1, ..., inputktoNn]], dtype=np.float32)
        # Not sure why this choice of distribution; included also uniform used in LIFCell model
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        '''
        self.input_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers['input_weights'],
            trainable=True,
            name='input_weights'
        )

        # Create the recurrent weights, their value here is not important
        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 trainable=True,
                                                 name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses

        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # Store neurons' signs
        #try:
        if self.rewiring:
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype = tf.float32) # +1 for excitatory and -1 for inhibitory
        else:
            self.rec_sign = tf.sign(self.recurrent_weights) # as above but 0 for zeros
        #except:
        #    self.rec_sign = tf.sign(self.recurrent_weights) # as above but 0 for zeros
        #    logging.warn(f'neuronal sign storage defaulted in {type(self)}')

        # Needed to disconnect self-connections if self.rewiring
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)),tf.bool)

        # Bias_currents; commented out because we are not using it and it might affect the way I am assigning the weights
        # self.bias_currents = self.add_weight(shape=(self.units,),
        #                                      initializer=tf.keras.initializers.Zeros(),
        #                                      name='bias_currents')

        super().build(input_shape)


    def call(self, inputs, state):  # seems to contain a lot of duplicate code

        # Old states
        old_v, old_r, old_w, old_z = state[:4]

        if self.rewiring:
            # Make sure all self-connections are 0
            self.recurrent_weights.assign(tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights))
            # If the sign of a weight changed, make it 0
            self.recurrent_weights.assign(tf.where(self.rec_sign * self.recurrent_weights > 0, self.recurrent_weights, 0))
        else:
            # If the sign of a weight changed or the weight is no longer 0, make the weight 0
            self.recurrent_weights.assign(tf.where(self.rec_sign * self.recurrent_weights > 0, self.recurrent_weights, 0))

        # This is where eligibility traces and stop gradients should be used: stop gradients from flowing along old_z
        # Gradient --> Computes the gradient of the loss with respect to the trainable variables using operations recorded in context of this tape, i.e. all of this
        # Stop gradient --> If you insert this op in the graph its inputs are masked from the gradient generator. They are not taken into account for computing gradients.
        # I think, the adaptation should use old_z before stop_gradient-- Update adaptation term
        new_w = old_w - ((self._dt / self.tauw) * old_w) + (self.dt_a__tauw * (old_v - self.EL))
        new_w += self.b * old_z
        old_z = tf.stop_gradient(old_z)

        # Calculate input current
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, self.recurrent_weights)
        # There is no reset current because we are setting new_V to V_reset if old_z > 0.5
        i_t = i_in + i_rec  # + self.bias_currents[None]

        # Update voltage;
        # Note: Moved this to inside new_v because wasn't sure about stopping gradient at it: exp_terms = tf.clip_by_value(tf.exp((old_v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)  # These min and max values were taken from tf1
        new_v = old_v - (self.dt_gL__C * (old_v - self.EL)) + (self.dt_gL__C * self.deltaT * tf.clip_by_value(tf.exp((old_v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)) + ((i_t - old_w) * self._dt / self.C)
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)

        # Determine if the neuron is spiking
        is_refractory = tf.greater(old_r, 0)
        # v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.thr-new_v) / (self.thr-self.EL)
        new_z = self.spike_function(v_scaled, self.dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Determine if the neuron is in a refractory period (and remaining time if yes)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self.n_refrac, tf.int32), 0, self.n_refrac)

        # New state and output
        new_state = (new_v, new_r, new_w, new_z)
        output = (new_v, new_z)

        return output, new_state


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.float32):
        # Voltage (all at EL)
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL  # Do we want to start with random V?
        # Refractory (all 0)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # Adaptation (all 0)
        w0 = tf.zeros((batch_size, self.units), tf.float32)
        # Spike (all not spiking)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return [v0, r0, w0, z_buf0]

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ E-prop Adaptive Exponential Integrate-and-Fire (AdEx) Neuron              │
#┴───────────────────────────────────────────────────────────────────────────╯

class EligAdEx(_EligAdexCore):
    def build(self, input_shape):
        super().build(
            input_shape,
            CMG(self.units, self.p, self.cfg['misc'].mu, self.cfg['misc'].sigma),
            initializers={
                'input_weights': kinits.RandomUniform(minval=0.0, maxval=0.4)
            }
        )


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ E-prop Excitatory/Inhibitory AdEx Neuron                                  │
#┴───────────────────────────────────────────────────────────────────────────╯

class EligExInAdEx(_EligAdexCore):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_excite = int(frac_e * self.units)
        self.n_inhib = self.units - self.n_excite
        self.p_ee = self.p_ee
        self.p_ei = self.p_ei
        self.p_ie = self.p_ie
        self.p_ii = self.p_ii


    def build(self, input_shape):
        super().build(
            input_shape,
            connmat_generator=ExInCMG(
                self.n_excite, self.n_inhib,
                self.p_ee, self.p_ei, self.p_ie, self.p_ii,
                self.cfg['misc'].mu, self.cfg['misc'].sigma
            ),
            initializers={
                # TODO?: minval/maxval might be good HJSON config items
                'input_weights': kinits.RandomUniform(minval=0.0, maxval=0.4)
            }
        )
