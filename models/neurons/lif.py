"""
TODO: module docs
"""

from typing import Any, Dict, Union

import numpy as np
import tensorflow as tf

# local
from models.neurons.base import BaseNeuron
from utils.connmat import ConnectivityMatrixGenerator as CMG
from utils.connmat import ExInConnectivityMatrixGenerator as ExInCMG

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Core Properties                                                           │
#┴───────────────────────────────────────────────────────────────────────────╯

class _LIFCore(BaseNeuron):
    """TODO: docs

    Core properties/behaviors common to all LIF neurons in this file
    (though the methods seen here might be overwritten or modified in
    child classes)
    """

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(
        self,
        uni: Any,
        units: int,            # number of neurons in the model
        thr: float,            # thr
        EL: float,             #
        n_refrac: int,         #
        tau: float,            #
        dampening_factor: Any, #
        p: Union[float, Dict[str, float]]  # connectivity
    ):
        super().__init__()

        self.uni = uni
        self.units = units
        self.thr = thr
        self.EL = EL
        self.n_refrac = n_refrac
        self.tau = tau
        self.dt = float(uni.dt)
        self.dampening_factor = dampening_factor

        self._decay = tf.exp(-self.dt / tau)
        self._n_refrac = n_refrac
        self._dampening_factor = dampening_factor
        self.p = p

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        # (voltage, refractory, previous_spikes)
        self.state_size = (units, units, units)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape, connmat_generator):
        """TODO: docs"""
        # using uniform weight dist for inputs as opposed to
        #
        # ```
        # RandomNormal(
        #     mean=1.0,
        #     stddev=1.0 / np.sqrt(input_shape[-1] + self.units)
        # )
        # self.input_weights = self.add_weight(
        #     shape=(input_shape[-1], self.units),
        #     initializer=tf.keras.initializers.RandomNormal(
        #         stddev=1.0 / np.sqrt(input_shape[-1] + self.units)
        #     ),
        #     name='input_weights'
        # )
        # ```
        uni = self.uni

        self.input_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0,
                maxval=0.4
            ),
            trainable=True,
            name='input_weights'
        )

        # Disconnect self-recurrent weights
        self.disconnect_mask = tf.cast(
            np.diag(np.ones(self.units, dtype=np.bool)),
            tf.bool
        )

        #self.recurrent_weights = self.add_weight(
        #    shape=(self.units, self.units),
        #    initializer = LogNormal(self.mu, uni.sigma, self.units, self.p),
        #    trainable=True,
        #    name='recurrent_weights'
        #)

        self.recurrent_weights = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(gain=0.7),
            #initializer = tf.keras.initializers.RandomNormal();
            trainable = True,
            name='recurrent_weights'
        )

        # 2020-Nov-07: works with CMG's lognormal weight specification
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # not currently using bias currents
        #self.bias_currents = self.add_weight(
        #    shape=(self.units,),
        #    initializer=tf.keras.initializers.Zeros(),
        #    name='bias_currents'
        #)

        # Store neurons' signs
        if uni.rewiring:
            # Store using +1 for excitatory, -1 for inhibitory
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype = tf.float32)
        else:
            # Store using 0 for
            # zerosself.rec_sign = tf.sign(self.recurrent_weights)
            self.rec_sign = tf.sign(self.recurrent_weights)

        super().build(input_shape)

    def call(self, inputs, state):
        """
        TODO: method docs (forward pass, &c | 'call' not '__call__ bc keras')
        """
        [old_v, old_r, old_z] = state[:3]

        if self.uni.rewiring:
            # Make sure all self-connections remain 0
            self.recurrent_weights.assign(tf.where(
                self.disconnect_mask,
                tf.zeros_like(self.recurrent_weights),
                self.recurrent_weights
            ))

        # If the sign of a weight changed from the original unit's
        # designation or the weight is no longer 0, make it 0
        self.recurrent_weights.assign(tf.where(
            self.rec_sign * self.recurrent_weights > 0,
            self.recurrent_weights,
            0
        ))

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, self.recurrent_weights)

        # to circumvent the problem of voltage reset, we have a
        # subtractive current applied if a spike occurred in previous
        # time step i_reset = -self.thr * old_z # in the
        # toy-valued case, we can just subtract thr which was 1,
        # to return to baseline 0, or approximately baseline now to
        # have the analogous behavior using real voltage values, we
        # must subtract the difference between thr and EL
        i_reset = -(self.thr - self.EL) * old_z
        # ^ approx driving the voltage 20 mV more negative

        input_current = i_in + i_rec + i_reset #+ self.bias_currents[None]

        # previously, whether old_v was below or above 0, you would
        # still decay gradually back to 0 decay was dependent on the
        # distance between your old voltage and resting 0 the equation
        # was simply new_v = self._decay * old_v + input_current we are
        # now writing it with the same concept: the decay is dependent
        # on the distance between old voltage and rest at -70mV that
        # decay is then added to the resting value in the same way that
        # decay was previously implicitly added to 0 (rest) this
        # ensures the same basic behavior s.t. if you're above EL, you
        # hyperpolarize to EL and if you are below EL, you depolarize
        # to EL
        new_v = self.EL + self._decay * (old_v - self.EL) + input_current

        is_refractory = tf.greater(old_r, 0)
        #v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.thr - new_v) / (self.thr - self.EL)
        new_z = self.spike_function(v_scaled, self._dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(
            old_r - 1 + tf.cast(new_z * self._n_refrac, tf.int32),
            0,
            self._n_refrac
        )

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.float32):
        """TODO: docs."""
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return v0, r0, z_buf0  # voltage, refractory, spike

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Leaky Integrate-and-Fire (LIF) Neuron                                     │
#┴───────────────────────────────────────────────────────────────────────────╯

class LIF(_LIFCore):
    """
    TODO: formalize class docs

    October 16th, 2020:
        In this version of LIF, when a synaptic weight changes sign
        from its initialization, it goes to zero. However, it need not
        remain at zero. It is allowed to change in the direction of its
        initial sign. Due to the initialization with a lognormal
        distribution, all synapses here are excitatory. Only synapses
        that begin at 0 must remain at 0. These include self-recurrent
        connections. Therefore connectivity can never go above initial
        p, but it can go below it. We have not yet implemented
        rewiring. That will come with its own considerations.
    """

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape):
        """Build from _LIFCore using the standard CMG"""
        uni = self.uni
        super().build(input_shape, CMG(self.units, self.p, uni.mu, uni.sigma))

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory LIF Neuron                                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInLIF(_LIFCore):
    """TODO: docs, emphasizing differenct from _LIFCore"""

    # base template from October 16th, 2020 version of LIFCell

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, frac_e, *args, **kwargs):
        """
        TODO: method docs
        """
        super().__init__(*args, **kwargs)

        self.n_excite = int(frac_e * self.units)
        self.n_inhib = self.units - self.n_excite
        self.p_ee = self.p['ee']
        self.p_ei = self.p['ei']
        self.p_ie = self.p['ie']
        self.p_ii = self.p['ii']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape):
        """TODO: docs"""
        super().build(
            input_shape,
            ExInCMG(
                self.n_excite, self.n_inhib,
                self.p_ee, self.p_ei, self.p_ie, self.p_ii,
                self.uni.mu, self.uni.sigma
            )
        )

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory Adaptive LIF (ALIF) Neuron                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInALIF(_LIFCore):
    """TODO: docs"""
    # base template from October 22nd, 2020 version of LIF_EI

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, frac_e, beta, tau_adaptation, *args, **kwargs):
        """
        TODO: method docs
        """
        super().__init__(*args, **kwargs)

        # ExIn paramaters
        self.n_excite = int(frac_e * self.units)
        self.n_inhib = self.units - self.n_excite
        self.p_ee = self.p['ee']
        self.p_ei = self.p['ei']
        self.p_ie = self.p['ie']
        self.p_ii = self.p['ii']

        # Adaptation parameters
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = tf.exp(-self.dt / tau_adaptation)

        # voltage, refractory, adaptation, prior spikes
        self.state_size = tuple([self.units]*4)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape):
        self.input_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0,
                maxval=0.4
            ),
            trainable = True,
            name='input_weights'
        )

        # disconnect self-recurrent weights
        self.disconnect_mask = tf.cast(
            np.diag(np.ones(self.units, dtype=np.bool)),
            tf.bool
        )

        # weights set here do not matter
        self.recurrent_weights = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(gain=0.7),
            trainable=True,
            name='recurrent_weights'
        )

        # weights are lognormal
        connmat_generator = ExInCMG(
            self.n_excite,
            self.n_inhib,
            self.p_ee,
            self.p_ei,
            self.p_ie,
            self.p_ii,
            self.uni.mu,
            self.uni.sigma
        )
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # Store neurons' signs
        if self.uni.rewiring:
            # +1 for excitatory and -1 for inhibitory
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype=tf.float32)
        else:
            # as above but 0 for zeros
            self.rec_sign = tf.sign(self.recurrent_weights)

        BaseNeuron().build(input_shape)  # TODO?: refactor for _LIFCore

    def call(self, inputs, state):
        """TODO: docs"""
        [old_v, old_r, old_b, old_z] = state[:4]

        if self.uni.rewiring:
            # Make sure all self-connections remain 0
            self.recurrent_weights.assign(tf.where(
                self.disconnect_mask, tf.zeros_like(self.recurrent_weights),
                self.recurrent_weights
            ))

        # If the sign of a weight changed from the original or the
        # weight is no longer 0, make the weight 0
        self.recurrent_weights.assign(tf.where(
            self.rec_sign * self.recurrent_weights > 0,
            self.recurrent_weights,
            0
        ))

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, self.recurrent_weights)

        # to circumvent the problem of voltage reset, we have a
        # subtractive current applied if a spike occurred in previous
        # time step i_reset = -self.thr * old_z # in the
        # toy-valued case, we can just subtract thr which was 1,
        # to return to baseline 0, or approximately baseline now to
        # have the analogous behavior using real voltage values, we
        # must subtract the difference between thr and EL
        i_reset = -(self.thr-self.EL) * old_z
        # ^ approx driving the voltage 20 mV more negative

        input_current = i_in + i_rec + i_reset # + self.bias_currents[None]

        # previously, whether old_v was below or above 0, you would
        # still decay gradually back to 0 decay was dependent on the
        # distance between your old voltage and resting 0 the equation
        # was simply new_v = self._decay * old_v + input_current we are
        # now writing it with the same concept: the decay is dependent
        # on the distance between old voltage and rest at -70mV that
        # decay is then added to the resting value in the same way that
        # decay was previously implicitly added to 0 (rest) this
        # ensures the same basic behavior s.t. if you're above EL, you
        # hyperpolarize to EL and if you are below EL, you depolarize
        # to EL
        new_v = self.EL + (self._decay) * (old_v - self.EL) + input_current

        is_refractory = tf.greater(old_r, 0)
        adaptive_thr = self.thr + old_b * self.beta
        v_scaled = -(adaptive_thr-new_v) / (adaptive_thr-self.EL)
        new_b = self.decay_b * old_b + old_z
        new_z = self.spike_function(v_scaled, self._dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(
            old_r - 1 + tf.cast(new_z * self._n_refrac, tf.int32),
            0,
            self._n_refrac)

        new_state = (new_v, new_r, new_b, new_z)
        output = (new_v, new_z)

        return output, new_state

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.float32):
        """TODO: docs."""
        v0, r0, z_buf0 = super().zero_state(batch_size, dtype)
        b0 = tf.zeros((batch_size, self.units), tf.float32)

        # voltage, refractory, spike, adaptive thr
        return v0, r0, b0, z_buf0
