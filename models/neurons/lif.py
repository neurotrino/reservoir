"""Leaky integrate-and-fire (LIF) neuron implementations."""

# external modules
import numpy as np
import tensorflow as tf

# internal modules
from models.neurons.base import BaseNeuron
from utils.connmat import ConnectivityMatrixGenerator as CMG
from utils.connmat import ExInConnectivityMatrixGenerator as ExInCMG

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Leaky Integrate-and-Fire (LIF) Neuron                                     │
#┴───────────────────────────────────────────────────────────────────────────╯

class _LIFCore(BaseNeuron):
    """Layer of leaky integrate-and-fire neurons.

    All other neurons in the lif.py module inherit from this class.

    Configuration Parameters:
        rewiring - enable/disable rewiring
        tau - parameter used in signal decay calculations
        units - number of neurons in the layer

    Public Attributes:
        state_size - layer dimensions

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
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg, cfg_key='cell'):
        super().__init__()

        self.cfg = cfg
        cell_cfg = cfg[cfg_key]

        # Configuration parameters
        self.EL = cell_cfg.EL
        self.tau = cell_cfg.tau
        self.thr = cell_cfg.thr
        self.units = cell_cfg.units
        self.mu = cell_cfg.mu # [?] check if all LIF should have this
        self.sigma = cell_cfg.sigma # [?] check if all LIF should have this
        self.rewiring = cell_cfg.rewiring # [?] check if all cells should have this
        self.p = cell_cfg.p  # [?] check if all LIF/cells should have this
        # TODO: move `p` to BaseNeuron and inherit

        # Derived attributes
        self._decay = tf.exp(-cfg['misc'].dt / self.tau)

        # Other attributes
        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        # (voltage, refractory, previous_spikes)
        self.state_size = tuple([cfg['cell'].units] * 3)

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
            trainable=True,
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
        if self.rewiring:
            # Store using +1 for excitatory, -1 for inhibitory
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype = tf.float32)
        else:
            # Store using 0 for
            # zerosself.rec_sign = tf.sign(self.recurrent_weights)
            #self.rec_sign = tf.sign(self.recurrent_weights)
            self.rec_sign = tf.sign(self.recurrent_weights)


    def call(self, inputs, state):
        """
        TODO: method docs (forward pass, &c | 'call' not '__call__ bc keras')
        """
        [old_v, old_r, old_z] = state[:3]

        if self.rewiring:
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

    def zero_state(self, batch_size, dtype=tf.dtypes.float32):
        """TODO: docs."""
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return v0, r0, z_buf0  # voltage, refractory, spike

class LIF(_LIFCore):
    def build(self, input_shape):
        super().build(
            input_shape,
            CMG(
                self.units,
                self.p,
                self.cfg['misc'].mu, self.cfg['misc'].sigma
            )
        )

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory LIF Neuron                                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInLIF(_LIFCore):
    """TODO: docs, emphasizing difference from _LIFCore"""

    # base template from October 16th, 2020 version of LIFCell

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        """ExInLIF layers are initialized to track the number of
        excitatory and inhibitory cells in the layer, in addition to
        the core initialization properties inherent to a LIF cell.
        """
        super().__init__(cfg)

        # Number of excitatory and inhibitory neurons in the layer
        self.n_excite = int(cfg['cell'].frac_e * self.cfg['cell'].units)
        self.n_inhib = self.cfg['cell'].units - self.n_excite

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape):
        """TODO: docs"""
        super().build(
            input_shape,
            ExInCMG(
                self.n_excite, self.n_inhib,
                self.p.ee, self.p.ei, self.p.ie, self.p.ii,
                self.mu, self.sigma
            )
        )

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory Adaptive LIF (ALIF) Neuron                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInALIF(_LIFCore):
    """Layer of adaptive leaky integrate-and-fire neurons containing
    both excitatory and inhibitory connections.

    Configuration Parameters:
        p - set of four values (p.ee, p.ei, p.ie, p.ii) determining
            connection probablities between and within excitatory and
            inhibitory neurons
        tau_adaptation -
        beta -

    Public Attributes:
        ecount - number of excitatory neurons in the layer
        icount - number of inhibitory neurons in the layer
        decay_b -
        state_size -
    """
    # base template from October 22nd, 2020 version of LIF_EI

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        """
        TODO: method docs
        """
        super().__init__(cfg)  # core LIF attributes and initialization

        # ExIn paramaters
        self.n_excite = int(frac_e * cfg['cell'].units)
        self.n_inhib = self.units - self.n_excite

        # Adaptation parameters
        self.decay_b = tf.exp(-cfg['misc'].dt / cfg['cell'].tau_adaptation)

        # voltage, refractory, adaptation, prior spikes
        self.state_size = tuple([self.units] * 4)

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
            self.cfg['misc'].mu,
            self.cfg['misc'].sigma
        )
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # Store neurons' signs
        if self.rewiring:
            # +1 for excitatory and -1 for inhibitory
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.n_excite,:] = -1 * wmat[0:self.n_excite,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype=tf.float32)
        else:
            # as above but 0 for zeros
            self.rec_sign = tf.sign(self.recurrent_weights)

        super().build(input_shape, connmat_generator)


    def call(self, inputs, state):
        """TODO: docs"""
        [old_v, old_r, old_b, old_z] = state[:4]

        if self.rewiring:
            # Make sure all self-connections remain 0
            self.recurrent_weights.assign(tf.where(
                self.disconnect_mask, tf.zeros_like(self.recurrent_weights),
                self.recurrent_weights
            ))

        # If the sign of a weight changed from the original or the
        # weight is no longer 0, make the weight 0
        #
        # i.e. keep all Is as Is and all Es as Es
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
