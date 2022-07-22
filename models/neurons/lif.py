"""Leaky integrate-and-fire (LIF) neuron implementations."""

# external modules
import numpy as np
import tensorflow as tf
import os

# internal modules
from models.neurons.base import ExIn, Neuron
from utils.stats import StatisticalDistribution

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Leaky Integrate-and-Fire (LIF) Neuron                                     │
#┴───────────────────────────────────────────────────────────────────────────╯

class LIF(Neuron):
    """Layer of leaky integrate-and-fire neurons.

    All other neurons in the lif.py module inherit from this class.

    Configuration Parameters:
        freewiring - enable/disable wiring without any constraints
        rewiring - when a synapse goes to 0, a random new synapse will form
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

    def __init__(self, cfg):
        Neuron.__init__(self, cfg)

        self.cfg = cfg
        cell_cfg = cfg['cell']

        # Configuration parameters
        self.EL = cell_cfg.EL
        self.tau = cell_cfg.tau
        self.thr = cell_cfg.thr

        # initial voltage distribution
        self.v0_sdist = StatisticalDistribution(
            tf.random.normal,
            mean=-65.0,  # [mV]
            stddev=5.0,  # [mV]
        )

        # Derived attributes
        self._decay = tf.exp(-cfg['misc'].dt / self.tau)

        # Other attributes
        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = tf.cast(
            np.diag(np.ones(self.units, dtype=np.bool)),
            tf.bool
        )

        # (voltage, refractory, previous_spikes)
        self.state_size = tuple([cfg['cell'].units] * 3)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Reserved Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def build(self, input_shape):
        """Create initial layer weights.

        Create layer weights the first time `.__call__()` is called.
        Layers weights for the LIF neuron {...docs...}
        """
        Neuron.build(self, input_shape)

        if self.cfg["cell"].specify_input:
            self.input_weights = self.add_weight(
                shape=(self.cfg["data"].n_input, self.units),
                initializer=tf.keras.initializers.Orthogonal(gain=0.7),
                trainable=self.cfg["train"].input_trainable,
                name="input_weights",
            )
            input_weights_mat = self.input_connmat_generator.run_generator()
            self.input_weights.assign(input_weights_mat)
        else:
            self.input_weights = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=0.0,
                    maxval=0.4
                ),
                trainable=True,
                name='input_weights'
            )
        # save initial input weights
        np.save(os.path.join(self.cfg['save'].exp_dir, "npz-data/input_preweights.npy"), self.input_weights.numpy())

        # Disconnect self-recurrent weights
        self.disconnect_mask = tf.cast(
            np.diag(np.ones(self.units, dtype=np.bool)),
            tf.bool
        )

        self.recurrent_weights = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(gain=0.7),
            trainable=True,
            name='recurrent_weights'
        )
        initial_weights_mat = self.connmat_generator.run_generator()
        # save initial recurrent weights
        np.save(os.path.join(self.cfg['save'].exp_dir, "npz-data/main_preweights.npy"), initial_weights_mat)

        if self.cfg["cell"].specify_input:
            # just set main weights, as input weights were set earlier
            self.recurrent_weights.assign_add(initial_weights_mat)
        else:
            # set both input and main layer weights
            self.set_weights([self.input_weights.value(), initial_weights_mat])

        # Store neurons' signs
        if self.freewiring:
            # Store using +1 for excitatory, -1 for inhibitory
            wmat = -1 * np.ones([self.units, self.units])
            wmat[0:self.num_ex,:] = -1 * wmat[0:self.num_ex,:]
            self.rec_sign = tf.convert_to_tensor(wmat, dtype=tf.float32)
        else:
            # as above but 0 for zeros
            self.rec_sign = tf.sign(self.recurrent_weights)


    def call(self, inputs, state):
        """
        TODO: method docs (forward pass, &c | 'call' not '__call__ bc keras')
        """
        [old_v, old_r, old_z] = state[:3]

        """
        # now (correctly) implemented in trainers/std_single_task.py
        if self.freewiring:
            # Make sure all self-connections remain 0
            self.recurrent_weights.assign(tf.where(
                self.disconnect_mask,
                tf.zeros_like(self.recurrent_weights),
                self.recurrent_weights
            ))

        # If the sign of a weight changed from the original unit's
        # designation or the weight is no longer 0, make it 0
        preweights = self.recurrent_weights
        self.recurrent_weights.assign(tf.where(
            self.rec_sign * self.recurrent_weights > 0,
            self.recurrent_weights,
            0
        ))
        """

        # If rewiring is permitted, then count new zeros
        # Create that same # of new connections (from post-update zero connections)
        """
        if self.rewiring:
            pre_zeros = tf.where(tf.equal(preweights, 0))
            #pre_zeros_ct = tf.cast(tf.size(pre_zeros)/2, tf.int32)
            post_zeros = tf.where(tf.equal(self.recurrent_weights, 0))
            #post_zeros_ct = tf.where(tf.size(post_zeros)/2, tf.int32)
            #new_zeros_ct = tf.subtract(post_zeros_ct, pre_zeros_ct)
            new_zeros_ct = tf.subtract(tf.shape(pre_zeros)[0],tf.shape(post_zeros)[0])
            if new_zeros_ct > 0:
                for i in range(0,new_zeros_ct): # for all new zeros
                    # randomly select a position from post_zeros (total possible zeros)
                    new_pos_idx = numpy.random.randint(0, tf.shape(post_zeros)[0])
                    # draw a new weight
                    new_w = numpy.random.lognormal(self.mu, self.sigma)
                    if post_zeros[new_pos_idx][0] >= self.num_ex:
                        # if inhib, make weight -10x
                        new_w = - new_w * 10
                    # reassign to self.recurrent_weights
                    self.recurrent_weights.assign(post_zeros[new_pos_idx], new_w)
        """

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
        new_z = self.spike_function(v_scaled, self.cfg['cell'].dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(
            old_r - 1 + tf.cast(new_z * self.cfg['cell'].n_refrac, tf.int32),
            0,
            self.cfg['cell'].n_refrac
        )

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.dtypes.float32):
        """TODO: docs."""
        sz = (batch_size, self.units)
        v0 = self.v0_sdist.sample(sz) + self.EL
        #v0 = tf.zeros(sz, dtype) + self.EL
        r0 = tf.zeros(sz, tf.int32)
        z_buf0 = tf.zeros(sz, tf.float32)
        return v0, r0, z_buf0  # voltage, refractory, spike

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory LIF Neuron                                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInLIF(ExIn, LIF):
    """LIF neuron with both excitatory and inhibitory synapses."""

    def __init__(self, cfg):
        LIF.__init__(self, cfg)
        ExIn.__init__(self, cfg)

    def build(self, input_shape):
        ExIn.build(self, input_shape)
        LIF.build(self, input_shape)


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Excitatory/Inhibitory Adaptive LIF (ALIF) Neuron                          │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExInALIF(ExIn, LIF):
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
    #┤ Keras Layer Methods                                                   │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        """Initializes like ExInLIF, but with adaptation parameters."""
        LIF.__init__(self, cfg)
        ExIn.__init__(self, cfg)

        # Adaptation parameters
        self.beta = cfg['cell'].beta
        self.decay_b = tf.exp(
            -cfg['misc'].dt / self.cfg['cell'].tau_adaptation
        )

        # voltage, refractory, adaptation, prior spikes
        self.state_size = tuple([self.units] * 4)


    def build(self, input_shape):
        """Built like LIF, but with an ExInCMG instead of a CMG."""
        ExIn.build(self, input_shape)
        LIF.build(self, input_shape)


    # [?] why do we pass state instead of maintaining w/ attributes?
    def call(self, inputs, state):
        """TODO: docs.

        This method is the primary area of distinction between the
        ExInALIF class and the LIF/ExInLIF classes.
        """
        [old_v, old_r, old_b, old_z] = state[:4]

        """
        # Now correctly implemented in trainer
        if self.freewiring:
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
        """

        i_in = tf.matmul(inputs, self.input_weights)  # input current
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
        new_z = self.spike_function(v_scaled, self.cfg['cell'].dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        if self.cfg['cell'].refrac_stop_grad:
            new_r = tf.stop_gradient(
                tf.clip_by_value(
                    old_r - 1 + tf.cast(new_z * self.cfg['cell'].n_refrac, tf.int32),
                    0,
                    self.cfg['cell'].n_refrac
                )
            )
        else:
            new_r = tf.clip_by_value(
                old_r - 1 + tf.cast(new_z * self.cfg['cell'].n_refrac, tf.int32),
                0,
                self.cfg['cell'].n_refrac
            )

        new_state = (new_v, new_r, new_b, new_z)
        output = (new_v, new_z)

        return output, new_state

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def zero_state(self, batch_size, dtype=tf.float32):
        """TODO: docs."""
        v0, r0, z_buf0 = LIF.zero_state(self, batch_size, dtype)
        b0 = tf.zeros((batch_size, self.units), tf.float32)

        # voltage, refractory, spike, adaptive thr
        return v0, r0, b0, z_buf0
