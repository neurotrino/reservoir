"""TODO: module docs
"""

import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Core Neuron                                                               │
#┴───────────────────────────────────────────────────────────────────────────╯

class Neuron(tf.keras.layers.Layer):
    """Parent class for all neuron variants."""

    def pseudo_derivative(self, v_scaled, dampening_factor):
        """TODO: docs"""
        return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

    @tf.custom_gradient
    def spike_function(self, v_scaled, dampening_factor):
        """TODO: more docs?

        :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
        we must ensure our membrane dynamics (with negative real valued thresholds, etc.) remains consistent with this voltage-scaling spike mechanic
        in this case, we normalize using -(thr-V)/(thr-EL), which is a variation on the way one would normalize x between 0 and 1 using (x-min)/(max-min)
        (it would be a case of -(max-x)/(max-min)
        :param dampening_factor: parameter to stabilize learning
        """
        z_ = tf.greater(v_scaled, 0.) # returns bool of whether v_scaled is above thr or not, since it would be equal to 0 at thr
        z_ = tf.cast(z_, tf.float32) # cast as number [0, 1]

        def grad(dy):
            de_dz = dy
            dz_dv_scaled = self.pseudo_derivative(v_scaled, dampening_factor)

            de_dv_scaled = de_dz * dz_dv_scaled

            return [de_dv_scaled, tf.zeros_like(dampening_factor)]

        return tf.identity(z_, name="spike_function"), grad

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Modifiers                                                                 │
#┴───────────────────────────────────────────────────────────────────────────╯

class ExIn:
    """Excitatory/inhibitory modifier to the neuron class.

    Neurons with a mix of excitatory and inhibitory synapses should
    inherit from this class **in addition to** the `Neuron` class.

    All ExIn neurons have `ex_mask` and `in_mask` attributes which,
    when used with `tf.boolean_mask()` allow focused access to the
    excitatory or inhibitory synapses in the layer's weight matrix,
    respectively.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # Connection probabilities between neurons (excitatory to
        # excitatory, excitatory to inhbitory, and so on)
        self.p_ee = cfg['cell'].p_ee
        self.p_ei = cfg['cell'].p_ei
        self.p_ie = cfg['cell'].p_ie
        self.p_ii = cfg['cell'].p_ii

        # Number of excitatory and inhibitory neurons in the layer
        self.num_ex = int(cfg['cell'].frac_e * self.cfg['cell'].units)
        self.num_in = self.cfg['cell'].units - self.num_ex

        # Masks enabling easy selection of either all the excitatory
        # or all the inhibitory neurons in the layer.
        #
        # Intended for use with tf.boolean_mask().
        #
        # The logic used to construct these masks, at the moment,
        # assumes that the first k neuron indices are for excitatory
        # synapses and the rest are for inhibitory synapses. This is
        # consistent with the rest of the codebase as of 2021-Aug-26,
        # but is not enforced programmatically, so be mindful.
        mask_shape = (cfg['cell'].units, cfg['cell'].units)
        self.ex_mask = np.zeros(mask_shape, dtype=bool)
        self.ex_mask[0:self.num_ex] = True
        self.in_mask = np.invert(self.ex_mask)
