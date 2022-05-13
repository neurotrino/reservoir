"""TODO: module docs

If you're not familiar with the rules of Python's multiple inheritance
and MRO, remember this rule of thumb: inherit in grammatical order and
then call __init__() in the opposite order, e.g.

```
# inheritance makes grammatical sense (ExIn + Neuron == ExIn Neuron)
class ExInALIF(ExIn, Neuron):

    # init goes in the opposite order
    def __init__(self, cfg):
        Neuron.__init__(self, cfg)
        ExIn.__init__(self, cfg)
```
"""
from utils.connmat import ConnectivityMatrixGenerator as CMG
from utils.connmat import ExInConnectivityMatrixGenerator as ExInCMG
from utils.connmat import InputMatrixGenerator as IMG

import logging
import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Core Neuron                                                               │
#┴───────────────────────────────────────────────────────────────────────────╯

class Neuron(tf.keras.layers.Layer):
    """Parent class for all neuron variants."""

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Keras Layer Methods                                                   │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        super().__init__()

        cell_cfg = cfg['cell']

        # Internal flag to see if CMG has been built already
        self._cmg_set = False

        # Rewiring behavior
        self.freewiring = cell_cfg.freewiring
        self.rewiring = cell_cfg.rewiring

        # Number of input units
        self.n_in = self.cfg["data"].n_input

        # Layer shape
        self.units = cell_cfg.units

        # Parameters for initial weight distribution
        self.mu = cell_cfg.mu
        self.sigma = cell_cfg.sigma

        # Number of zeros to try to maintain if rewiring is enabled
        self._target_zcount = None


    def build(self, input_shape):
        """Setup  connectivity parameters and IMG and CMG."""

        # [!] Current solution to issues abstracting CMG was to have
        #     an internal flag in each Neuron object tracking whether
        #     or not a CMG was already set, with the idea being that
        #     if it's already set, then the build logic shouldn't be
        #     repeated. I'm not a huge fan of this solution, but it's
        #     good enough for now. I've decided to also (at least for
        #     now) stretch this logic to also handle what to do when
        #     given a probability of the wrong format. The problem with
        #     this is that it's not exactly fantastic encapsulation,
        #     but it's better than most of the solutions I thought of,
        #     so for now we'll leave it be.
        if not self._cmg_set:
            # Read connectivity parameters
            self.p = cell_cfg.p
            self.p_input = self.cfg["cell"].p_input

            # Generate generic input connectivity
            self.input_connmat_generator = IMG(
                self.n_in, self.units, self.p_input, self.mu, self.sigma
            )

            # Generate connectiviy matrix
            self.connmat_generator = CMG(
                self.units,
                self.p,
                self.mu, self.sigma
            )
            self._cmg_set = True  # bookkeeeping


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Additional Methods                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def pseudo_derivative(self, v_scaled, dampening_factor):
        """Calculate the pseudo-derivative."""
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

class ExIn(object):
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

        # Number of excitatory and inhibitory neurons in the layer
        self.num_ex = int(cfg['cell'].frac_e * self.cfg['cell'].units)
        self.num_in = self.cfg['cell'].units - self.num_ex

        # Number of input units
        self.n_in = self.cfg["data"].n_input

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


    def build(self, input_shape):
        """Setup ExIn connectivity parameters and ExInCMG."""

        if not self._cmg_set:
            # Read connectivity parameters
            self.p_ee = self.cfg['cell'].p_ee
            self.p_ei = self.cfg['cell'].p_ei
            self.p_ie = self.cfg['cell'].p_ie
            self.p_ii = self.cfg['cell'].p_ii
            # Read input connectivity parameters
            self.p_input = self.cfg["cell"].p_input

            # Generate input connectivity
            self.input_connmat_generator = IMG(
                self.n_in, self.units, self.p_input, self.mu, self.sigma
            )

            # Generate connectivity matrix
            self.connmat_generator = ExInCMG(
                self.num_ex, self.num_in,
                self.p_ee, self.p_ei, self.p_ie, self.p_ii,
                self.mu, self.sigma
            )
            self._cmg_set = True  # bookkeeeping


    def rewire(self):
        if self._target_zcount is None:
            # When no target number of zeros is recorded, count how
            # many zeros there currently are, save that for future
            # comparison, and return
            self._target_zcount = len(tf.where(self.recurrent_weights == 0))
            self._target_zcount -= self.units  # adjust for diagonal
            logging.debug(f'cell will maintain {self._target_zcount} zeros')
            return

        # Determine how many weights went to zero in this step
        #
        # ...so what I'm doing here is basically just setting the
        # diagonal to non-zero for a second so that it doesn't get
        # put into the "draw pile" and then we set it back. Not
        # sure if this is an efficient or unproblematic solution
        # but it's easy
        #
        # if there are side effects, an option would be to make a
        # deep copy of the recurrent weights and use *that* to
        # generate the indices we need...
        self.recurrent_weights.assign(tf.where(
            self.disconnect_mask,
            tf.ones_like(self.recurrent_weights),
            self.recurrent_weights
        ))
        zero_indices = tf.where(self.recurrent_weights == 0)
        num_new_zeros = tf.shape(zero_indices)[0] - self._target_zcount
        self.recurrent_weights.assign(tf.where(  # undo the thing
            self.disconnect_mask,
            tf.zeros_like(self.recurrent_weights),
            self.recurrent_weights
        ))

        # Replace any new zeros (not necessarily in the same spot)
        if num_new_zeros > 0:
            logging.debug(
                f'found {num_new_zeros} new zeros for a total of '
                + f'{len(zero_indices)} zeros')
            # Generate a list of non-zero replacement weights
            new_weights = np.random.lognormal(
                self.mu,
                self.sigma,
                num_new_zeros
            )
            new_weights[np.where(new_weights == 0)] += 0.01

            # Randomly select zero-weight indices (without replacement)
            # [?] use tf instead of np
            meta_indices = np.random.choice(len(zero_indices), num_new_zeros, False)
            zero_indices = tf.gather(zero_indices, meta_indices)

            # Invert and scale inhibitory neurons
            # [!] Eventually prefer to use .in_mask instead of a loop
            for i in range(len(zero_indices)):
                if zero_indices[i][0] >= self.num_ex:
                    new_weights[i] *= -10

            # Update recurrent weights
            x = tf.tensor_scatter_nd_update(
                tf.zeros(self.recurrent_weights.shape),
                zero_indices,
                new_weights
            )
            logging.debug(f'{tf.math.count_nonzero(x)} non-zero values generated in recurrent weight patch')

            # as of version 2.6.0, tensorflow does not support in-place
            # operation of tf.tensor_scatter_nd_update(), so we just
            # add it to our recurrent weights, which works because
            # scatter_nd_update, only has values in places where
            # recurrent weights are zero
            self.recurrent_weights.assign_add(x)
            logging.debug(
                f'{tf.math.count_nonzero(self.recurrent_weights)} non-zeroes in recurrent layer after adjustments'
            )
