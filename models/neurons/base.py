"""TODO: module docs
"""

import tensorflow as tf

# Base model for the neurons we use
class BaseNeuron(tf.keras.layers.Layer):
    """TODO: docs"""
    def __init__(self):
        super().__init__(dynamic=False)

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
