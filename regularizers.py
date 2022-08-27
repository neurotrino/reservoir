"""Regularizers to enforce biologically-plausible behaviors."""

# external ----
import tensorflow as tf

# internal ----
from utils.misc import SwitchedDecorator

DEBUG_MODE = False

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE


class RateRegularizer(tf.keras.regularizers.Regularizer):
    """Regularizes firing rate.

    Models are penalized according to the deviation of their mean
    firing rate from the target firing rate.
    """

    def __init__(self, target_rate=0.02, weight=1.0):
        """Initialize the regularizer.

        Rate regularization is parameterized by a target firing rate
        and (de)emphasized by the given weighting.
        """
        self.target_rate = target_rate  # target spiking rate
        self.weight = weight  # scaling of loss value

    @switched_tf_function
    def __call__(self, model_output):
        """Calculate the rate regularization loss of a given input.

        Where y is the desired spiking and X is the actual spiking,
        this loss is defined as Σᵢ((xᵢ - y)²) for xᵢ ∈ X.
        """
        (_, spikes, _) = model_output

        # Sum the squared differences between mean and desired spiking
        rate_loss = tf.reduce_mean(spikes, axis=(0, 1)) - self.target_rate
        rate_loss = tf.square(rate_loss)
        rate_loss = tf.reduce_sum(rate_loss)

        # Apply any specified weighting to the final loss
        return self.weight * rate_loss

    def get_config(self):
        """Return a JSON-serializable configuration of this object.

        The output of this method is the input to `.from_config()`.
        """
        return {
            "target_rate": self.target_rate,
            "weight": self.weight,
        }
