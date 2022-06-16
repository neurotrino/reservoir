"""Statistical tools."""

# external ----
import tensorflow as tf

# internal ----
from utils.misc import SwitchedDecorator

DEBUG_MODE = False

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE

class StatisticalDistribution:
    """Generic function wrapper aliasing statistical distributions."""

    def __init__(self, sampling_fn, **kwargs) -> None:

        @switched_tf_function
        def wrapped_fn(shape, **calltime_kwargs):
            """
            Preconfigured sampling function capable of taking further
            arguments at calltime.
            """
            joint_kwargs = {**kwargs, **calltime_kwargs}
            return sampling_fn(shape, **joint_kwargs)


        self._sampling_fn = wrapped_fn


    @switched_tf_function
    def sample(self, shape, **calltime_kwargs):
        """Sample from the configured distribution.

        Arguments:
            shape: shape of samples to return.
            calltime_kwargs: arguments which must be given to the
                sampling function at call time.
        """
        return self._sampling_fn(shape, **calltime_kwargs)
