"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`logging.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

from loggers.base import BaseLogger

import os
import tensorflow as tf

class Logger(BaseLogger):
    """Logging interface used while training."""

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Core Operations                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)

    def plot_sinusoid(data, filepath, method="memmap"):
        """Save a numpy array to disk."""
        if method == "memmap":
            raise NotImplementedError("memmap is currently unsupported")
        elif method == "hdf5":
            raise NotImplementedError("HDF5 is currently unsupported")
        elif method == "pickle":
            raise NotImplementedError("pickling is currently unsupported")
        else:
            raise ValueError(f"unrecognized save option: {method}")

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Pseudo Callbacks                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def on_epoch_end(self, epoch_idx, model):
        pass
