"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`logging.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

import os
import tensorflow as tf

class Logger(object):
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        """Create a new logger."""
        save_cfg = cfg['save']

        # Initialize summary writers
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(save_cfg.summary_dir, "train")
        )
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(save_cfg.summary_dir, "test")
        )

        # List of logging callbacks active during the session
        self.callbacks = []
        if cb is not None:
            self.add_callback(cb)


    def add_callback(self, cb):
        """Add a callback or list of callbacks to the logger.

        Callbacks run at certain points of TensorFlow execution, and
        are one of the two primary logging mechanisms (the other being
        usage of `tf.summary` in a custom training loop).

        Typically run with Keras-templated trainers or to create plots.

        args:
          cb: `CallBack` or list of `CallBack`s
        """
        if isinstance(cb, list):
            # Add a list of callbacks
            for x in cb:
                self.callbacks.append(cb)
        else:
            # Add a single callback
            self.callbacks.append(cb)


    def summarize(self, index, summary_items=[], writer="train"):
        """Log scalar values.

        Typically run with custom trainers.

        args:
          index: TODO
          summary_items: tuple list containing the string identifier
            and scalar value of the item being summarized
          writer: either "train" or "test", selects which writer to use
        """
        _writer = self.train_writer if writer == "train" else self.test_writer

        for (label, value) in summary_items:
            tf.summary.scalar(label, value)
        tf.summary.scalar()