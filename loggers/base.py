"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`logging.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

import logging

import os
import tensorflow as tf

class BaseLogger:
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        """Create a new logger."""
        self.cfg = cfg

        # Step/epoch counters
        self.cur_epoch = 0
        self.cur_step = 0

        # (epoch, step) of last post. `(None, None)` if yet to post.
        self.last_post = (None, None)

        #
        self.logvars = {}

        # metadata of logvars
        self.meta = dict()

        # Initialize summary writers
        #
        # TensorFlow summary writers write summary data (scalar values)
        # to TensorBoard event files (`.v2` files). These can be looked
        # at in TensorBoard directly, or processed into another format
        # using scripts in `utils\postproc.py`.
        #
        # Two summary writers are contained in the base logger: one for
        # use during training, the other during testing. Other loggers
        # may or may not add additional summary writers (they generally
        # shouldn't need to).
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(cfg['save'].summary_dir, "train")
        )
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(cfg['save'].summary_dir, "test")
        )

        # List of logging callbacks active during the session
        self.callbacks = []
        if cb is not None:
            self.add_callback(cb)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ TensorBoard Logging                                                   │
    #┴───────────────────────────────────────────────────────────────────────╯

    def summarize(self, index, summary_items=[], writer="train"):
        """Log scalar values.

        Typically run with custom trainers.

        Uses auto-increment unless there's

        args:
          index: x-axis value for the scalar, typically an epoch index
            or a step index. Note that to properly plot in TensorBoard,
            step indices must maintain position between epochs (e.g.
            the last step index on epoch 1 being 5 means the first step
            index on epoch 2 should be 6, assume a step size of 1)
          summary_items: tuple list containing the string identifier
            and scalar value of the item being summarized
          writer: either "train" or "test"; selects which writer to use
        """
        _writer = self.train_writer if writer == "train" else self.test_writer

        with _writer.as_default():
            for (label, value) in summary_items:
                try:
                    tf.summary.scalar(label, value, step=index)
                except:
                    continue
                _writer.flush


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Direct Logging                                                        │
    #┴───────────────────────────────────────────────────────────────────────╯

    def log(self, data_label, data, meta):
        pass

    # [?] considering adding a `register` method that does both
    # summarize and updates logger state
    #
    # Also considering:
    #
    #    ```
    #    if not self.cfg['log'].tb_compat:
    #        return
    #    ```
    #
    # Where `tb_compat` is a bool specified in the HJSON config. This
    # might save filespace

    def post(self):
        """Operations you want to do with/on the data post-training.
        """
        logging.warning(
            self.__class__.__name__ + ".post() called but not implemented"
        )


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Callbacks and Pseudo-Callbacks                                        │
    #┴───────────────────────────────────────────────────────────────────────╯

    def add_callback(self, cb):
        """Add a keras callback or list of keras callbacks to logger.

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


    def on_train_begin(self):
        pass


    def on_train_end(self):
        pass


    def on_epoch_begin(self):
        pass


    def on_epoch_end(self, inc=1):
        """
        Any logic to be performed in the logger whenever an epoch
        completes in the training.
        """
        self.cur_epoch += inc


    def on_step_begin(self):
        """
        """
        pass


    def on_step_end(self, inc=1):
        """
        Any logic to be performed in the logger whenever a step
        completes in the training.
        """
        self.cur_step += inc


    def on_batch_begin(self):
        """
        """
        pass


    def on_batch_end(self):
        """
        """
        pass
