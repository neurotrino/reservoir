"""Logger class(es) for monitoring network training and testing."""

import logging
import numpy as np
import os
import tensorflow as tf

class BaseLogger:
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        """Create a new logger."""
        self.cfg = cfg


        # Bookkeeping -------------------------------------------------

        self.cur_epoch = np.uint16(0)  # current epoch in training
        self.cur_step = np.uint16(0)   # current step in current epoch

        # (epoch, step) of last post. `(None, None)` if yet to post.
        self.last_post = (None, None)


        # Logging buffer(s) -------------------------------------------

        # Buffered data for each logvar.
        #
        # Over the course of training, variables of interest (logvars)
        # should have their values stored here at regular intervals
        # (usually per-step or per-epoch), to be flushed to disk with
        # each call to `.post()`.
        self.logvars = {}

        # Metadata for each logvar.
        #
        # Contains values such as 'stride' (static, step, epoch, etc.),
        # 'description', and so on. Also serves as a catalog for
        # observed logvars. Each logvar should have exactly one entry
        # in `meta` by the end of training, at which point something
        # like `meta.pickle` should be saved to disk alongside the
        # posted data.
        self.meta = dict()


        # TensorFlow/TensorBoard --------------------------------------

        # Summary writers.
        #
        # TensorFlow summary writers write summary data (scalar values)
        # to TensorBoard event files (`.v2` files). These can be looked
        # at in TensorBoard directly, or processed into another format
        # via scripts not currently included in this infrastructure.
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

        # Keras callbacks.
        #
        # List of logging callbacks active during the session. See the
        # `.add_callback()` documentation for more information.
        self.callbacks = []
        if cb is not None:
            self.add_callback(cb)


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Core Operations                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def log(self, data_label, data, meta):
        """Add data to the logging buffer."""
        class_name = self.__class__.__name__
        logging.warning(f"{class_name}.log() called but not implemented")


    def post(self):
        """Flush data from the logging buffer to disk, possibly with
        additional processing.
        """
        class_name = self.__class__.__name__
        logging.warning(
            f"{class_name}.post() called but not implemented: flushing buffer"
        )
        self.logvars = {}  # flush buffer to avoid running out of RAM


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
    #┤ Callbacks and Pseudo-Callbacks                                        │
    #┴───────────────────────────────────────────────────────────────────────╯

    def add_callback(self, cb):
        """Add a keras callback or list of keras callbacks to logger.

        Keras callbacks are designed for use with `.fit()`, running at
        specific points of execution during training. It's not advised
        these be used for logging in custom training loops. See
        TensorFlow's callback documentation for more information.

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
        """Logging logic/operations performed at the start of training.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list


    def on_train_end(self):
        """Logging logic/operations performed at the end of training.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list


    def on_epoch_begin(self):
        """Logging logic/operations performed at the start of each
        epoch.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}

        self.cur_epoch += 1  # bookkeeping

        return action_list


    def on_epoch_end(self):
        """Logging logic/operations performed at the end of each epoch.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list


    def on_step_begin(self):
        """Logging logic/operations performed at the start of each
        step.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}

        self.cur_step += 1  # bookkeeping

        return action_list


    def on_step_end(self):
        """Logging logic/operations performed at the end of each step.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list


    def on_batch_begin(self):
        """Logging logic/operations performed at the start of each
        batch processing step.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list


    def on_batch_end(self):
        """Logging logic/operations performed at the end of each batch
        processing step.

        Returns an "action list" of command strings and values
        indicating operations for the trainer to perform (e.g. saving
        weights).
        """
        action_list = {}
        return action_list
