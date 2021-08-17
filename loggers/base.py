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

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Bookkeeping                                                       │
        #┴───────────────────────────────────────────────────────────────────╯

        self.cur_epoch = 0  # current epoch in training
        self.cur_step = 0   # current step in current epoch

        # (epoch, step) of last post. `(None, None)` if yet to post.
        self.last_post = {'epoch': None, 'step': None}

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Logging Buffer(s)                                                 │
        #┴───────────────────────────────────────────────────────────────────╯

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

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Logging Filters                                                   │
        #┴───────────────────────────────────────────────────────────────────╯

        # Only log vars with labels in this list
        self.logvar_whitelist = None

        # Log all vars no in this list (overrides whitelist)
        self.logvar_blacklist = None

        # Only log vars with labels in this list
        self.todisk_whitelist = None

        # Log all vars no in this list (overrides whitelist)
        self.todisk_blacklist = None

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ TensorFlow/TensorBoard Integration                                │
        #┴───────────────────────────────────────────────────────────────────╯

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

        # TODO: move checkpointing to the logger (as much as possible)
        #       > add checkpoint manager in .__init__()


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Core Operations                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def log(self, data_label, data, meta={}):
        """Main logging interface.

        All data must be reduceable to a numpy array. If you have data
        you want nested, you'll have to name it in a way that that info
        can be recovered.

        Any metadata you want about the main data will be included in
        `meta.pickle`. This should be about the nature of the variable,
        not the individually logged values, as it will only be added
        the first time the data is observed. Examples of good metadata
        include whether the data is stepwise or epochwise, a text
        description, etc. `stride` is the expected keyword as either
        'step' or 'epoch' or 'static' (never changes)
        """

        # Primary data
        if data_label not in self.logvars:
            bl = self.logvar_blacklist
            wl = self.logvar_whitelist
            if (bl == None) or (data_label not in bl):  # blacklisting
                if (wl == None) or (data_label in wl):  # whitelisting
                    self.logvars[data_label] = [data]
        else:
            self.logvars[data_label].append(data)

        # Metadata
        if data_label not in self.meta:
            meta['dtype'] = type(data)
            self.meta[data_label] = meta

            if 'stride' not in meta:
                # TODO: add case for `static` stride
                logging.warning('stride unspecified for ' + data_label)


    def post(self):
        """Flush data from the logging buffer to disk, possibly with
        additional processing.
        """
        lo_epoch = 1 if self.last_post['epoch'] is None else self.last_post['epoch'] + 1
        hi_epoch = self.cur_epoch

        fp = os.path.join(  # [?] I wonder if I can encapsulate this better
            self.cfg['save'].main_output_dir,
            f"{lo_epoch}-{hi_epoch}.npz"
        )

        # Save data to disk
        #
        # TODO: be more intelligent about when a value
        #       shouldn't be converted and when another
        #       error occurs [?]
        vars_to_save = {}
        for data_label in self.logvars:

            bl = self.todisk_blacklist
            wl = self.todisk_whitelist

            if (bl != None) and (data_label in bl):
                # Data label explicitly blacklisted
                continue
            elif (wl != None) and (data_label not in wl):
                # Data label explicitly *not* whitelisted
                continue

            # Convert to numpy array
            vars_to_save[data_label] = np.array(
                self.logvars[data_label]
            )

            # Adjust precision if specified in the HJSON
            old_type = vars_to_save[data_label].dtype
            new_type = None

            # Check for casting rules
            if old_type in [np.float64, np.float32, np.float16]:
                new_type = eval(f"np.{self.cfg['log'].float_dtype}")
            elif old_type == np.int64:
                new_type = eval(f"np.{self.cfg['log'].int_dtype}")

            # Apply casting rules where they exist
            if new_type is not None and new_type != old_type:
                vars_to_save[data_label] = vars_to_save[data_label].astype(
                    new_type
                )
                logging.debug(f'cast {data_label} ({old_type}) to {new_type}')

        if vars_to_save != {}:
            np.savez_compressed(fp, **vars_to_save)

        # Flush buffer to avoid running out of RAM
        self.logvars = {}

        # Bookkeeping
        self.last_post = {'epoch': self.cur_epoch, 'step': self.cur_step}


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
    #┤ Callbacks                                                             │
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


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Pseudo-Callbacks                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

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

        self.cur_step = 0  # bookkeeping

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
