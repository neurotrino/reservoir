"""TODO: Docs"""

import tensorflow as tf

class BaseTrainer(object):
    """TODO: docs"""
    def __init__(self, cfg, model, data, logger):
        """TODO: docs

        args:
          model: model to train.

          data: any format which suits the training methods. Clearly
            document how your data should be formatted and processed
            when implementing a trainer. Take this into account when
            creating functions in `data/` and when putting data into a
            trainer (e.g. make sure you're not producing a tuple of
            training and validation data then passing that into a
            method expecting a single dataset which it will, itself
            partition).

          cfg: configuration object produced by
            `utils.config.load_hjson_config()`.

          logger: logger compatible with the format found in
            `utils.logger`.
        """
        self.model = model
        self.data = data
        self.cfg = cfg
        self.logger = logger

    def train(self):
        """Train the model.

        This function is the core of a trainer and should always be
        implemented. This is typically done with Keras' `fit` or a
        custom training loop (which should require implementing
        `train_step` and `train_epoch` in your trainer).
        """
        raise NotImplementedError("Trainer requires this function")

    def loss(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: loss")

    def grad(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: grad")
    
    @tf.function
    def train_step(self, x, y):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: train_step")

    def train_epoch(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: train_epoch")
