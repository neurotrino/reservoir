"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

from tqdm import tqdm

import logging
import tensorflow as tf

# local
from trainers.base import BaseTrainer

# Base class for custom trainers/training loops (inherited)
class Trainer(BaseTrainer):
    """Example of how to idiomatically build a Keras-based trainer.

    TODO: docs
    """
    def __init__(
        self,
        cfg,
        model,
        data,
        logger
    ):
        """TODO: docs  | note how `optimizer` isn't in the parent"""
        super().__init__(model, cfg, data, logger)

        train_cfg = self.cfg['train']

        # Configure the optimizer
        self.optimizer = exec(train_cfg.optimizer)
        try:
            self.optimizer = self.optimizer(lr=train_cfg.learning_rate)
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

        # TODO: maybe adjust this (will have to use for a bit to see
        # what's best)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc: tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

    def loss(self, actual, prediction):
        """TODO: docs | also, does this work for unlabelled data?"""
        loss_object = tf.keras.losses.MeanSquaredError()
        return loss_object(actual, prediction)

    def grad(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: grad")

    @tf.function
    def train_step(self, x, y):
        """TODO: docs"""
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update step-level log variables
        self.train_loss(loss)
        self.train_acc(y, predictions)

        # Report step-level log variables
        return self.train_loss.result(), self.train_acc.result()

    def train_epoch(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        lp = tqdm(range(train_cfg.n_batch))

        # Declare epoch-level log variables
        logvars = {
            "losses": [],
            "accs": []
        }

        # Complete one training epoch
        for _, (bx, by) in zip(lp, self.data):
            train_loss, train_acc = self.train_step(bx, by)

            # Update epoch-level log variables
            logvars['losses'].append(train_loss)
            logvars['accs'].append(train_acc)

        logvars['losses'] = np.mean(logvars['losses'])
        logvars['accs'] = np.mean(logvars['accs'])

        # Report epoch-level log variables
        return logvars


    def train(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        for epoch in range(train_cfg.n_epochs):
            logvars = self.train_epoch()
            # TODO: logging via logger
