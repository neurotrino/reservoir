"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

from tensorflow.keras.utils import Progbar

import logging
import numpy as np
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
        data,  # should be "belt-fed"
        logger
    ):
        """TODO: docs  | note how `optimizer` isn't in the parent"""
        super().__init__(model, cfg, data, logger)

        train_cfg = self.cfg['train']

        # Configure the optimizer
        #self.optimizer = tf.keras.optimizers.Adam()
        try:
            self.optimizer = tf.keras.optimizers.Adam(
                lr=train_cfg.learning_rate
            )
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

        # TODO: maybe adjust this (will have to use for a bit to see
        # what's best)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc: tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

    def loss(self, x, y):
        """Calculate the loss on data x labeled y."""
        loss_object = tf.keras.losses.MeanSquaredError()
        voltage, spikes, prediction = self.model(x) # tripartite output
        # Dev note: see training=training in guide if we need to have
        # diff training and inference behaviors
        return loss_object(y_true=y, y_pred=prediction)

    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs, targets)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        return loss_val, grads

    #@tf.function
    def train_step(self, batch_x, batch_y, batch_idx=None, pb=None):
        """Train on the next batch."""
        #batch_x, batch_y = next(
        #    self.data.next_batch(self.cfg['train'].batch_size)
        #)
        loss, grads = self.grad(batch_x, batch_y)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        acc = 0# TODO: calculate actual acc

        return loss, acc

    #@tf.function
    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg['train']

        # [!] Declare epoch-level log variables
        losses = []
        accs = []

        # Training takes place here
        pb = Progbar(
            self.cfg['train'].batch_size * self.cfg['train'].n_batch,
            stateful_metrics=None
        )
        for batch_idx, (batch_x, batch_y) in enumerate(self.data.dataset):
            loss, acc = self.train_step(batch_x, batch_y, batch_idx, pb)
            if pb is not None:
                pb.add(self.cfg['train'].batch_size)

            # [!] Update epoch-level log variables
            #losses.append(loss)
            #accs.append(acc)

        # [!] Post-training operations on epoch-level log variables
        #epoch_loss = np.mean(losses)
        #epoch_acc = np.mean(accs)

        # [!] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx, # TODO? consider replacing w/ generic 'ID' field
            summary_items={
                #("epoch_loss", epoch_loss),
                #("epoch_acc", epoch_acc)
            }
        )

    #@tf.function
    def train(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        for epoch_idx in range(train_cfg.n_epochs):
            self.train_epoch(epoch_idx)
