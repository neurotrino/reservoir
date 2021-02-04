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
    """TODO: docs  | note how `optimizer` isn't in the parent"""

    def __init__(
        self,
        cfg,
        model,
        data,  # should be "belt-fed"
        logger
    ):
        super().__init__(cfg, model, data, logger)

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Training Setup                                                    │
        #┴───────────────────────────────────────────────────────────────────╯

        train_cfg = cfg['train']

        try:
            self.optimizer = tf.keras.optimizers.Adam(  # mv from HJSON
                lr=train_cfg.learning_rate
            )
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

        # TODO: maybe adjust this (will have to use for a bit to see
        # what's best)
        #
        # Didn't end up using, at least not yet
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc: tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

        #┬───────────────────────────────────────────────────────────────────╮
        #┤ Logging Setup                                                     │
        #┴───────────────────────────────────────────────────────────────────╯

        buffer_dim = (train_cfg.n_batch, cfg['data'].seq_len)

        # The way this is setup is that we have numpy arrays which fill
        # with values we want to log until some point we specify in the
        # training loop which dumps their contents to a file. I wasn't
        # sure where the overhead would be the worst: keeping large
        # numpy arrays in memory, or saving to disk, so this seemed
        # like the logical decision. If there's a better/idiomatic way
        # of doing this, please let me know or just implement it.
        self.voltage_buffer = np.empty(buffer_dim)
        self.spikes_buffer = np.empty(buffer_dim)
        self.prediction_buffer = np.empty(buffer_dim)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Helper Functions (Logging)                                            │
    #┴───────────────────────────────────────────────────────────────────────╯

    def some_name():
        pass

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Loop                                                         │
    #┴───────────────────────────────────────────────────────────────────────╯

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

    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg['train']

        # [!] Declare epoch-level log variables
        losses = []

        # Training takes place here
        pb = Progbar(
            self.cfg['train'].batch_size * self.cfg['train'].n_batch,
            stateful_metrics=None
        )
        for batch_idx, (batch_x, batch_y) in enumerate(self.data.dataset):
            loss, acc = self.train_step(batch_x, batch_y, batch_idx, pb)
            if pb is not None:
                pb.add(
                    self.cfg['train'].batch_size,

                    # [!] Register real-time epoch-level log variables
                    values=[
                        ('loss', loss),
                    ]
                )

            # [!] Update epoch-level log variables
            losses.append(loss)
            #accs.append(acc)

        # [!] Post-training operations on epoch-level log variables
        epoch_loss = np.mean(losses)
        #epoch_acc = np.mean(accs)

        # [!] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx, # TODO? consider replacing w/ generic 'ID' field
            summary_items={
                ("epoch_loss", epoch_loss),
            }
        )

    def train(self):
        """TODO: docs"""
        n_epochs = self.cfg['train'].n_epochs

        for epoch_idx in range(n_epochs):
            print(f"Epoch {epoch_idx + 1} / {n_epochs}:")
            self.train_epoch(epoch_idx)
            # Should we be using @tf.function somewhere?
