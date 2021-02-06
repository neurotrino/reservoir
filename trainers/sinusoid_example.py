"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

from tensorflow.keras.utils import Progbar

import logging
import numpy as np
import os
import tensorflow as tf

# local
from trainers.base import BaseTrainer

# Base class for custom trainers/training loops (inherited)
class Trainer(BaseTrainer):
    """TODO: docs  | note how `optimizer` isn't in the parent"""

    def __init__(self, cfg, model, data, logger):
        super().__init__(cfg, model, data, logger)

        train_cfg = cfg['train']

        try:
            self.optimizer = tf.keras.optimizers.Adam(
                lr=train_cfg.learning_rate
            )
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Loop                                                         │
    #┴───────────────────────────────────────────────────────────────────────╯

    def loss(self, x, y):
        """Calculate the loss on data x labeled y."""
        loss_object = tf.keras.losses.MeanSquaredError()
        voltage, spikes, prediction = self.model(x) # tripartite output

        # [*] Update logger
        self.logger.voltages.append(voltage.numpy())
        self.logger.spikes.append(spikes.numpy())
        self.logger.pred_ys.append(prediction.numpy())

        self.logger.inputs.append(x.numpy())
        self.logger.true_ys.append(y.numpy())

        # training=training is needed only if there are layers with
        # different behavior during training versus inference
        # (e.g. Dropout).

        return loss_object(y_true=y, y_pred=prediction)

    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs, targets)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        return loss_val, grads

    # [?] Are we saying that each batch steps with dt?
    def train_step(self, batch_x, batch_y, batch_idx=None, pb=None):
        """Train on the next batch."""

        # [*] If we were using `.next()` instead of `.get()` with our
        # data generator, this is where we'd invoke the method

        loss, grads = self.grad(batch_x, batch_y)  # [?] logging
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        acc = 0# TODO: calculate actual acc  # [?] logging

        return loss, acc

    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg['train']

        # [*] Declare epoch-level log variables
        losses = []

        # Training takes place here
        pb = Progbar(
            self.cfg['train'].n_batch,
            stateful_metrics=None
        )
        for batch_idx, (batch_x, batch_y) in enumerate(self.data.get()):
            loss, acc = self.train_step(batch_x, batch_y, batch_idx, pb)
            if pb is not None:
                pb.add(
                    1,

                    # [*] Register real-time epoch-level log variables
                    values=[
                        ('loss', loss),
                    ]
                )

            # [*] Update epoch-level log variables
            losses.append(loss)
            #accs.append(acc)

        # [*] Post-training operations on epoch-level log variables
        epoch_loss = np.mean(losses)
        #epoch_acc = np.mean(accs)

        # [*] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx, # TODO? consider replacing w/ generic 'ID' field
            summary_items={
                ("epoch_loss", epoch_loss),
            }
        )

        return epoch_loss

    # [?] Should we be using @tf.function somewhere?
    # [!] Annoying how plt logging shows up
    def train(self):
        """TODO: docs"""
        n_epochs = self.cfg['train'].n_epochs

        for epoch_idx in range(n_epochs):
            print(
                f"\nEpoch {epoch_idx + 1} / {n_epochs}"
                + f" (batch size = {self.cfg['train'].batch_size}):"
            )
            loss = self.train_epoch(epoch_idx)

            # plot every n epochs, or when the loss gets nice and low
            if loss < 0.1 or (epoch_idx + 1) % self.cfg['log'].plot_every == 0:
                self.logger.plot_sinusoid(
                    epoch_idx=epoch_idx,
                    filename=f"{epoch_idx}_{loss}.png"
                )
