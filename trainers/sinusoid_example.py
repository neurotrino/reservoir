"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

from tensorflow.keras.utils import Progbar

import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

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
        self.logger.logvars['mvars'].append({
            # Output variables
            "voltages": voltage.numpy(),
            "spikes": spikes.numpy(),
            "pred_ys": prediction.numpy(),

            # Input/reference variables
            "inputs": x.numpy(),
            "true_ys": y.numpy(),
        })

        # training=training is needed only if there are layers with
        # different behavior during training versus inference
        # (e.g. Dropout).

        return loss_object(y_true=y, y_pred=prediction)


    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs, targets)

        # Calculate the gradient of the loss with respect to each
        # layer's trainable variables. In this example, calculates the
        # gradients for (in order):
        # > `rnn/ex_in_lif/input_weights:0`
        # > `rnn/ex_in_lif/recurrent_weights:0`
        # > `dense/kernel:0`
        # > `dense/bias:0`
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        return loss_val, grads


    # [?] Are we saying that each batch steps with dt?
    def train_step(self, batch_x, batch_y, batch_idx=None, pb=None):
        """Train on the next batch."""

        # [*] If we were using `.next()` instead of `.get()` with our
        # data generator, this is where we'd invoke that method

        # [*] Calculate step-wise logging values we're interested in
        # before the gradients are applied
        pre_weights = [x.numpy() for x in self.model.trainable_variables]

        # Calculate the gradients and update model weights
        loss, grads = self.grad(batch_x, batch_y)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        acc = 0  # acc doesn't make sense with this dataset

        # [*] We can log the weights &c of specific layers at each step
        # like so:
        #
        # [?] way to get layers by name?
        for layer in self.model.layers:
            # Layers of note in this example are input_1, rnn,
            # spike_regularization, and dense
            #
            # See layer attributes here:
            # tensorflow.org/api_docs/python/tf/keras/layers/Layer
            if layer.name == "spike_regularization":
                self.logger.logvars['sr_wgt'].append(layer.weights)
                self.logger.logvars['sr_losses'].append(layer.losses)

        # [*] Log any step-wise variables
        #
        # These values are in different lists, but indexed the same,
        # i.e. `grad_data['names'][i]` will produce the name of the
        # layer with shape `grad_data['shapes'][i]`, and likewise for
        # values and gradients.
        self.logger.logvars['svars'].append({
            # Names of the layers the gradient of the loss is being
            # calculated with respect to.
            #
            # List of strings, one per watched layer.
            "names": [x.name for x in self.model.trainable_variables],

            # Shapes of the variable and gradient tensors (should be
            # equal)
            #
            # List of integer tuples, one per watched layer.
            "shapes": [x.shape for x in self.model.trainable_variables],

            # Layer weights *before* the gradients were applied.
            #
            # List of numpy arrays (np.float32), one per watched layer.
            "pre_weights": pre_weights,

            # Layer weights *after* the gradients were applied.
            #
            # List of numpy arrays (np.float32), one per watched layer.
            "post_weights": [
                x.numpy() for x in self.model.trainable_variables
            ],

            # Values of the gradient tensors.
            #
            # List of numpy arrays (np.float32), one per watched layer.
            "grads": [x.numpy() for x in grads],

            # Calculated loss
            #
            # A single float.
            "loss": float(loss)
        })

        return loss, acc  # [*] Log these if you want step loss logged


    def train_epoch(self, epoch_idx=None):
        """Train over an epoch.

        Also performs logging at the level of epoch metrics.
        """
        train_cfg = self.cfg['train']

        # [*] Declare epoch-level log variables
        losses = []

        # Training takes place here
        pb = Progbar(self.cfg['train'].n_batch, stateful_metrics=None)

        for batch_idx, (batch_x, batch_y) in enumerate(self.data.get()):
            loss, acc = self.train_step(batch_x, batch_y, batch_idx, pb)
            pb.add(
                1,
                values=[
                    # [*] Register real-time epoch-level log variables
                    ('loss', loss),
                ]
            )

            # [*] Update epoch-level log variables
            losses.append(loss)
            #accs.append(acc)

        # [*] Post-training operations on epoch-level log variables
        epoch_loss = np.mean(losses)

        # [*] Log any epoch-wise variables.
        self.logger.logvars['evars'].append({
            "loss": epoch_loss,
        })

        # [*] Summarize epoch-level log variables here
        # [?] Register epoch-level log variables here
        self.logger.summarize(
            epoch_idx,
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

        # Create checkpoint manager
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            net=self.model
        )
        cpm = tf.train.CheckpointManager(
            ckpt,
            self.cfg['save'].checkpoint_dir,
            max_to_keep=None
        )

        # Train the model
        for epoch_idx in range(n_epochs):
            """[*] Other stuff you can log
            print("W = {}, B = {}".format(*self.model.trainable_variables))
            for k in self.model.trainable_variables:
                print("trainable_variables:")
                print(k)
            """

            print(
                f"\nEpoch {epoch_idx + 1} / {n_epochs}"
                + f" (batch size = {self.cfg['train'].batch_size}):"
            )
            loss = self.train_epoch(epoch_idx)

            if (epoch_idx + 1) % self.cfg['log'].post_every == 0:
                # Create checkpoints
                # [?] Can also put in the step loop
                # [?] Originally used a CheckpointManager in the logger
                self.model.save_weights(os.path.join(
                    self.cfg['save'].checkpoint_dir,
                    f"checkpoint_e{epoch_idx + 1}"
                ))

                # Other logging
                self.logger.post(epoch_idx + 1)
