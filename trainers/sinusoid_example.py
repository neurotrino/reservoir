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
        #self.optimizer = tf.keras.optimizers.Adam()
        try:
            self.optimizer = tf.keras.optimizers.Adam(lr=train_cfg.learning_rate)
        except Exception as e:
            logging.warning(f"learning rate not set: {e}")

        # TODO: maybe adjust this (will have to use for a bit to see
        # what's best)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc: tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

    def loss(self, x, y):
        """TODO: docs | also, does this work for unlabelled data?"""
        loss_object = tf.keras.losses.MeanSquaredError()
        voltage, spikes, prediction = self.model(x) # tripartite output
        # Dev note: see training=training in guide if we need to have
        # diff training and inference behaviors
        return loss_object(y_true=y, y_pred=prediction)

    def grad(self, inputs, targets):
        """Gradient calculation(s)"""
        with tf.GradientTape() as tape:
            loss_val = self.loss(inputs, targets)
        return loss_val, tape.gradient(loss_val, self.model.trainable_variables)

    @tf.function
    def train_step(self, x, y):
        # TODO (?) : do we need this for logging purposes or can we get
        # away with batch-level?
        pass

    def train_epoch(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        lp = tqdm(range(train_cfg.n_batch))

        # Declare epoch-level log variables
        logvars = {
            "losses": [],
            "accs": []
        }

        # Iterate over batches
        for _, (x, y) in zip(lp, self.data):
            loss_val, grads = self.grad(x, y)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables)
            )

        """
            train_loss, train_acc = self.train_step(x, y)
            # Update epoch-level log variables
            logvars['losses'].append(train_loss)
            logvars['accs'].append(train_acc)

        logvars['losses'] = np.mean(logvars['losses'])
        logvars['accs'] = np.mean(logvars['accs'])

        # Report epoch-level log variables
        """
        return logvars


    def train(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        for epoch in range(train_cfg.n_epochs):
            logvars = self.train_epoch()
            # TODO: logging via logger
