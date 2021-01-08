"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

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
        except e:
            logging.warning(f"learning rate not set: {e}")

        # TODO: maybe adjust this (will have to use for a bit to see
        # what's best)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc: tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

    def loss(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: loss")

    def grad(self):
        """TODO: docs"""
        raise NotImplementedError("Trainer missing method: grad")

    @tf.function
    def train_step(self, x, y):
        """TODO: docs"""
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.model.loss_object(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update step-level log variables
        self.train_loss(loss)
        self.train_acc(y, predictions)

        # Report step-level log variables
        return self.train_loss, self.train_acc

    def train_epoch(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        lp = tqdm(range(self.))

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

        # Report epoch-level log variables
        return logvars


    def train(self):
        """TODO: docs"""
        train_cfg = self.cfg['train']

        for epoch in train_cfg.n_epochs:
            loss, acc = self.train_epoch()
            # TODO: logging via logger



        self.model.compile(
            optimizer=self.optimizer(lr=train_cfg.learning_rate),
            loss=dict(tf_op_layer_output=tf.keras.losses.MeanSquaredError())
        )
        self.model.fit(
            self.data,
            epochs=train_cfg.n_epochs,
            callbacks=self.logger.callbacks
        )
        """TODO:
        if config['do_plot'] and config['do_save']:
            model.fit(data, epochs=config['epochs'])
            # TODO: decide if save is method of model, logger, or just util
        elif config['do_plot']:
            model.fit(data, epochs=config['epochs'],
                callbacks=[logger.plot])
        elif do_save:
            model.fit(data, epochs=config['epochs'],
                callbacks=[logger.save,   # Update logger so this
                           logger.plot])  # is less awk <-- TODO
        else:
            #nada
        # Logger initialization filters callbacks based on config file
        """
