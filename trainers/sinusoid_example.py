"""Sample of the two idiomatic ways to train.

tensorflow.org/tutorials/customization/custom_training_walkthrough
"""

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
        logger,
        optimizer=tf.keras.optimizers.Adam
    ):
        """TODO: docs  | note how `optimizer` isn't in the parent"""
        super().__init__(model, cfg, data, logger)
        self.optimizer = optimizer


    def train(self):
        """Fit the model using Keras."""
        train_cfg = self.cfg['train']

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
