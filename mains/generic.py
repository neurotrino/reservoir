"""Most commonly used script to run models from."""

from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config

# Build model ----
from models.grey_goo import GreyGoo

# Load Data ------
from data import sinusoid

# Log ------------
from loggers.sinusoid_example import Logger as Logger

# Train ----------
from trainers.sinusoid_example import Trainer

def main():
    # Use command line arguments to load data, create directories, etc.
    cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    # Build model
    model = GreyGoo(cfg)
    logging.info("model built")

    # Load data
    data = sinusoid.DataGenerator(cfg)
    logging.info("dataset loaded")

    # Instantiate logger
    logger = Logger(cfg)
    logging.info("logger instantiated")

    # Instantiate trainer
    trainer = Trainer(cfg, model, data, logger)
    logging.info("trainer instantiated")

    # Train model
    trainer.train()
    logging.info("training complete")


if __name__ == '__main__':
    main()
