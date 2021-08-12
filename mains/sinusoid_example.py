from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config

# Build model ----
from models.neurons.lif import ExInLIF
from models.sinusoid_example import SinusoidSlayer

# Load Data ------
from data import sinusoid

# Log ------------
from loggers.sinusoid_example import Logger as Logger

# Train ----------
from trainers.sinusoid_example import Trainer

def main():
    # Use command line arguments to load data, create directories, etc.
    form, cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    # Build model
    model = SinusoidSlayer(cfg)
    logging.info("model built")

    # Load data
    data = sinusoid.DataGenerator(cfg)
    logging.info("dataset loaded")

    # Instantiate logger
    logger = Logger(cfg)
    logging.info("logger instantiated")

    print()
    print()
    print(cfg)
    print()
    print()
    print(model)
    print()
    print()
    print(data)
    print()
    print()
    print(logger)

    # Instantiate trainer
    trainer = Trainer(cfg, model, data, logger, logger)
    logging.info("trainer instantiated")

    # Train model
    trainer.train()
    logging.info("training complete")


if __name__ == '__main__':
    main()
