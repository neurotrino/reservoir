"""Most commonly used script to run models from."""

from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config

# Build model ----
from models import *

# Load Data ------
from data import *

# Log ------------
from loggers import *

# Train ----------
from trainers import *

def main():
    # Use command line arguments to load data, create directories, etc.
    cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    # Build model
    model = eval(f'models.{cfg['model'].type}').Model(cfg)
    logging.info(f"model built: {cfg['model'].type}")

    # Load data
    data = eval(f'data.{cfg['data'].type}').DataGenerator(cfg)
    logging.info(f"dataset loaded: {cfg['data'].type}")

    # Instantiate logger
    logger = eval(f'loggers.{cfg['log'].type}').Logger(cfg)
    logging.info(f"logger instantiated: {cfg['log'].type}")

    # Instantiate trainer
    trainer = eval(f'trainers.{cfg['train'].type}').Trainer(
        cfg, model, data, logger
    )
    logging.info(f"trainer instantiated: {cfg['train'].type}")

    # Train model
    trainer.train()
    logging.info("training complete")


if __name__ == '__main__':
    main()
