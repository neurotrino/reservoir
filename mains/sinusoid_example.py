import logging
import tensorflow as tf
import utils.config

# Build model ----
from models.neurons.lif import LIF
from models.sinusoid_example import SinusoidSlayer

# Load Data ------
from data import sinusoid

# Log ------------
from loggers.callbacks.plots import LIF as PlotCB
from loggers.base import BaseLogger as Logger

# Train ----------
from trainers.sinusoid_example import Trainer

# Postprocess ----
import utils.dataproc as dataproc

def main():
    # Use command line arguments to load data, create directories, etc.
    form, cfg = utils.config.boot()
    logging.info("experiment directory: " + cfg['save'].exp_dir)

    # Build model
    template =                                                               \
    {
        "_class": SinusoidSlayer,

        "cell":
        {
            "_class": LIF
        }
    }
    model = form(template).build(cfg)
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

    # Perform postprocessing
    if cfg['save'].postprocess:
        dataproc.process(cfg, trainer)

if __name__ == '__main__':
    main()
