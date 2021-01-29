"""TODO: module (example) docs

CONFIGURING SESSIONS
  Session variables are predominantly specified in two places: the
  HJSON configuration file and the command line arguments.

BUILDING A MODEL
  Presently, a template specifying the class of nested objects is
  necessary.

LOADING DATA
  Data generation itself will vary. See the sample code in `data` for
  some examples.

LOGGING
  ...

TRAINING
  ...
"""

# external ----
import logging
import tensorflow as tf

import sys

sys.path.append('/home/macleanlab/tf2_migration/')

# local -------
from models.sinusoid_example import SinusoidSlayer
from models.neurons.lif import LIF
from data import sinusoid_example as sinusoid
from trainers.sinusoid_example import Trainer
from loggers.logger import Logger
from loggers.callbacks.plots import Generic as PlotLogger
from loggers.callbacks.scalars import Generic as ValueLogger

import utils.config
import utils.dirs

def main():
    """TODO: docs"""

    # Use command line arguments to load data, create directories, etc.
    form, cfg = utils.config.boot()

    # Build model
    template =                                                               \
    {
        "_class": SinusoidSlayer,

        "cell":
        {
            "_class": LIF
        }
    }
    model = form(template).build()
    logging.info("Model built.")

    # Load data
    data = sinusoid.load_data(cfg)
    logging.info("Dataset loaded.")

    # Instantiate logger
    logger = Logger(cfg, cb=[
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg['save'].log_dir,
            histogram_freq=1,
            write_graph=False
        ),
        ValueLogger(cfg),
        #PlotLogger(cfg)
    ])
    logging.info("Logger instantiated.")

    # Instantiate trainer
    trainer = Trainer(model, cfg, data, logger)
    logging.info("Trainer instantiated.")

    # Train model
    trainer.train()
    logging.info("Training complete.")


if __name__ == '__main__':
    main()
