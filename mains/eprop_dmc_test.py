# external ----
import logging
import tensorflow as tf

# local -------
from models.eprop_dmc_model import EpropDMC
from models.neurons.adex import EligExInAdEx
from trainers.eprop_test_trainer import Trainer
from data import dmc
from loggers.logger import Logger
from loggers.callbacks.plots import Generic as PlotLogger
from loggers.callbacks.scalars import Generic as ValueLogger

from utils import config
from utils import dirs

def main():
    """TODO: docs"""

    # Use command line arguments to load data, create directories, etc.
    form, cfg = config.boot()

    # Build model
    template =                                                               \
    {
        "_class": EpropDMC,

        "cell":
        {
            "_class": EligExInAdEx
        }
    }
    model = form(template).build()
    logging.info("Model built.")

    # Load data
    data = dmc.load_data(cfg)
    logging.info("Dataset loaded.")

    # Instantiate logger
    logger = Logger(cfg, cb=[
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg['save'].log_dir,
            histogram_freq=1,
            write_graph=False
        ),
        ValueLogger(cfg)#,
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