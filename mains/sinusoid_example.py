import logging
import tensorflow as tf
import utils.config

# Build model ----
from models.neurons.lif import LIF
from models.sinusoid_example import SinusoidSlayer

# Load Data ------
from data import sinusoid_example as sinusoid

# Log ------------
from loggers.callbacks.plots import LIF as PlotLogger
from loggers.callbacks.scalars import Generic as ValueLogger
from loggers.logger import Logger

# Train ----------
from trainers.sinusoid_example import Trainer

def main():
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
    model = form(template).build(cfg)
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
        PlotLogger(cfg)
    ])
    logging.info("Logger instantiated.")

    # Instantiate trainer
    trainer = Trainer(model, cfg, data, logger)
    logging.info("Trainer instantiated.")

    # Train model
    trainer.train()
    logging.info("Training complete.")

    # Postprocessing

if __name__ == '__main__':
    main()
