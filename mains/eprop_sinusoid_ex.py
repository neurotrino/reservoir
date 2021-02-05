import logging
import tensorflow as tf
import utils.config

# Build model ----
from models.neurons.adex import EligExInAdEx
from models.eprop_sinusoid_ex import SinusoidSlayer  # Not sure if I needed a new one

# Load Data ------
from data import sinusoid_example as sinusoid

# Log ------------
from loggers.callbacks.plots import LIF as PlotLogger  # Didn't create one for Adex b/c same output
from loggers.callbacks.scalars import Generic as ValueLogger
from loggers.logger import Logger

# Train ----------
from trainers.eprop_sinusoid_ex import Trainer

def main():
    # Use command line arguments to load data, create directories, etc.
    form, cfg = utils.config.boot()
    logging.info("Experiment directory: " + cfg['save'].exp_dir)

    # Build model
    template =                                                               \
    {
        "_class": SinusoidSlayer,

        "cell":
        {
            "_class": EligExInAdEx
        }
    }
    model = form(template).build(cfg)
    logging.info("Model built.")

    # Load data
    data = sinusoid.DataGenerator(cfg)
    logging.info("Dataset loaded.")

    # Instantiate logger
    logger = Logger(cfg)   #, cb=[
        #tf.keras.callbacks.TensorBoard(
        #    log_dir=cfg['save'].log_dir,
        #    histogram_freq=1,
        #    write_graph=False
        #),
        #ValueLogger(cfg),
        #PlotLogger(cfg)
    #])
    logging.info("Logger instantiated.")

    # Instantiate trainer
    trainer = Trainer(cfg, model, data, logger)
    logging.info("Trainer instantiated.")

    # Train model
    logging.info("About to start training...")
    trainer.train()
    logging.info("Training complete.")

    # Postprocessing
    if cfg['save'].postprocess:
        pass

if __name__ == '__main__':
    main()
