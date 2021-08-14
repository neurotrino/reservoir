"""Most commonly used script to run models from."""

from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config


#=================================

from pathlib import Path

import importlib.util

import glob

for mod in ['data', 'loggers', 'models', 'trainers']:
    for submod in glob.glob(f'{mod}/*.py'):
        print(submod)
exit()

for filepath in Path('.').rglob('*.py'):

    fp_str = str(filepath)

    if not fp_str.endswith("__init__.py"):
        # Convert filepath syntax to modulepath syntax
        module_name = fp_str[:-3]
        module_name = module_name.replace('/', '.')

        # Load module into python
        spec = importlib.util.spec_from_file_location(module_name, fp_str)
        print(spec)
        print()
        x = importlib.util.module_from_spec(spec)
        """
        spec.loader.exec_module(x)
        """

#=================================

def main():
    # Use command line arguments to load data, create directories, etc.
    cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    # Build model
    model = eval(f"models.{cfg['model'].type}.Model")(cfg)
    logging.info(f"model built: {cfg['model'].type}")

    # Load data
    data = eval(f"data.{cfg['data'].type}").DataGenerator(cfg)
    logging.info(f"dataset loaded: {cfg['data'].type}")

    # Instantiate logger
    logger = eval(f"loggers.{cfg['log'].type}").Logger(cfg)
    logging.info(f"logger instantiated: {cfg['log'].type}")

    # Instantiate trainer
    trainer = eval(f"trainers.{cfg['train'].type}").Trainer(
        cfg, model, data, logger
    )
    logging.info(f"trainer instantiated: {cfg['train'].type}")

    # Train model
    trainer.train()
    logging.info("training complete")


if __name__ == '__main__':
    main()
