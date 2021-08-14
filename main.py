"""Most commonly used script to run models from."""

from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config


#=================================

import glob
from pathlib import Path

for filepath in Path('.').rglob('*.py'):

    fp = str(filepath)

    if not fp.endswith("__init__.py"):
        module_name = fp[:-3]  # remove '.py' from string
        module_name = module_name.replace('/', '.')
        print()
        print(module_name)
        print()
print("---------------")

# Get file paths of all modules.
modules = glob.glob('*.py')

print()
print()
print(modules)
print()
print()
exit()
"""
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
"""

#=================================


def main():
    # Use command line arguments to load data, create directories, etc.
    cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    # Build model
    model = eval(f"models.grey_goo.{cfg['model'].type}")(cfg)
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
