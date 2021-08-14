"""Most commonly used script to run models from."""

from os.path import abspath

import logging
import os
import tensorflow as tf
import utils.config



import sys
from setuptools import find_packages
from pkgutil import iter_modules

def find_modules(path):
    modules = set()
    for pkg in find_packages(path):
        modules.add(pkg)
        pkgpath = path + '/' + pkg.replace('.', '/')
        if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.add(pkg + '.' + name)
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.add(pkg + '.' + info.name)
    return modules

print()
print()
print(find_modules('.'))
print()
print()


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
