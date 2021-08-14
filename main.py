from os.path import abspath
from pathlib import Path

import importlib.util
import logging
import utils.config

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

    #======================================================================
    # What is the below block of code? I'm sorry you asked. Basically,
    # Python has a trash import system, so we need to hard code our import
    # logic below in order to implement adequate abstraction in `main.py`.
    #
    # Functionally, I recommend ignoring the witchcraft below and just
    # know that what's basically happening is submodules are getting
    # registered by Python as things like `models.eprop_sinusoid_ex` or
    # `data.ccd`. It's as though we had manually called `import data.ccd`
    # and so on, just without the "manual" part.
    #
    # Why do we want this? It allows us to specify in our HJSON files the
    # modules we want to use per experiment. We were already doing this,
    # but now...
    # 1. we only need to change class names in one place (the HJSON file)
    # 2. we can run this script outside of virtual environments, meaning we
    #    don't have to worry about weird `pip install -e .` behaviors
    # 3. we only need one `main.py` file, and we never have to touch it
    #----------------------------------------------------------------------
    for filepath in Path('.').rglob('*.py'):
        fp_str = str(filepath)

        if fp_str not in [__file__, "__init__.py", "setup.py"]:
            # Convert filepaths to module paths
            module_name = fp_str[:-3]
            module_name = module_name.replace('/', '.')

            # Load module into python
            spec = importlib.util.spec_from_file_location(module_name, fp_str)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

            logging.debug(f"registered {fp_str}")
    #======================================================================

    main()
