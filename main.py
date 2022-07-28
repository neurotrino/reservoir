from os.path import abspath
from pathlib import Path

import importlib.util
import logging
import utils.config


for filepath in Path('.').rglob('*.py'):
    """Hacky code to abstract imports.

    Python doesn't like it when you abstract imports, which is meant to
    encourage good readability standards. That said, in order to get
    everything working nicely for our use case(s), hacky it is.

    Why do we want to do this? It allows us to specify in our HJSON
    files the modules we want to use per experiment. We were already
    doing this, but now...
    1. we only need to change class names in one place (the HJSON file)
    2. we can run this script outside of virtual environments, meaning
       we don't have to worry about weird `pip install -e .` behaviors
    3. we only need one `main.py` file, and we never have to touch it
    """
    fp_str = str(filepath)

    if fp_str not in [__file__, "__init__.py", "setup.py"]:
        # Convert filepaths to module paths
        module_name = fp_str[:-3]
        module_name = module_name.replace('/', '.')

        # Call `import data.ccd` and so on
        exec(f'import {module_name}')
        print(f"registered {fp_str} as {module_name}")


def main():
    # Use command line arguments to load data, create directories, etc.
    cfg = utils.config.boot()
    logging.info("experiment directory: " + abspath(cfg['save'].exp_dir))

    data_module = eval(f"data.{cfg['data'].type}")

    # Build model
    while True:

        model_module = eval(f"models.{cfg['model'].type}")
        logger_module = eval(f"loggers.{cfg['log'].type}")
        trainer_module = eval(f"trainers.{cfg['train'].type}")

        model = model_module.Model(cfg)
        logging.info(f"instantiated {cfg['model'].type}.Model")

        # Load data
        data = data_module.DataGenerator(cfg)
        logging.info(f"instantiated {cfg['data'].type}.DataGenerator")

        # Instantiate logger
        logger = logger_module.Logger(cfg)
        logging.info(f"instantiated {cfg['log'].type}.Logger")

        # Instantiate trainer
        trainer = trainer_module.Trainer(cfg, model, data, logger)
        logging.info(f"instantiated {cfg['train'].type}.Trainer")

        # Train model
        trainer.train()
        logging.info("training complete")

if __name__ == '__main__':
    main()
