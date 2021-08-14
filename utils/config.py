"""Load, process, and format session-relevant configuration settings."""

# external ----
from collections import OrderedDict
from datetime import datetime
from tensorflow.python.client import device_lib
from types import SimpleNamespace

import argparse
import hjson
import json
import logging
import os
import shutil
import tensorflow as tf
import time


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Python (Not TensorFlow) Logger                                            │
#┴───────────────────────────────────────────────────────────────────────────╯

def start_logger(clevel_str, flevel_str, fpath, writemode='w+'):
    """Initialize and configure the *vanilla* Python logger.

    Provided three configuration strings, set the logging output levels
    for the terminal and filewriter, whose output path can also be
    specified.

    args:
      clevel_str: logging level for terminal logging. Value must
        be "DEBUG", "INFO", "WARN", "CRITICAL", or "OFF".

      flevel_str: logging level for file logging. Value must be
        "DEBUG", "INFO", "WARN", "CRITICAL", or "OFF".

      fpath: filepath file logging will be written into.

      writemode: specified how the filehandler writes to file (e.g.
        'w+'). See `logging.FileHandler` for further documentation.
    """
    logger = logging.getLogger()
    # [!] current also turns on external module logging :/

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-9s- %(message)s', '%Y-%m-%d %H:%M:%S'
    )

    # Console logging
    if clevel_str != 'OFF':
        clevel = eval(f'logging.{clevel_str}')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(clevel)

        logger.addHandler(ch)
    else:
        clevel = logging.CRITICAL + 1

    # File logging
    if flevel_str != 'OFF':
        flevel = eval(f'logging.{flevel_str}')

        fh = logging.FileHandler(fpath, writemode)
        fh.setFormatter(formatter)
        fh.setLevel(flevel)

        logger.addHandler(fh)
    else:
        flevel = logging.CRITICAL + 1

    logger.setLevel(min(clevel, flevel))


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Command Line Parsing                                                      │
#┴───────────────────────────────────────────────────────────────────────────╯

def get_args():
    """Parse command line arguments parameterizing the session.

    Returns:
      An argument parser with the following namespaces:
      - config: string of relative filepath to HJSON config file
    """
    parser = argparse.ArgumentParser(description="Configure session")

    # Path to HJSON configuration file specifying session variables
    parser.add_argument(
        '-c',          # short command-line flag
        '--config',    # long flag; becomes attribute (`args.config`)
        metavar='C',
        default=None,  # value used when user provides none
        help='path to HJSON configuration file specifying session variables'
    )

    # Console-logging level
    parser.add_argument(
        '-lc',
        '--log-level-console',
        metavar='LC',
        default='INFO',
        help='Console logging level (DEBUG, INFO, WARN, ERROR, CRITICAL, OFF)'
    )

    # File-logging level
    parser.add_argument(
        '-lf',
        '--log-level-file',
        metavar='F',
        default='OFF',
        help='File logging level (DEBUG, INFO, WARN, ERROR, CRITICAL, OFF)'
    )

    # File-loggging output
    parser.add_argument(
        '-lo',
        '--log-output',
        metavar='LO',
        default='latest.log',
        help='File logging output'
    )

    return parser.parse_args()


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ HJSON Parsing and Configuration Operations                                │
#┴───────────────────────────────────────────────────────────────────────────╯

def recursively_make_namespace(src_dict):
    """Convert each key into a namespace. Recurse if the key leads to a
    dictionary value.
    """
    new_dict = {}
    for key in src_dict.keys():
        if type(src_dict[key]) == OrderedDict:
            new_dict[key] = recursively_make_namespace(src_dict[key])
        else:
            new_dict[key] = src_dict[key]
    return SimpleNamespace(**new_dict)

    model_cfg = recursively_make_namespace()


def subconfig(cfg, subcfg, old_label='model', new_label='cell'):
    """Create a new configuration for a submodel or layer.

    This is done to preserve abstraction, so that when designing a
    model or layer, it doesn't matter how nested it is in the initial
    model call. E.g. you don't have to code
    `cfg['model'].submodel.submodel.param` in the class definition.
    """
    new_cfg = cfg.copy()  # create deep copy to avoid weirdness

    new_cfg.pop(old_label)       # preserve encapsulation
    new_cfg[new_label] = subcfg  # preserve abstraction/generalization

    return new_cfg  # configuration for use by sub- model/layer


def load_hjson_config(filepath):
    """Read configuration settings in from an HJSON file.

    UPDATE: no longer does any model instantiation

    Reads an HJSON file into various formats. Model configurations are
    stored as strings, allowing easy instantiation of models with the
    `dataclasses_json` library, which save configurations are simple
    dictionaries.

    Make sure your HJSON file correctly encodes your network and save
    settings. Look at the template HJSON file provided in the repository
    for further explanation and an example.

    If you prefer manipulating the save configuration from a python
    script, `custom_save_cfg` will overwrite the HJSON values.

    Args:
        filepath: string of relative filepath to HJSON config file

    Returns:
        bundled_cfg:
          - bundled_cfg: dictionary containing various non-model config
              settings
              - 'save': save config

    Raises:
        ValueError: if no experiment ID was provided
    """
    with open(filepath, 'r') as config_file:
        config = hjson.load(config_file)  # read HJSON from filepath

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Configuration Steps for File-Saving Settings                  │
    #┴───────────────────────────────────────────────────────────────────────╯

    save_cfg = config['save']

    # Check for null base directory
    if save_cfg['exp_dir'] is None:
        raise ValueError('cannot begin with null experiment directory')

    # Directory for this experiment
    if save_cfg['timestamp']:
        # Timestamp the experiment directory if requested
        s = "%Y-%m-%d %H.%M.%S"
        s = datetime.utcfromtimestamp(time.time()).strftime(s)
        s = " [" + s + "]"
        save_cfg['exp_dir'] += s

    # Check if the experiment directory already exits
    if os.path.exists(save_cfg['exp_dir']):
        logging.info(f"{os.path.abspath(save_cfg['exp_dir'])} already exists")

        # If we're *not* okay overwriting this directory...
        if save_cfg['avoid_overwrite']:
            # Append a number at the end of the filepath so it's unique
            original = save_cfg['exp_dir']
            unique_id = 1

            while os.path.exists(save_cfg['exp_dir']):
                save_cfg['exp_dir'] = original + f"_{unique_id}"
                unique_id += 1

            # Inform the user that we've selected a new directory to
            # save into
            logging.warning(
                "renamed output directory to avoid overwriting data"
            )

        # If we *are* okay overwriting this directory...
        else:
            # Set the save path to the existing directory
            fullpath = os.path.abspath(save_cfg['exp_dir'])

            # Alert the user that we'll be writing into this directory
            if save_cfg['hard_overwrite']:
                # Remove the preexisting directory
                logging.warning(f"purging old data in {fullpath}")
                shutil.rmtree(save_cfg['exp_dir'])
            else:
                # Warn the user that new data will be mixed in with the
                # old data and might overwrite preexisting files
                logging.warning(f"potentially overwriting data in {fullpath}")

    # Instantiate subdirectories
    for subdir in save_cfg['subdirs']:
        sd_path = save_cfg['subdirs'][subdir]

        # Ignore null filepaths
        if sd_path is None:
            # Alert the user we found a null filepath
            logging.warning(f"{subdir} is null")
            continue

        # Create directories within the experiment directory
        save_cfg[subdir] = os.path.join(
            save_cfg['exp_dir'],
            save_cfg['subdirs'][subdir]
        )
        try:
            if not os.path.exists(save_cfg[subdir]):
                os.makedirs(save_cfg[subdir])
        except Exception as err:
            print("Error creating directories: {0}".format(err))
            raise Exception(err)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Finalize Configuration Settings                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    cfg = {
        'model': recursively_make_namespace(config['model']),
        'save': recursively_make_namespace(config['save']),
        'data': recursively_make_namespace(config['data']),
        'log': recursively_make_namespace(config['log']),
        'train': recursively_make_namespace(config['train']),
        'misc': recursively_make_namespace(config['misc'])
    }
    return cfg


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Startup Boilerplate                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯

def boot():
    """Parse command line arguments and HJSON configuration files."""

    # Parse command-line arguments
    try:
        args = get_args()
    except:
        raise Exception("missing or invalid arguments")

    # Initialize the vanilla python logger (configured in command line)
    start_logger(
        args.log_level_console,
        args.log_level_file,
        args.log_output
    )

    # Parse HJSON configuration files
    try:
        cfg = load_hjson_config(args.config)
    except Exception as e:
        raise Exception(e, "issue parsing HJSON")

    # Save a copy of this file, if the flags are set
    if cfg['save'].log_config:
        shutil.copyfile(args.config, cfg['save'].exp_dir + "/config.hjson")

    # Log the git SHA of the commit in use by the virtual environment
    logging.debug(f"venv running from commit SHA {os.environ['MSNN_GITSHA']}")

    # Check for a GPU
    device_name = tf.test.gpu_device_name()

    if not device_name:
        logging.warning('GPU device not found')
        logging.debug(
            'output of device_lib.list_local_devices(): '
            + f'{device_lib.list_local_devices()}'
        )
        logging.debug(
            'output of tf.config.list_physical_devices(): '
            + f'{tf.config.list_physical_devices()}'
        )
    else:
        logging.debug(f'found GPU at {device_name}')

    #=================================
    print('\nA\n')

    from pathlib import Path

    import importlib.util

    for filepath in Path('.').rglob('*.py'):

        fp_str = str(filepath)

        if not (fp_str.endswith("__init__.py") or fp_str == __file__):
            # Convert filepath syntax to modulepath syntax
            module_name = fp_str[:-3]
            module_name = module_name.replace('/', '.')

            # Load module into python
            spec = importlib.util.spec_from_file_location(module_name, fp_str)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
    print('\nB\n')

    #=================================

    # Return configuration settings
    return cfg
