"""Load, process, and format session-relevant configuration settings."""

# external ----
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import argparse
import hjson
import inspect
import json
import logging
import os
import shutil
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
        '-c',            # short command-line flag
        '--config',      # long flag; becomes attribute (`args.config`)
        metavar='C',
        default='None',  # value used when user provides none
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
#┤ HJSON Parsing                                                             │
#┴───────────────────────────────────────────────────────────────────────────╯

def load_hjson_config(filepath, custom_save_cfg=None):
    """Read configuration settings in from an HJSON file.

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
        form, bundled_cfg:
          - form: function taking a model class (*not* an object) which
              instantiates that model
          - bundled_cfg: dictionary containing various non-model config
              settings
              - 'save': save config

    Raises:
        ValueError: if no experiment ID was provided
    """
    with open(filepath, 'r') as config_file:
        config = hjson.load(config_file)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Model Configuration                                                   │
    #┴───────────────────────────────────────────────────────────────────────╯

    # [!] Right now this is only applied to some fields (model)
    def recursively_make_namespace(src_dict):
        new_dict = {}
        for key in src_dict.keys():
            if type(src_dict[key]) == OrderedDict:
                new_dict[key] = recursively_make_namespace(src_dict[key])
            else:
                new_dict[key] = src_dict[key]
        return SimpleNamespace(**new_dict)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ File-Saving Configuration                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    # Check for script-based save settings
    if custom_save_cfg is None:
        save_cfg = config['save']
    else:
        save_cfg = custom_save_cfg
        logging.warning("HJSON save settings overwritten by script")

    # Check for null base directory
    if save_cfg['exp_dir'] is None:
        raise ValueError('cannot begin with null experiment directory')

    # Directory for this experiment
    if save_cfg['timestamp']:
        s = "%Y-%m-%d %H.%M.%S"
        s = datetime.utcfromtimestamp(time.time()).strftime(s)
        s = " [" + s + "]"
        save_cfg['exp_dir'] += s

    # Check if the experiment directory already exits
    if os.path.exists(save_cfg['exp_dir']):
        logging.info(f"{os.path.abspath(save_cfg['exp_dir'])} already exists")

        # Check if we're okay writing into this directory
        if save_cfg['avoid_overwrite']:
            # Append a number at the end of the filepath so it's unique
            original = save_cfg['exp_dir']
            unique_id = 1

            while os.path.exists(save_cfg['exp_dir']):
                save_cfg['exp_dir'] = original + f"_{unique_id}"
                unique_id += 1

            logging.warning(
                "renamed output directory to avoid overwriting data"
            )
        else:
            # Alert the user that we'll be writing into this directory
            if save_cfg['hard_overwrite']:
                # Remove the preexisting directory
                logging.warning(f"purging old data in {save_cfg['exp_dir']}")
                shutil.rmtree(save_cfg['exp_dir'])
            else:
                logging.warning(
                    f"potentially overwriting data in {save_cfg['exp_dir']}"
                )

    # Instantiate subdirectories
    for subdir in save_cfg['subdirs']:
        sd_path = save_cfg['subdirs'][subdir]

        # Ignore null filepaths
        if sd_path is None:
            # Alert the user we found a null filepath
            logging.warning(f"{subdir} is null")
            continue;

        # Create directories
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
    #┤ Data Configuration                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    data_cfg = config['data']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Logging Configuration                                                 │
    #┴───────────────────────────────────────────────────────────────────────╯

    log_cfg = config['log']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Configuration                                                │
    #┴───────────────────────────────────────────────────────────────────────╯

    train_cfg = config['train']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Miscellaneous Configuration                                           │
    #┴───────────────────────────────────────────────────────────────────────╯

    misc_cfg = config['misc']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Packaging                                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    bundled_cfg = {
        'model': recursively_make_namespace(config['model']),

        'save': SimpleNamespace(**save_cfg),

        'data': SimpleNamespace(**data_cfg),
        'log': SimpleNamespace(**log_cfg),
        'train': SimpleNamespace(**train_cfg),

        'misc': recursively_make_namespace(misc_cfg)
    }

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Model Instantiator                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def form(template):
        """Instantiates a model according to a configured template."""

        # Create a model from the provided template
        try:
            def cfg_to_class(template, actual):
                """Create nested models from a JSON dictionary."""

                # peel off class data
                c = template['_class']
                del template['_class']

                logging.debug(f'parsing HJSON for {c}')

                # reformat provided data into the specified class
                if 'cfg' in actual:
                    # if the class is requesting access to the
                    # configuration data, provide it, allowing it to be
                    # referenced in the class' instantiation function
                    #
                    # classes requestion the configuration data must
                    # have it as one of their positional arguments
                    del actual['cfg']
                    m = c(cfg=bundled_cfg, **actual)
                else:
                    m = c(**actual)

                # recurse
                for k in template.keys():
                    x = cfg_to_class(template[k], actual[k])
                    setattr(m, k, x)
                return m

            model_cfg = json.loads(hjson.dumpsJSON(config['model']))
            model = cfg_to_class(template, model_cfg)
        except Exception as e:
            logging.critical('failed to instantiate model from HJSON')
            raise Exception(e, 'issue instantiating model from HJSON')

        logging.info(f'instantiated {type(model)} from HJSON')
        return model

    return form, bundled_cfg

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
        form, cfg = load_hjson_config(args.config)
    except Exception as e:
        raise Exception(e, "issue parsing HJSON")

    # Save a copy of this file, if the flags are set
    if cfg['save'].log_config:
        shutil.copyfile(args.config, cfg['save'].exp_dir + "/config.hjson")

    return form, cfg
