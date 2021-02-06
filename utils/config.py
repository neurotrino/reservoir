"""Load, process, and format session-relevant configuration settings."""

# external ----
import argparse
import hjson
import inspect
import json
import logging
import os
import time

from datetime import datetime
from shutil import copyfile
from types import SimpleNamespace

# local -------
import utils.dirs

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
        help='Console logging level (DEBUG, INFO, WARN, CRITICAL, OFF)'
    )

    # File-logging level
    parser.add_argument(
        '-lf',
        '--log-level-file',
        metavar='F',
        default='OFF',
        help='File logging level (DEBUG, INFO, WARN, CRITICAL, OFF)'
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
    #┤ File-Saving Configuration                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    # Check for script-based save settings
    if custom_save_cfg is None:
        save_cfg = config['save']
    else:
        save_cfg = custom_save_cfg
        logging.warning("HJSON save settings overwritten by script.")

    # Ensure all outputs are assigned a valid save directory
    if None in [
        save_cfg['host_dir'],  # directory for all related experiments
        save_cfg['exp_dir'],
        save_cfg['checkpoint_dir'],
        save_cfg['summary_dir'],
        save_cfg['tb_logdir']
    ]:
        raise ValueError('cannot begin with null save directories')

    # Directory for this experiment
    if save_cfg['timestamp']:
        s = "%Y-%m-%d %H.%M.%S"
        s = datetime.utcfromtimestamp(time.time()).strftime(s)
        s = " [" + s + "]"
        save_cfg['exp_dir'] += s

    save_cfg['exp_dir'] = os.path.join(
        save_cfg['host_dir'], save_cfg['exp_dir']
    )

    if os.path.exists(save_cfg['exp_dir']):
        if save_cfg['avoid_overwrite']:
            original = save_cfg['exp_dir']
            unique_id = 1

            while os.path.exists(save_cfg['exp_dir']):
                save_cfg['exp_dir'] = original + f"_{unique_id}"
                unique_id += 1

            logging.warning(
                "Renamed output directory to avoid overwriting data."
            )
        else:
            logging.warning(
                f"Potentially overwriting data in {save_cfg['exp_dir']}"
            )


    # Directory for model checkpoints from this experiment
    save_cfg['checkpoint_dir'] = os.path.join(
        save_cfg['exp_dir'], save_cfg['checkpoint_dir']
    )

    # Directory for summary files from this experiment
    save_cfg['summary_dir'] = os.path.join(
        save_cfg['exp_dir'], save_cfg['summary_dir']
    )

    # Directory for TensorBoard logdirs from this experiment
    save_cfg['tb_logdir'] = os.path.join(
        save_cfg['exp_dir'], save_cfg['tb_logdir']
    )

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

    misc = config['misc']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Packaging                                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    bundled_cfg = {
        'save': SimpleNamespace(**save_cfg),

        'data': SimpleNamespace(**data_cfg),
        'log': SimpleNamespace(**log_cfg),
        'train': SimpleNamespace(**train_cfg),

        'misc': SimpleNamespace(**misc)
    }

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Model Instantiator                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def form(template):
        """Instantiates a model according to a configured template."""

        # Create a model from the provided template
        try:
            def cfg_to_class(template, actual):
                """Create nested models from a JSON dictionary.

                TODO: make docs pretty
                - template is dict (see main)
                - actual is also dict, but diff (I think)
                """

                # peel off class data
                c = template['_class']
                del template['_class']

                logging.debug(f'Parsing HJSON for {c}...')

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
            logging.critical('Failed to instantiate model from HJSON.')
            raise Exception(e, 'issue instantiating model from HJSON.')

        logging.info(f'Instantiated {type(model)} from HJSON.')
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

    # Create output directories
    utils.dirs.create_dirs(cfg['save'])

    # Save a copy of this file, if the flags are set
    if cfg['save'].log_config:
        copyfile(args.config, cfg['save'].exp_dir + "\\config.hjson")

    return form, cfg
