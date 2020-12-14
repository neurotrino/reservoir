"""Load, process, and format session-relevant configuration settings."""

# external ----
import argparse
import hjson
import inspect
import json
import logging
import os

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

def load_hjson_config(filepath):
    """Read configuration settings in from an HJSON file.

    Reads an HJSON file into various formats. Model configurations are
    stored as strings, allowing easy instantiation of models with the
    `dataclasses_json` library, which save configurations are simple
    dictionaries.

    Make sure your HJSON file correctly encodes your network and save
    settings. Look at the template HJSON file provided in the repository
    for further explanation and an example.

    Args:
        filepath: string of relative filepath to HJSON config file

    Returns:
        form, meta_cfg:
          - form: function taking a model class (*not* an object) which
              instantiates that model
          - meta_cfg: dictionary containing various non-model config
              settings
              - 'save': save config

    Raises:
        ValueError: if no experiment ID was provided
    """
    with open(filepath, 'r') as config_file:
        config = hjson.load(config_file)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Model Instantiator                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    # Load universal parameters into the `uni` namespace
    uni_cfg = config['uni']
    uni = json.loads(
        hjson.dumpsJSON(uni_cfg),
        object_hook=lambda x: SimpleNamespace(**x)
    )

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
                if 'uni' in actual:
                    # if the class is requesting access to the `uni`
                    # attribute, provide the fully-instantiated `uni`
                    # object, allowing it to be referenced in the
                    # class' instantiation function.
                    #
                    # note, this does mean classes using `uni` need to
                    # have it as one of their positional arguments, but
                    # that's just how Python rolls (i.e. not my fault)
                    del actual['uni']
                    m = c(uni=uni, **actual)
                else:
                    # create class without `uni` attribute
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

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ File-Saving Configuration                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    save_cfg = config['save']  # Data-saving config (dict)

    def append_where_value_exists(so_far, fallback, some_or_none):
        """Extend a filepath, either with some value or a fallback."""
        if some_or_none is None:
            return os.path.join(so_far, fallback)
        else:
            return os.path.join(so_far, some_or_none)

    # Directory containing all experiments
    dir_path = append_where_value_exists(
        '', '../experiments', save_cfg['experiment_dir']
    )

    # Directory of this single experiment
    if save_cfg['exp_id'] is None:
        raise ValueError('cannot begin without an experiment ID')
    else:
        dir_path = os.path.join(dir_path, save_cfg['exp_id'])

    save_cfg['summary_dir'] = append_where_value_exists(
        # Directory of summary data from this experiment
        dir_path, 'summary/', save_cfg['summary_dir']
    )
    save_cfg['checkpoint_dir'] = append_where_value_exists(
        # Directory of checkpoint data from this experiment
        dir_path, 'checkpoint/', save_cfg['checkpoint_dir']
    )
    save_cfg['log_dir'] = append_where_value_exists(
        # Directory of logdirs from this experiment
        dir_path, 'logdir/', save_cfg['log_dir']
    )

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Training Configuration                                                │
    #┴───────────────────────────────────────────────────────────────────────╯

    train_cfg = config['train']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Logging Configuration                                                 │
    #┴───────────────────────────────────────────────────────────────────────╯

    log_cfg = config['log']

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Packaging                                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    meta_cfg = {
        'uni': uni,  # universal values
        'train': SimpleNamespace(**train_cfg),
        'save': SimpleNamespace(**save_cfg),    # controls file-saving
        'log': SimpleNamespace(**log_cfg)       # controls logging
    }
    return form, meta_cfg

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

    return form, cfg
