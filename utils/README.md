# Utilities

## Commandline Interaction
The basic call to run a script is as follows:

```bash
python main.py mains/some_main.py -c configs/some_config.hjson
```

The commandline parser is defined by `get_args()` in `config.py`. The following flags are currently supported:
- Mandatory:
    - `-c` or `--config`: path to HJSON configuration file defining the experimental parameters
- Optional:
    - `-lc` or `--log-level-console`: logging level for console output. Defaults to `INFO`.
    - `-lf` or `--log-level-file`: logging level for log file output. Defaults to `OFF`.
    - `-lo` or `--log-output`: filename of the log output file. Defaults to `latest.log`.

