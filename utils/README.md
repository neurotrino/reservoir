# Utilities
Various miscellaneous utilities are required for everything else to run smoothly.


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


## HJSON Parsing
One of the core design principles in this project is to have the parameters of an experiment controlled through
semi-interchangeable HJSON files. `load_hjson_config()` in `config.py` handles HJSON deserialization.

In brief, the entire tree structure of the specified HJSON file is converted into a dictionary of namespaces, `cfg`,
which is typically passed to each module in a `main.py` file (see `mains\sinusoid_example.py`). Below is an example of
accessing data in `cfg`:

```python
output_dir = self.cfg['save'].main_output_dir
```

Note how the primary namespace is a dictionary key while all values therein are namespace attributes (this is
recursive, i.e. `self.cfg['save'].a.b` is a valid call, `self['save']['a'].b` is not). See `configs\README.md` for a
more comprehensive breakdown of these namespaces.

The only field which receives special processing during deserialization is the `model` namespace. In `cfg`, there is
no noticeable difference. The difference lies in model instantiation. `boot()` in `config.py` returns a curried
function `form`, needed to instantiate the actual model. This requires special processing of the `model` namespace,
which in turn is why a template must be provided in the `main.py` script.
