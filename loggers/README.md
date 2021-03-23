# Logging
The ultimate aim of this project is to gain insight into these networks. To
that end, it's often helpful to think of our training loop as doing nothing
more than populating our loggers, which store, record, and write any values
we're interested in.

_**Note:** there are also references to Python's_
[`logging`](docs.python.org/3/library/logging) _module scattered throughout
the code. Use of this module is also encouraged, but serves a function
distinct from our loggers: when you're interested in the state or products of
the model, use one of our loggers; when you're interested in the state of the
code, use Python's. Semantic meaning for these logging levels can be found
[here](https://stackoverflow.com/questions/2031163) and
[here](https://www.ibm.com/support/knowledgecenter/en/SSEP7J_10.2.2/com.ibm.swg.ba.cognos.ug_rtm_wb.10.2.2.doc/c_n30e74.html).
See_ `get_args()` _and_ `start_logger()` _in_ `\utils\config.py` _for further
documentation._

## The Base Class
All loggers inherit from `BaseLogger` in `\loggers\base.py`.

### Standard Methods
- `.add_callback()` appends a `tf.keras.callbacks.Callback` object to a list
  of callbacks internally maintained by the logger. The logger should pass
  these (or an appropriate subset) to `fit()` whenever `fit()` is called in
  training.

- `.summarize()` is called inside training loops to log scalar values, usually
  into an event file (located in the summaries directory specified by an HJSON
  configuration file), available for immediate analysis in TensorBoard or for
  later postprocessing into a more generic file format.

- `.post()` writes everything to disk.

## Output
There's a `npz` file with all the data in `logvars` per `.post()`. Access like so:

```python
import numpy as np

data=np.load('1-5.npz')

# check what np arrays are in the npz
for k in data.files:
    print(k)
    try:
        print(k.shape)
    except:
        print('(no shape)')
    print()
```

Some `np.arrays` of particular note in the `npz` files:
- `step` and `sw_epoch` have step and epoch values such that for any stepwise
  numpy array `a` in the npz file, `a[i]`'s data corresponds to step `step[i]`
  of epoch `epoch[i]`
- `ew_epoch` is the same as `sw_epoch` but for epochwise numpy arrays

Additionally, there's a `meta.pickle` and `config.pickle` file included beside
these with metadata about the variables in the `npz` files and the HJSON
configuration file, respectively. Access like so:

```python
import pickle

with open("meta.pickle", "rb") as file:
    meta = pickle.load(file)
with open("config.pickle", "rb") as file:
    cfg = pickle.load(file)

# check what's in the files
for k in meta.keys():
    print(k)
for k in cfg.keys():
    print(k)
```

## Pseudo-Callbacks
Each returns an action list. Currently supported keywords:
- `save_weights`: tells the trainer to save the model weights

## Simple Logging
Loggers are meant to store any and all variables you're interested in logging. These should be passed to the logger by
the training loop, and will be flushed to disk everytime the logger's `.post()` method is invoked. This can be done in
any manner, but is presently achieved by [pickling](https://docs.python.org/3/library/pickle.html) data and creating
plots. Output directories are to be specified in the HJSON configuration file.

## TensorBoard Compatible Logging
Support is offered for checkpoints, summaries, and logdirs.
