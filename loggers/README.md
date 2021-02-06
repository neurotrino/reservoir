# Logging

## The Base Class

All loggers inherit from `BaseLogger` in `\loggers\base.py`.


## Logging by Value

### Scalars
Scalar values should always be logged using the `.summarize()` method.

### Tensors
It's typically easiest to convert tensors to numpy arrays, then hold on to
them for a period of time (perhaps the whole training) and perform any
processing at then end, then saving them.

### Checkpoints and Network Weights

### Plots


## Logging Paradigms
Broadly, the logging API supports two mechanisms for logging data

### Keras Callbacks
It's possible to use [Keras callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
for logging, much like we did in our TensorFlow 1.0 framework (with the added
step of attaching them to a logger). These are easy to setup, but their
granularity is pretty coarse, limiting their usage. See `\loggers\callbacks\`
for further documentation.

### Custom Methods
Loggers should contain methods which make the lines of logging code in the
training loop as concise as possible (e.g. the `.summarize()` method).
Additionally, you can effectively create more powerful versions of Keras
callbacks by creating a method to be called at every juncture X in the
training; unlike Keras callbacks, these can have arbitrary access to your
model state, based on what their parameters are.


## Concluding a Logging Session

Loggers will clean themselves up.
