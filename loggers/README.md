# Logging

## The Base Class

All loggers inherit from `BaseLogger` in `\loggers\base.py`.


## Logging by Value

### Scalars
Scalar values should always be logged using the `.summarize()` method.

### Tensors

### Plots


## Logging by Paradigm

### Logging with Keras Callbacks
It's possible to use [Keras callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
for logging, much like we did in our TensorFlow 1.0 framework (with the added
step of attaching them to a logger). These are easy to setup, but their
granularity is pretty coarse, limiting their usage. See `\loggers\callbacks\`
for further documentation.

### Logging with Methods


### Logging within the Training Loop

While the above options nicely preserve modularity, sometimes the information
you want to log is too granular for callbacks and doesn't really need its own
method.


## Concluding a Logging Session

Loggers will clean themselves up.
