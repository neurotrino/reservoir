# Logging

## The Base Class

All loggers inherit from `BaseLogger` in `loggers\base.py`.


## Logging by Value

### Scalars

Scalar values should always be logged using the `summarize()` method.

### Tensors

### Plots


## Logging by Paradigm

### Logging with Callbacks

### Logging with Methods

### Logging within the Training Loop

While the above options nicely preserve modularity, sometimes the information
you want to log is too granular for callbacks and doesn't really need its own
method.


## Concluding a Logging Session

Loggers will clean themselves up.



## See Also
