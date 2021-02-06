# Datasets

_**Note:** "generator" is being used here more loosely than in Python. Python
generators are a subset of these generators, which are more like "procurers."
There is no need for (nor restriction against) these generators dynamically
producing data. TensorFlow 2.0 doesn't use_ `yield` _very frequently._

## The Base Class

All datasets should be wrapped in a class inheriting from `BaseDataGenerator`
in `data\base.py`.

### Standard Methods
`.get()` should return a complete dataset. If `.get()` is not implemented,
`.next()` must be.

`.next()` should return the next desired segment of the dataset (e.g. a
batch). If `.next()` is not implemented, `.get()` must be.


## Standard Functions
Every data script should contain a `load_data()` function which returns a
"plug and play" dataset, for testing and experimenting convenience.


## Iteration

How data should be processed varies with the data itself, but there are three
broad approaches best supported by this framework:

    1. Data generators can define their own iterators
    2. Data can be statefully accessed from the generator
    3. Data can be created all at once using `load_data()`

See the documentation for trainers for more on interating through data during
training.


## Pre and Post -Processing

Data scripts should also include any pre/post -processing infrastructure
specific to that dataset.
