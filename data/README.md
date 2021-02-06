# Datasets

## The Base Class

All datasets should be wrapped in a class inheriting from `BaseDataGenerator`
in `data\base.py`.

`.get()` and `.next()` are the standard methods for the class. At least one
must be implemented in any given data generator. The former should `return` a
complete dataset, the latter should `yield` the next desired segment of the
dataset (e.g. a batch).

_Note: "generator" is being used here more loosely than in Python. Python
generators are a subset of these generators, which are more like "procurers."
There is no need for (nor restriction against) these generators dynamically
producing data._


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
