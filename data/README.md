# Data
Data generators provide the data used by our trainers, either wholesale or
piecemeal.

_**Note:** when referring to these objects, "generator" is being used more
loosely than in Python. Python generators, which dynamically generate data
(think_ `yield` _operator), are a subset of these generators, whose only
obligation is to procure data, not necessarily generate it "on the fly."_

## The Base Class
All data generators inherit from `BaseDataGenerator` in `\data\base.py`.

### Standard Methods
- `.get()` returns a complete dataset. If `.get()` is not implemented, `.next()`
  must be.

- `.next()` returns the next desired portion of a dataset (e.g. a batch). If
  `.next()` is not implemented, `.get()` must be.


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
