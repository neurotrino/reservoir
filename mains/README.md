# Scripts
These are the point of entry into the code. They follow a very standard format, with the main source of variation being
in the `template` variable, which is necessary to specify the classes of subclasses in the model. This must align with
the HJSON configuration for the model.

Brief descriptions of each component are described below. See the associated subdirectories for further documentation.

## Setup
First, `form` and `cfg` are generated. `form` is a function which must be used to instantiate the model, based on the
template. `cfg` is a six-key dictionary mirroring the provided HJSON configuration file. Each value in this dictionary
is a `SimpleNamespace` of either primitives or `SimpleNamespaces` (or more complex objects in `cfg['model']`).

## Model Instantiation
Calling `form(template).build(cfg)` instantiates your model. The template itself is used to specify the classes of
models and submodels using the `_class` tag, which is reserved in this context.

## Data Generation
Data generation is expected to be varied, but should always provide either an iterable (object) or an iterator
(method).

## Logging & Training
Logging and training are intertwined. It's often best to think of training as populating your logger. Training itself
should only require a call to the `.train()` method.
