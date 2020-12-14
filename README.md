# MacLean-1 SNN

An infrastructure for the creation and study of spiking neural networks.

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Documentation](#documentation)
4. [Contributing](#contributing)
5. [Credits](#credits)
6. [License](#license)

## Installation
These scripts must be run as source,
[ideally](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended)
in a Python virtual environment.

To get setup, clone the repository and follow the instructions below:

### Creating a [Python virtual environment](docs.python.org/3/tutorial/venv)
Enter the directory containing `setup.py` and run the following:

#### On Mac/Linux
```bash
python -m venv venv         # create virtual environment
source ./venv/bin/activate  # open virtual environment
pip install -e              # add this repository to the virtual environment
```

#### On Windows
```bash
python -m venv venv      # create virtual environment
./venv/Scripts/activate  # open virtual environment
pip install -e           # add this repository to the virtual environment
```

### Closing the virtual environment
When you're done working with the codebase, `deactivate` closes the virtual
environment.

### Troubleshooting
If at any point modules stop being recognized, delete the virtual environment
and repeat the above steps.

## Usage

### Summary Overview

### Creating Models

#### Rapid prototyping
While you're adding and removing structural elements, working with the HJSON
configuration could be more trouble than it's worth. It's advised you leave
your HJSON a skeleton and use default values in the class definition while
you're in this phase.

### Creating Trainers

### Configuring Experiments

### Running Experiments

### Loggers & TensorBoard












There are three parts to this section, each describing the same workflow from
a different perspective:
1. **[Overview](#overview):** explains the codebase in general terms,
   describing the roles and connections between modules, but light on specific
   examples.
2. **[Walkthrough](#walkthrough):** explains the codebase by walking through a
   sample experiment, mentioning the more general aspects only where needed.
3. **[Changes from version 1](#changes-from-version-1):** explains only the
   prominent shifts in workflow relative to the previous version.

### Overview
#### Console Flags
### Walkthrough
#### Creating new models
- `uni` must be the first argument after `self` due to how the HJSON parser works
  - (TODO: consider making a decorator to abstract all this away, like `@uni` or
    `@doeverythingthatsboilerplatery`)
### Changes from version 1






Repository structure is an adaptation of the
[Gemy Standard](https://github.com/MrGemy95/Tensorflow-Project-Template):


**THIS LAYOUT IS SLIGHTLY OUT OF DATE**
```
 maclean-snn
│
├── models
│  ├── base.py              - Base model inherited by experimental models
│  ├── *.py                 - Models used in experiments
│  └── neurons
│     ├── base.py           - Neuron model all other neurons inherit from
│     ├── adex.py           - Contains all AdEx neuron variants
│     └── lif.py            - Contains all LIF neuron variants
│
├── trainers
│  └── *.py                 - Model trainers
│
├── mains
│  └── *.py                 - Each experiment template can have its own main
│
├── data_loader
│  ├── data_generator.py    - Contains all our data generators (might split)
│  └── tools
│     └── *                 - Apparati to make files used by data generators
│
├── configs
│  └── *.hjson              - Model configs are set and stored in HJSON format
│
└── utils
   ├── logger.py            - Contains logger(s) to run during training
   ├── config.py            - Parser/loader for model configurations
   └── *.py                 - Any other misc. utilities go here
```

If you don't want to worry about modules (not recommended), just hard code
parameters in a `models/*.py` file and do everything else in a `mains/*.py`
file, like before. Otherwise, read on.

Additionally, most directories come with a template
demonstrating how to use that part of the repository.

### Build your architecture in `models/*.py`
These files are meant to be semi-experiment-specific network architectures.
Generic models, composed as best suits your need.

Everything here is probably going to be a class. Definitely avoid putting
values in here for parameters involved in the class instantiation. Those
should all be maintained in an associated `configs/*.hjson`.

### Set parameters in `configs`
These files determine experiment-specific variables, allowing for quick
adjustment to network and filesaving parameters without having to scroll
through miles of python code.

### Prepare training in `trainers`
Classes to train models. Should be self-explanatory.

### Generate data in `data_loader/data_generator.py`
This contains all the functions and classes required to generate TensorFlow
data for training. This should really only be for the generation of TensorFlow
formatted data, however: generation of novel stimuli or large datasets is the
purview of the `data_loader/tools/` directory, as that allows us to break off
our data generating tools for other projects, sans hassle.

### Verb noun in `utils`
For the most part, the only thing you'll need to worry about here is
`utils/logger.py`, where you can choose which logger you want. If you're
entering this directory for any other reason, you probably already know what
your looking for and what to do with it.

### Make science in `mains`
This is where the magic happens (by which I mean this is the entry point for
your code). If you're going with the modular approach, this file should really
only contain a few instantiations, a call to run training, and a post-training
report. Each `mains/*.py` file is best associated with a specific class of
experiment, allowing A) reuse; B) scalable allowance of multiple experiements
in just one group repository; and C) always-up-to-date access to any new
modules produced by other lab members, allowing easy exchange of models,
methods, and data.



## Documentation

### Inline
The source code contains extensive documentation, accessible via Python's
`help` function.

### External
No documentation presently exists beyond this repository's contents.

## Meet the Neurons

### Leaky Integrate-and-Fire (LIF)

_Description_

### Adaptive LIF (ALIF)

_Description_

### Adaptive Exponential Integrate-and-Fire (AdEx)

_Description_

### Etc.

_Description_


## Contributing


_TODO_
_TODO: use logging.x() instead of print() now_

### Best practices

### Style

### Issue system


### Things Chad Loses Sleep Over

- [ ] Having the `_class` tag hardcoded is a little icky

### Style Guidelines

This codebase (extremely) loosely adheres first to the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
**then** to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and
[PEP 257](https://www.python.org/dev/peps/pep-0257). Legibility is always the
most important, however.

Hard tabs are a sin.

Section headers and subheaders are encouraged.


## Credits

### Lab Members ([MacLean Lab](https://www.macleanlab.com/))
_The codebase in its current form is built and maintained by the MacLean lab._

Principle Investigator:
- Jason MacLean

Graduate Researchers:
- Yuqing Zhu

Undergraduate Researchers:
- Tarek Jabri
- Chadbourne Smith
- Mufeng Tang

### Maass Group, PI [W. Maass](https://igi-web.tugraz.at/people/maass/)
_TODO: explanation of credit_

**TODO**: looking for Maass' lab homepage

### Scientific articles
_The codebase wouldn't exist without prior contributions from the field, both
technical and theoretical._

_TODO: add content_


## License
This software and associated documentation files are open source, licensed
under the [MIT License](https://opensource.org/licenses/MIT).
