# MacLean Lab SNN Infrastructure
An infrastructure for the creation and study of spiking neural networks.


## Installation
These scripts must be run as source,
[ideally](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended) in a Python virtual
environment. To get started, clone this repository and follow the instructions below.

### Creating a [Python virtual environment](docs.python.org/3/tutorial/venv)
`setup.py` will create in the virtual environment `msnn-custom-evars.pth`,
which in turn creates particular environmental variables any time that
virtual environment is active. For a list of such variables, while in the
virtual environment, run a Python script like

```Python
import os

for k in os.environ:
    if 'MSNN' in k:
        print(f"{k} == {os.environ[k]}")
```

At any point in the code, you can call reference these values as
`os.environ['MSNN_GITSHA']`, for example (this feature was originally added so
we would always have a way of identifying which version the virtual
environment was running).

Enter the directory containing `setup.py` and run the following:

#### With a Conda Environment
```bash
conda activate tf2gpu
pip install -e .
```

#### On Mac/Linux (no Conda)
```bash
python -m venv venv         # create virtual environment
source ./venv/bin/activate  # open virtual environment
pip install -e .            # add this repository to the virtual environment
```

#### On Windows (no Conda)
```bash
python -m venv venv      # create virtual environment
./venv/Scripts/activate  # open virtual environment
pip install -e .         # add this repository to the virtual environment
```

### Closing the virtual environment
When you're done running experiments, `deactivate` closes the virtual environment.

## Usage
The core principle of this infrastructure design is to create modular components useful to SNN construction and
analysis easily configurable (once initially constructed) via minimal changes in scripts and tweaking in HJSON files.
To this end, it is recommended you fork the repository when designing new components, then reintroduce them after
something crystallizes.

### Overview
The overarching repository structure is designed around streamlining collective contribution and modularity, drawing
heavily from [MrGemy95](https://github.com/MrGemy95/Tensorflow-Project-Template)'s machine learning project template:

```
maclean-snn
├──configs   # Experiment configuration files
├──data      # Data generators
├──loggers   # Loggers
├──mains     # Entry points
├──models    # Model structures
├──trainers  # Model trainers
└──utils     # Miscellaneous utilities
```

Within each sudirectory is a README explaining said module's functionality and development paradigms.

### Quick Start
If you just want to run a model, execute the following:

```bash
python main.py -c config.hjson
```

where `main.py` is a script located in `\mains` and `config.hjson` is an experiment configuration file located in
`\configs`. See the README in `\utils` for more documentation on commandline interactions.

### Troubleshooting
- **Local modules aren't being recognized**
  1. Delete the virtual environment and repeat the installation steps


## Documentation
Documentation for this project is contained in this README, a README in each subdirectory, and extensive inline
documentation throughout the `.py` scripts.


## Contributing

### Note on Rapid prototyping
While you're adding and removing structural elements, working with the HJSON
configuration could be more trouble than it's worth. It's advised you leave
your HJSON a skeleton and use default values in the class definition while
you're in this phase.

### Common Modifications
While the script is designed to ultimately provide a model you can control through manipulation of just one HJSON
file, there are many changes you will often find yourself wanting to make in the Python scripts while building the
initial model.

#### Flags and Toggles
If you want to toggle some behavior in data generation, logging, et cetera, you need to (1) add a flag to the HJSON
script and (2) encapsulate the toggled behavior in an `if` guard. For example, say you want to toggle logging `.npz`
files. Add something like the following to your HJSON script:

```
log:
{
    log_npz: true  # new flag, with the behavior enabled
}
```

Then, in your logger, under `.post()`:

```python
if self.cfg['log'].log_npz:
    # log npz files here
```

### Style Guidelines
When in doubt, refer first to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), then
to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257).


## Credits
- MacLean Lab (maintainers)
- Maass Group (collaborators)


## License
This software and associated documentation have been made open source under the
[MIT License](https://opensource.org/licenses/MIT).
