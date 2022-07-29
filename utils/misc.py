# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

"""Generic wrapper to make decorators switchable."""

class SwitchedDecorator:
    """Wrapper providing a programmatic toggle for decorators."""

    def __init__(self, decorator_fn, enabled=True):
        """Instantiate object."""
        self._enabled = enabled
        self._decorator_fn = decorator_fn

    def __call__(self, target):
        """Runs the function (un)decorated if (not) enabled."""
        if self._enabled:
            return self._decorator_fn(target)
        return target

    @property
    def enabled(self):
        """Flag indicating decorator status."""
        return self._enabled

    @enabled.setter
    def enabled(self, new_val):
        """Setter to update the `enabled` flag."""
        if not isinstance(new_val, bool):
            raise ValueError(
                f"expected boolean flag, got value '{new_val}' of type " +
                f"{type(new_val)}"
            )
        self._enabled = new_val

def get_experiments(data_dir, experiment_string):
    """Get a list of filenames for all experiments that contain a desired string"""
    exp_dirs = []
    for file in os.listdir(data_dir):
        if file.startswith(experiment_string):
            exp_dirs.append(os.path.join(data_dir,file))
    return exp_dirs

def filenames(num_epochs, epochs_per_file):
    """Get the filenames storing data for epoch ranges.
    Our data is stored in npz files titled 'x-y.npz' indicating that
    file contains the data for epochs x through y, inclusive. For
    example, 1-10.npz has all the data associated with the first 10
    epochs of an experiment.
    """
    return [
        f"{i}-{i + epochs_per_file - 1}.npz"
        for i in range(1, num_epochs, epochs_per_file)
    ]
