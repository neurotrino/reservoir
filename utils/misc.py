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
                f"expected boolean flag, got value '{new_val}' of type "
                + f"{type(new_val)}"
            )
        self._enabled = new_val


def get_experiments(data_dir, experiment_string, final_npz=None):
    """Get a list of filenames for all experiments that contain a
    desired string.

    Experiments begun on 2022-07-25 (containing string) are defunct.

    Do not include experiments that have not yet completed 1000 epochs
    of training.
    """
    exp_dirs = []
    null_date = "2022-07-25"
    for fname in os.listdir(data_dir):
        if not fname.startswith(experiment_string):
            continue
        if null_date in fname:
            continue
        if final_npz is None:
            if os.path.exists(
                os.path.join(data_dir, fname, "npz-data", "991-1000.npz")
            ):
                exp_dirs.append(os.path.join(data_dir, fname))
        else:
            if os.path.exists(
                os.path.join(data_dir, fname, "npz-data", final_npz)
            ):
                exp_dirs.append(os.path.join(data_dir, fname))
    return exp_dirs


def filenames(num_epochs, epochs_per_file, final_npz=None):
    """Get the filenames storing data for epoch ranges.
    Our data is stored in npz files titled 'x-y.npz' indicating that
    file contains the data for epochs x through y, inclusive. For
    example, 1-10.npz has all the data associated with the first 10
    epochs of an experiment.
    """
    if final_npz is None:
        return [
            f"{i}-{i + epochs_per_file - 1}.npz"
            for i in range(1, num_epochs, epochs_per_file)
        ]
    else:
        fnames = [
            f"{i}-{i + epochs_per_file - 1}.npz"
            for i in range(1, num_epochs, epochs_per_file)
        ]
        end_idx = np.where(np.char.equal(fnames,final_npz))[0][0]
        return fnames[:end_idx+1]


def generic_filenames(num_epochs, epochs_per_file):
    return [
        f"{i}-{i + epochs_per_file - 1}.npy"
        for i in range(1, num_epochs, epochs_per_file)
    ]
