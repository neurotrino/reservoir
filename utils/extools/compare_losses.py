"""Plot loss data from network output files."""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os

# internal ----
from utils.misc import filenames

data_dir = "/data/experiments/"
num_epochs = 300
epochs_per_file = 10

fwd_experiments = [
    "fwd-input-img-15x-trainable",
    "fwd-input-img-15x-fixed",
    "fwd-main-rewire-img-15x-fixed",
]
experiments = [
    "ccd_200_lif_sparse",
    "ccd_200_lif_rewiring",
    "ccd_500_lif_sparse",
    "ccd_500_lif_rewiring",
]
savepath = "/data/results/fwd/input-main-rewire.png"


def compare_losses(
    savepath=savepath,
    data_dir=data_dir,
    experiments=fwd_experiments,
    num_epochs=300,
    epochs_per_file=10,
    loss_of_interest="epoch_loss",
    title="Fwd Implementation",
    xlabel="epochs",
    ylabel="loss",
    legend=[
        "new input trainable",
        "new input fixed",
        "new input fixed with correct main rewiring",
    ],
):
    """Generate plots comparing losses from multiple experiments.
    Args:
    - experiments: list of lists of strings indicating experiment
        directories. All experiments must at least have `num_epochs`
        and have the same number of epochs per file. Can also just be a
        list of strings.
    Options for loss of interest:
    - 'step_loss'
    - 'epoch_loss'
    """
    if type(experiments[0]) == list:
        # When passed a list of lists (i.e. when we're comparing the
        # losses of multiple experiments), flatten it
        experiments = [item for sublist in experiments for item in sublist]

    # Epochs are saved in groups of 10,
    data_files = filenames(num_epochs, epochs_per_file)

    # Plot losses for all experiments specified
    plt.figure()
    for xdir in experiments:
        # Get all of a single experiment's losses
        losses = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            losses += data[loss_of_interest].tolist()

        # Plot losses for a single experiment
        plt.plot(losses[0 : num_epochs - epochs_per_file])

    # Label everything
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    # Create and save the final figure
    plt.draw()
    plt.savefig(savepath)
