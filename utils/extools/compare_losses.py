"""Plot loss data from network output files."""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
sys.path.append('../../')

# internal ----
from utils.misc import filenames

data_dir = "/data/experiments/"
num_epochs = 660
epochs_per_file = 10

fwd_experiments = [
    "fwd-input-img-15x-trainable",
    "fwd-input-img-15x-fixed",
    "fwd-main-rewire-img-15x-fixed"
]
regeninspikes_experiments = [
    "fwd-main-rewire-lowlroutput",
    "fwd-main-rewire-lowlroutput-0.0001-newallen-l23",
    "fwd-pipeline-inputspikeregen",
    "fwd-pipeline-inputspikeregen-newallen-l23",
]
runlonger_experiments=[
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-2",
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower",
]
regen_lr_experiments = [
    "fwd-pipeline-inputspikeregen-newallen-l23",
    "fwd-pipeline-inputspikeregen-newl23-lowerlr",
    "fwd-pipeline-inputspikeregen-newl23-evenlowerlr"
]
rewire_optimizer_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower",
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower-norewire",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-norewire",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-SGD"
]
all_combined_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-norewire-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-noisew-vdist"
]
all_less_some_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-novdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-novdist-tasklossonly",
]
rateloss_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-laxrateloss1",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50"
]
rateloss_legend = [
    "rate cost 10",
    "rate cost 1",
    "rate cost 1, lax rate loss inclusion",
    "rate cost 1, refractory stop gradient",
    "the above with batch size 50"
]
synchloss_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50",
    "fwd-pipeline-batchsize30-synchcost0.01",
    "fwd-pipeline-batchsize30-synchcost0.1",
    "fwd-pipeline-batchsize30-laxsynchrate0.25"
]
synchloss_legend = [
    "larger batch size, rate loss cost 1",
    "the above with synch loss cost 0.01",
    "the above with synch loss cost 0.1",
    "the above with both lax rate and synch loss"
]

experiments = ['ccd_200_lif_sparse','ccd_200_lif_rewiring','ccd_500_lif_sparse','ccd_500_lif_rewiring']

savepath = '/data/results/fwd/synchloss.png'

# remove loss_of_interest from arg

def compare_losses(
    savepath=savepath,
    data_dir=data_dir,
    experiments=synchloss_experiments,
    num_epochs=num_epochs,
    epochs_per_file=epochs_per_file,
    title="Initial V dist, main lr 0.001, output lr 0.00001, refractory stop grad",
    xlabel="batches",
    ylabel="total loss",
    legend=synchloss_legend
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
            if filename != "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50":
                arr = np.array(data['step_task_loss'],data['step_rate_loss'],data['step_synch_loss'])
                loss_of_interest = arr.sum(axis=0)
                losses += loss_of_interest.tolist()
            else:
                arr = np.array(data['step_task_loss'],data['step_rate_loss'])
                loss_of_interest = arr.sum(axis=0)
                losses += loss_of_interest.tolist()
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
