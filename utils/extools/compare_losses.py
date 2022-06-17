# plot the loss over time for networks of different sizes and components

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

data_dir = '/data/experiments/'
num_epochs = 1000
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

experiments = ['ccd_200_lif_sparse','ccd_200_lif_rewiring','ccd_500_lif_sparse','ccd_500_lif_rewiring']

savepath = '/data/results/fwd/regenlr-longer-norewire-sgd.png'

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

def compare_losses(
    savepath=savepath,
    data_dir=data_dir,
    experiments=runlonger_experiments,
    num_epochs=num_epochs,
    epochs_per_file=epochs_per_file,
    loss_of_interest="epoch_loss",
    title="Regenerated input spikes, new Allen connectivity, 1000 epochs",
    xlabel="epochs",
    ylabel="total loss",
    legend=[
        "main lr 0.005, output lr 0.00001",
        "the above, no rewiring",
        "main lr 0.001, output lr 0.00001",
        "the above, no rewiring",
        "the above w stochastic grad descent instead of Adam",
    ]
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
        # losses of multiple experiments), we need to first flatten the
        # list
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

"""
def compare_losses(experiments):
    epoch_groups = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180','181-190','191-200']
    nfiles = len(epoch_groups)
    plt.figure()
    for i in range(len(experiments)):
        loss = []
        for j in range(nfiles):
            fname = data_dir + experiments[i] + '/npz-data/' + epoch_groups[j] + '.npz'
            data = np.load(fname)
            for k in range(len(data['epoch_loss'])):
                loss.append(data['epoch_loss'][k])
        plt.plot(loss[0:190], label=experiments[i])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('LIF network on CCD task')
    plt.draw()
    plt.savefig('/data/results/ccd/compare_lif_network_size.png')
"""
