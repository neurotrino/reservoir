"""Plot loss data from network output files."""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append("../")
sys.path.append("../../")

# internal ----
from utils.misc import filenames
from utils.misc import get_experiments

data_dir = "/data/experiments/"
num_epochs = 1000
epochs_per_file = 10

onlinerate_experiments = [
    "fwd-pipeline-batchsize30-definedout-fixedsparserewire",
    "fwd-pipeline-batchsize30-definedout-fixedsparserewire-onlinerate0.5",
    "fwd-pipeline-batchsize30-definedout-fixedsparserewire-onlinerate0.1-endplot",
]

onlinerate_legend = [
    "realistic output with rewiring and global rate loss (1x)",
    "above with online rate loss 0.5x",
    "online rate loss 0.1x",
]

spec_output_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50",
    "fwd-pipeline-batchsize30-definedout",
    "fwd-pipeline-batchsize30-definedout-fixedsparserewire",
]

fwd_experiments = [
    "fwd-input-img-15x-trainable",
    "fwd-input-img-15x-fixed",
    "fwd-main-rewire-img-15x-fixed",
]
regeninspikes_experiments = [
    "fwd-main-rewire-lowlroutput",
    "fwd-main-rewire-lowlroutput-0.0001-newallen-l23",
    "fwd-pipeline-inputspikeregen",
    "fwd-pipeline-inputspikeregen-newallen-l23",
]
runlonger_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-2",
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower",
]
regen_lr_experiments = [
    "fwd-pipeline-inputspikeregen-newallen-l23",
    "fwd-pipeline-inputspikeregen-newl23-lowerlr",
    "fwd-pipeline-inputspikeregen-newl23-evenlowerlr",
]
rewire_optimizer_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower",
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower-norewire",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-norewire",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-SGD",
]
all_combined_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-norewire-vdist",
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-noisew-vdist",
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
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50",
]
rateloss_legend = [
    "rate cost 10",
    "rate cost 1",
    "rate cost 1, lax rate loss inclusion",
    "rate cost 1, refractory stop gradient",
    "the above with batch size 50",
]
synchloss_experiments = [
    "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50",
    "fwd-pipeline-batchsize30-synchcost0.01",
    "fwd-pipeline-batchsize30-synchcost0.1",
    "fwd-pipeline-batchsize30-laxsynchrate0.25",
]
synchloss_legend = [
    "larger batch size, rate loss cost 1",
    "the above with synch loss cost 0.01",
    "the above with synch loss cost 0.1",
    "the above with both lax rate and synch loss",
]
spec_output_legend = [
    "dense, unspecified output (previous)",
    "specified lognormal sparse output init",
    "specified output with enforced sparsity and rewiring",
]

experiments = [
    "ccd_200_lif_sparse",
    "ccd_200_lif_rewiring",
    "ccd_500_lif_sparse",
    "ccd_500_lif_rewiring",
]

savepath = "/data/results/fwd/onlinerate_task.png"

# remove loss_of_interest from arg

"""
def compare_losses_within_experiment_set(
    savepath = '/data/results/experiment1/set_loss.png',
    data_dir = data_dir,
    experiment_string = 'run-batch30-specout-onlinerate0.1-singlepreweight',
    num_epochs = num_epochs,
    epochs_per_file = epochs_per_file,
    title = 'Experiment set 1',
    xlabel = 'batches',
):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    #create 2x2 subplot figure
    #subplot 0: total loss of all experiments individually
    #subplot 1: mean total loss with standard deviation across experiments
    #subplot 2: task loss of all experiments individually
    #subplot 3: rate loss of all experiments individually

    loss_arr = []
    for xdir in experiments:
        losses = []
        task_losses = []
        rate_losses = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            losses += np.add(data['step_task_loss'],data['step_rate_loss']).tolist()
            task_losses += data['step_task_loss'].tolist()
            rate_losses += data['step_rate_loss'].tolist()
        # Append to ongoing array of shape experiment x time
        loss_arr = np.vstack([loss_arr, losses])
        ax[0].plot(losses)
        ax[2].plot(task_losses)
        ax[3].plot(rate_losses)
    ax[0].set_ylabel('total loss')
    ax[1].set_ylabel('mean total loss')
    ax[2].set_ylabel('task loss')
    ax[3].set_ylabel('rate_loss')
    for i in ax:
        ax[i].set_xlabel(xlabel)

    # Create mean total loss plot with standard deviation
    loss_std = np.std(loss_arr, axis=0)
    loss_mean = np.mean(loss_arr, axis=0)
    ax[1].plot(loss_mean)
    ax[1].fill_between(loss_mean-loss_std, loss_mean+loss_std, alpha=0.5)

    # Create and save the final figure
    fig.suptitle(title)
    plt.draw()
    plt.savefig(savepath,dpi=300)
    plt.clf()
    plt.close()"""


def compare_losses(
    savepath=savepath,
    data_dir=data_dir,
    experiments=onlinerate_experiments,
    num_epochs=num_epochs,
    epochs_per_file=epochs_per_file,
    title="Specified output layer",
    xlabel="batches",
    ylabel="total loss",
    legend=onlinerate_legend,
):
    """Generate plots comparing losses from multiple experiments.
    Args:
    - experiments: list of lists of strings indicating experiment
        directories. All experiments must at least have `num_epochs`
        and have the same number of epochs per file. Can also just be a
        list of strings.
    Options for loss of interest:
    - `step_loss`
    - `epoch_loss`
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

            # if xdir != "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger-vdist-rateloss1-refracstopgrad-batchsize50":
            # arr = np.array([data['step_task_loss'],data['step_rate_loss'],data['step_synch_loss']])
            # loss_of_interest = arr.sum(axis=0)
            # losses += loss_of_interest.tolist()
            # else:
            # loss_of_interest = np.add(data['step_task_loss'],data['step_rate_loss'])
            loss_of_interest = data["step_task_loss"]
            losses += loss_of_interest.tolist()
        # Plot losses for a single experiment
        plt.plot(losses)

    # Label everything
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    # Create and save the final figure
    plt.draw()
    plt.savefig(savepath)


def plot_single_experiment_loss(plot_dir, include_rate_loss):
    rate_losses = []
    task_losses = []
    total_losses = []
    num_epochs = 1000
    epochs_per_file = 10
    data_files = filenames(num_epochs, epochs_per_file)
    plt.figure()
    for filename in data_files:
        filepath = os.path.join(plot_dir, "../npz-data", filename)
        data = np.load(filepath)
        task_loss = data["step_task_loss"]
        task_losses += task_loss.tolist()
        if include_rate_loss:
            rate_loss = data["step_rate_loss"]
            rate_losses += rate_loss.tolist()
            total_loss = task_loss + rate_loss
            total_losses += total_loss.tolist()
    plt.plot(task_losses)
    if include_rate_loss:
        plt.plot(rate_losses)
        plt.plot(total_losses)
        legend = ["task loss", "rate loss", "total loss"]
    else:
        legend = ["task loss"]
    plt.title("Loss over time")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend(legend)
    plt.draw()
    plt.savefig(os.path.join(plot_dir, "loss_over_time.png"))
