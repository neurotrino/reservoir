"""UMAP Embedding Analysis

Intended for use within ava's scikit-learn virtual environment with
`source /home/macleanlab/sklearn-venv/bin/activate`.
"""

# external ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

sys.path.append("../")
sys.path.append("../../")

import umap

# internal ----
from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments


# ========== ========== ========== ========== ========== ========== ==========
# Global Variables
# ========== ========== ========== ========== ========== ========== ==========

DATA_DIR = "/data/experiments/"
TRIAL_LEN = 4080

#save_name='recruit_bin10_full'
set_save_name='set_plots'
#coh_lvl = 'coh0'
recruit_path = '/data/results/experiment1/recruitment_graphs_bin10_full/'
experiment_string = "run-batch30-specout-onlinerate0.1-savey"
task_experiment_string = 'run-batch30-onlytaskloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
# note e_end may be different somehow depending on what dataset you're working on
i_end = 300
savepath = "/data/results/experiment1/"


# ========== ========== ========== ========== ========== ========== ==========
# Data Processing
# ========== ========== ========== ========== ========== ========== ==========

def get_data_for_umap(xdir, separate_by_type=False):
    """Get activity for all static-coherence trails from NumPy data.

    Arguments:
        xdir: experiment directory (relative filepath)

        separate_by_type: TODO: document NOTE: flag is not currently
            supported

    Returns:
        naive_data_arr: list of spiking activities for the preserved
            trials (trials with no coherence change) from the naive
            version of the network

        trained_data_arr: list of spiking activities for the preserved
            trials (trials with no coherence change) from the trained
            version of the network

        naive_y_arr: list of coherence levels for the preserved trials
            from the naive version of the network

        trained_y_arr: list of coherence levels for the preserved
            trials from the trained version of the network
    """

    def np_to_umap_data(ys, zs, tags=[0, 1]):
        """
        Filter out coherence-change trials' data and format the
        remaining static-coherence trials' data.

        Arguments:
            ys: coherence labels
            zs: spiking activity
            tags: the values corresponding to each coherence level in
                the reformatted data

        Returns:
            dat_arr: list of spiking activities for the preserved
                trials (trials with no coherence change)

            lbl_arr: list of coherence levels for the preserved trials,
                using the labels specified in `tags`
        """

        dat_arr = []  # return variable: coh-static trials' spike data
        lbl_arr = []  # return variable: coh-static trails' coh labels

        for (y, z) in zip(ys, zs):

            # Ignore trials with changes in coherence
            if y[0] != y[TRIAL_LEN - 1]:
                continue

            # Track the spikes of trials with a consistent coherence
            dat_arr.append(z.transpose())

            # Track the coherence label for coherence-static trials
            if y[0] == 0:
                lbl_arr.append([tags[0]])
            else:
                lbl_arr.append([tags[1]])

        return dat_arr, lbl_arr


    # TODO: add support for this flag
    if separate_by_type:
        raise NotImplementedError(
            "the `separate_by_type` flag is not currently supported"
        )

    # Load data from memory
    np_dir = os.path.join(DATA_DIR, xdir, "npz-data")

    naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
    trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

    naive_y = naive_data["true_y"][0]
    trained_y = trained_data["true_y"][99]

    naive_spikes = naive_data["spikes"][0]
    trained_spikes = trained_data["spikes"][99]

    # Reformat data for umap analysis
    naive_data_arr, naive_y_arr = np_to_umap_data(
        naive_y, naive_spikes, [0, 1]
    )
    trained_data_arr, trained_y_arr = np_to_umap_data(
        trained_y, trained_spikes, [2, 3]
    )
    return [naive_data_arr, trained_data_arr, naive_y_arr, trained_y_arr]


# ========== ========== ========== ========== ========== ========== ==========
# Plotting
# ========== ========== ========== ========== ========== ========== ==========

def map_no_labels():
    """TODO: document function"""

    # NOTE: YQ - using numpy data appears sufficient;
    #       give in naive data spikes;
    #       give in trained data spikes;
    #       see if it does its own separation;

    all_data_arr = []
    all_y_arr = []

    data_dirs = get_experiments(DATA_DIR, experiment_string)

    for xdir in data_dirs:
        [naive_spikes, trained_spikes, naive_y, trained_y] = get_data_for_umap(
            xdir, separate_by_type=False
        )
        all_data = np.concatenate((naive_spikes, trained_spikes), axis=0)

        # flatten units and time, so we have just trial as the first dim
        all_data=all_data.reshape(np.shape(all_data)[0],np.shape(all_data)[1]*np.shape(all_data)[2])

        if all_data_arr==[]:
            all_data_arr = all_data
        else:
            all_data_arr = np.vstack([all_data_arr, all_data]) # aggregate spike data with trial as the first dim

        all_labels = np.ndarray.flatten(np.concatenate((naive_y, trained_y), axis=0))
        all_y_arr.append(all_labels)

    # turn list of arrays into one
    all_y_arr = np.concatenate(all_y_arr,axis=0)

    reducer = umap.UMAP(n_neighbors=2)
    embedding = reducer.fit_transform(all_data_arr)

    # Create and save plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=all_y_arr, cmap='Spectral')
    plt.colorbar()
    plt.title('UMAP projection of naive & trained coherence-level responses')
    exp_string = xdir[-9:-1]  # NOTE: for use if/when creating and saving each experiment's embedding separately
    save_fname = savepath+set_save_name+'/umap_2.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()
