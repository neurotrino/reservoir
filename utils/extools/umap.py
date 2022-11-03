"""UMAP Embedding Analysis"""

# Use scikit-learn virtualenv
# source sklearn-venv/bin/activate from ava /home/macleanlab/

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import umap

sys.path.append("../")
sys.path.append("../../")

# internal ----
from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments

#save_name='recruit_bin10_full'
set_save_name='set_plots'
#coh_lvl = 'coh0'
recruit_path = '/data/results/experiment1/recruitment_graphs_bin10_full/'
data_dir = "/data/experiments/"
experiment_string = "run-batch30-specout-onlinerate0.1-savey"
task_experiment_string = 'run-batch30-onlytaskloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
# note e_end may be different somehow depending on what dataset you're working on
i_end = 300
trial_len = 4080
savepath = "/data/results/experiment1/"

def get_data_for_umap(separate_by_type=False):
    if not separate_by_type:
        naive_data_arr = []
        trained_data_arr = []
        # get all experiments
        # get the activity for all runs that are of a single coherence level
        data_dirs = get_experiments(data_dir, experiment_string)
        for xdir in data_dirs:
            naive_data = np.load(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
            naive_spikes = naive_data['spikes'][0]
            naive_y = naive_data['true_y'][0]
            for i in range(np.shape(naive_y)[0]):
                if naive_y[i][0]==naive_y[i][trial_len-1]:
                    # this particular trial has no change in coherence
                    naive_data_arr.append(np.transpose(naive_spikes[i]))
                    # we do not care about which coherence level it is right now

            # repeat for trained data
            trained_data = np.load(os.path.join(data_dir, xdir, "npz-data/991-1000.npz"))
            trained_spikes = trained_data['spikes'][99]
            trained_y = trained_data['true_y'][99]
            for i in range(np.shape(trained_y)[0]):
                if trained_y[i][0]==trained_y[i][trial_len-1]:
                    trained_data_arr.append(np.transpose(trained_spikes[i]))

        return [naive_data_arr,trained_data_arr]
"""
def map_no_labels(save_name):
    # using numpy data appears sufficient
    # give in naive data spikes
    # give in trained data spikes
    # see if it does its own separation
    [naive_data, trained_data] = get_data_for_umap(separate_by_type=False)
    reducer = umap.UMAP()
    naive_embedding = reducer.fit_transform(naive_data)
    trained_embedding = reducer.fit_transform(trained_data)
"""
