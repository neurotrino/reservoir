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

def get_data_for_umap(xdir, separate_by_type=False):
    if not separate_by_type:
        naive_data_arr = []
        naive_y_arr = []
        trained_data_arr = []
        trained_y_arr = []
        # get the activity for all runs that are of a uniform coherence level
        naive_data = np.load(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        naive_spikes = naive_data['spikes'][0]
        naive_y = naive_data['true_y'][0]
        for i in range(np.shape(naive_y)[0]):
            if naive_y[i][0]==naive_y[i][trial_len-1]:
                # this particular trial has no change in coherence
                naive_data_arr.append(np.transpose(naive_spikes[i]))
                # specify the coherence level of this whole trial
                if naive_y[i][0]==0:
                    naive_y_arr.append([0])
                else:
                    naive_y_arr.append([1])

        # repeat for trained data
        trained_data = np.load(os.path.join(data_dir, xdir, "npz-data/991-1000.npz"))
        trained_spikes = trained_data['spikes'][99]
        trained_y = trained_data['true_y'][99]
        for i in range(np.shape(trained_y)[0]):
            if trained_y[i][0]==trained_y[i][trial_len-1]:
                trained_data_arr.append(np.transpose(trained_spikes[i]))
                if trained_y[i][0]==0:
                    trained_y_arr.append([2])
                else:
                    trained_y_arr.append([3])

        return [naive_data_arr,trained_data_arr,naive_y_arr,trained_y_arr]


def map_no_labels(save_name):
    # using numpy data appears sufficient
    # give in naive data spikes
    # give in trained data spikes
    # see if it does its own separation
    data_dirs = get_experiments(data_dir, experiment_string)
    # do for each experiment separately for now
    for xdir in data_dirs:
        [naive_spikes, trained_spikes, naive_y, trained_y] = get_data_for_umap(xdir, separate_by_type=False)
        all_data = np.concatenate((naive_spikes, trained_spikes), axis=0)
        # flatten units and time, so we have just trial as the first dim
        all_data=all_data.reshape(np.shape(all_data)[0],np.shape(all_data)[1]*np.shape(all_data)[2])
        all_labels = np.ndarray.flatten(np.concatenate((naive_y, trained_y), axis=0))
        
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(all_data)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='Spectral')
        plt.title('UMAP projection of naive & trained coherence-level responses')
        exp_string = xdir[-9:-1]
        save_fname = savepath+set_save_name+'/'+exp_string+'_umap.png'
        plt.savefig(save_fname,dpi=300)
        plt.clf()
        plt.close()
