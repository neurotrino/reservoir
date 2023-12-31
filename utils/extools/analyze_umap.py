"""UMAP Embedding Analysis

Intended for use within ava's scikit-learn virtual environment with
`source /home/macleanlab/sklearn-venv/bin/activate`.
"""

# external ----
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

sys.path.append("../")
sys.path.append("../../")

#import umap

# internal ----
from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments
#from analyze_final import plot_single_batch_delays


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
n_input = 16
savepath = "/data/results/experiment1/"


# ========== ========== ========== ========== ========== ========== ==========
# Data Processing
# ========== ========== ========== ========== ========== ========== ==========

def get_spike_data_for_umap(xdir, separate_by_type=False):
    """Get activity for all static-coherence trails from NumPy data.

    Arguments:
        xdir: experiment directory (relative filepath)

        separate_by_type: if True, separate into three categories:
        recurrent units that receive input, all recurrent units, recurrent units that project to output

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
            # Transpose so that the first dim is the unit id and the second is time

            # Track the coherence label for coherence-static trials
            if y[0] == 0:
                lbl_arr.append([tags[0]])
            else:
                lbl_arr.append([tags[1]])

        return dat_arr, lbl_arr


    # Load data from memory
    np_dir = os.path.join(DATA_DIR, xdir, "npz-data")

    naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
    trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

    naive_spikes_agg = []
    trained_spikes_agg = []
    naive_y_agg = []
    trained_y_agg = []
    for i in range(np.shape(naive_data['true_y'])[0]):
    #for i in range(3): # testing

        naive_y = naive_data["true_y"][i]
        trained_y = trained_data["true_y"][i]

        if separate_by_type:
            """
            # get only recurrent units' spikes that receive input
            in_id = np.argwhere(naive_data['tv0.postweights'][i][:]!=0)
            naive_spikes = naive_data['spikes'][i][in_id]
            in_id = np.argwhere(trained_data['tv0.postweights'][i]!=0)
            trained_spikes = trained_data['spikes'][i][in_id]
            """
            # get only recurrent units' spikes that project to output
            # ultimately separate into e and i as well
            in_id = np.argwhere(naive_data['tv2.postweights'][i]!=0)
            in_id = in_id[:,0]
            naive_spikes = []
            for j in range(np.shape(naive_data['true_y'][i])[0]):
                naive_spikes.append(np.transpose(np.transpose(naive_data['spikes'][i][j])[in_id,:]))

            in_id = np.argwhere(trained_data['tv2.postweights'][i]!=0)
            in_id = in_id[:,0]
            trained_spikes = []
            for j in range(np.shape(trained_data['true_y'][i])[0]):
                trained_spikes.append(np.transpose(np.transpose(trained_data['spikes'][i][j])[in_id,:]))

        else:
            # get all recurrent units' spikes
            naive_spikes = naive_data["spikes"][i]
            trained_spikes = trained_data["spikes"][i]

        # Reformat data for umap analysis
        naive_data_arr, naive_y_arr = np_to_umap_data(
            naive_y, naive_spikes, [0, 1]
        )
        trained_data_arr, trained_y_arr = np_to_umap_data(
            trained_y, trained_spikes, [2, 3]
        )

        if i==0:
            naive_spikes_agg = naive_data_arr
            trained_spikes_agg = trained_data_arr
            naive_y_agg = naive_y_arr
            trained_y_agg = trained_y_arr
        else:
            naive_spikes_agg = np.vstack([naive_spikes_agg,naive_data_arr])
            trained_spikes_agg = np.vstack([trained_spikes_agg,trained_data_arr])
            naive_y_agg = np.vstack([naive_y_agg,naive_y_arr])
            trained_y_agg = np.vstack([trained_y_agg,trained_y_arr])

    return [naive_spikes_agg, trained_spikes_agg, naive_y_agg.flatten(), trained_y_agg.flatten()]


# ========== ========== ========== ========== ========== ========== ==========
# Plotting
# ========== ========== ========== ========== ========== ========== ==========

def map_rns(rn_dir='/data/results/experiment1/spring_fns/21.06.01/trained/',n_neighbors=5):

    # find delays
    exp_str = rn_dir.split("/")[5]
    for exp_file in os.listdir('/data/experiments/'):
        if exp_str in exp_file:
            exp_fullpath = '/data/experiments/'+exp_file+'/npz-data/991-1000.npz'
    [starts,delay_durs] = plot_single_batch_delays(exp_fullpath,rn_dir)

    # loop through rns for all trials
    rn_files = os.listdir(rn_dir)

    if not os.path.isdir(os.path.join(rn_dir, 'umap')):
        os.makedirs(os.path.join(rn_dir, 'umap'))

    for fname in rn_files:
        if 'rns.npz' in fname:

            data = np.load(rn_dir+fname,allow_pickle=True)
            rns = data['rns']
            # plot separately for all 4 types of connections
            rn_ee = rns[:,:241,:241]
            timesteps = np.shape(rn_ee)[0]
            # flatten dimension of weights
            rn_ee = np.reshape(rn_ee,[np.shape(rn_ee)[0],np.shape(rn_ee)[1]*np.shape(rn_ee)[2]])
            rn_ei = rns[:,:241,241:]
            rn_ei = np.reshape(rn_ei,[np.shape(rn_ei)[0],np.shape(rn_ei)[1]*np.shape(rn_ei)[2]])
            rn_ie = rns[:,241:,:241]
            rn_ie = np.reshape(rn_ie,[np.shape(rn_ie)[0],np.shape(rn_ie)[1]*np.shape(rn_ie)[2]])
            rn_ii = rns[:,241:,241:]
            rn_ii = np.reshape(rn_ii,[np.shape(rn_ii)[0],np.shape(rn_ii)[1]*np.shape(rn_ii)[2]])

            reducer = umap.UMAP(n_neighbors)
            ee = reducer.fit_transform(rn_ee)
            ei = reducer.fit_transform(rn_ei)
            ie = reducer.fit_transform(rn_ie)
            ii = reducer.fit_transform(rn_ii)

            # color according to discrete pre/post/delay periods
            #trial_idx = int(fname.split("_")[1])
            #c_segmented = np.hstack([np.zeros([250]),np.ones([delay_durs[trial_idx]])/2,np.ones([250])])

            fig, ax = plt.subplots(nrows=2,ncols=2)

            # plot umap
            ax[0,0].scatter(ee[:,0],ee[:,1],c=np.arange(0,timesteps),cmap='winter')
            #ax[0,0].colorbar()
            ax[0,0].set_title('e->e',fontname='Ubuntu')

            ax[0,1].scatter(ei[:,0],ei[:,1],c=np.arange(0,timesteps),cmap='winter')
            #ax[0,1].colorbar()
            ax[0,1].set_title('e->i',fontname='Ubuntu')

            ax[1,0].scatter(ie[:,0],ie[:,1],c=np.arange(0,timesteps),cmap='winter')
            #ax[1,0].colorbar()
            ax[1,0].set_title('i->e',fontname='Ubuntu')

            ax[1,1].scatter(ii[:,0],ii[:,1],c=np.arange(0,timesteps),cmap='winter')
            #ax[1,1].colorbar()
            ax[1,1].set_title('i->i',fontname='Ubuntu')

            # would be useful to plot in different colors according to time. can you?
            # figure out exactly what umap does to begin with

            plt.suptitle('UMAP projections of trained recruit nets surrounding change',fontname='Ubuntu')
            ax = ax.flatten()
            for i in range(0,len(ax)):
                for tick in ax[i].get_xticklabels():
                    tick.set_fontname('Ubuntu')
                for tick in ax[i].get_yticklabels():
                    tick.set_fontname('Ubuntu')

            # draw and save plot
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            plt.draw()
            save_fname = rn_dir+'umap/'+fname[:-5]+'_umap_'+str(n_neighbors)+'.png'
            plt.savefig(save_fname,dpi=300)

            # Teardown
            plt.clf()
            plt.close()

def map_activity():
    """TODO: document function"""

    # NOTE: YQ - using numpy data appears sufficient;
    #       give in naive data spikes;
    #       give in trained data spikes;
    #       see if it does its own separation;

    all_data_arr = []
    all_y_arr = []

    data_dirs = get_experiments(DATA_DIR, experiment_string)

    for xdir in data_dirs:
        [naive_spikes, trained_spikes, naive_y, trained_y] = get_spike_data_for_umap(
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

    reducer = umap.UMAP(n_neighbors=5)
    embedding = reducer.fit_transform(all_data_arr)

    # Create and save plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=all_y_arr, cmap='Spectral')
    plt.colorbar()
    plt.title('UMAP projection of naive & trained coherence-level responses')
    exp_string = xdir[-9:-1]  # NOTE: for use if/when creating and saving each experiment's embedding separately
    save_fname = savepath+set_save_name+'/umap_fullepoch_5.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

"""
def map_input():
    # only the units that receive input

    all_data_arr = []
    all_y_arr = []

    data_dirs = get_experiments(DATA_DIR, experiment_string)

    for xdir in data_dirs:
        [naive_spikes, trained_spikes, naive_y, trained_y] = get_spike_data_for_umap(
            xdir, separate_by_type=True
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

    reducer = umap.UMAP(n_neighbors=5)
    embedding = reducer.fit_transform(all_data_arr)

    # Create and save plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=all_y_arr, cmap='Spectral')
    plt.colorbar()
    plt.title('UMAP projection of activity of units that receive input')
    exp_string = xdir[-9:-1]  # NOTE: for use if/when creating and saving each experiment's embedding separately
    save_fname = savepath+set_save_name+'/umap_fullepoch_input_5.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()
    """

def map_output():
    # only the units that project to output
    all_data_arr = []
    all_y_arr = []

    data_dirs = get_experiments(DATA_DIR, experiment_string)

    for xdir in data_dirs:
        [naive_spikes, trained_spikes, naive_y, trained_y] = get_spike_data_for_umap(
            xdir, separate_by_type=True
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

    reducer = umap.UMAP(n_neighbors=5)
    embedding = reducer.fit_transform(all_data_arr)

    # Create and save plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=all_y_arr, cmap='Spectral')
    plt.colorbar()
    plt.title('UMAP projection of activity of units that project to output')
    exp_string = xdir[-9:-1]  # NOTE: for use if/when creating and saving each experiment's embedding separately
    save_fname = savepath+set_save_name+'/umap_fullepoch_output_5.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()


def map_synaptic():
    # UMAP projections of the naive and trained input synaptic graphs

    all_w_arr = []
    all_labels = []

    data_dirs = get_experiments(DATA_DIR, experiment_string)

    for xdir in data_dirs:
        # load in synaptic data
        np_dir = os.path.join(DATA_DIR, xdir, "npz-data")

        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        for i in range(np.shape(naive_data['true_y'])[0]):

            all_w_arr.append(np.reshape(naive_data['tv0.postweights'][i],[n_input*i_end]))
            all_w_arr.append(np.reshape(trained_data['tv0.postweights'][i],[n_input*i_end]))
            all_labels.append([0]) # naive trial
            all_labels.append([1]) # trained trial

    all_labels = np.reshape(all_labels,[len(all_labels)])

    reducer = umap.UMAP(n_neighbors=15)
    embedding = reducer.fit_transform(all_w_arr)

    # Create and save plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='Spectral')
    plt.colorbar()
    plt.title('UMAP projection of naive & trained input weights')
    exp_string = xdir[-9:-1]  # NOTE: for use if/when creating and saving each experiment's embedding separately
    save_fname = savepath+set_save_name+'/umap_input_fullepoch_15.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()
