"""(simple) analyses and final plot generation for series of completed experiments"""

# ---- external imports -------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy
import sys
import seaborn as sns
import networkx as nx

from scipy.sparse import load_npz

# ---- internal imports -------------------------------------------------------
sys.path.append("../")
sys.path.append("../../")

from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments
from utils.extools.analyze_structure import get_degrees
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree

# ---- global variables -------------------------------------------------------
data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-savey'
task_experiment_string = 'run-batch30-onlytaskloss'
rate_experiment_string = 'run-batch30-onlyrateloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
i_end = 300

n_input = 16
seq_len = 4080

savepath = '/data/results/experiment1/'

e_only = True
positive_only = False
bin = 10

naive_batch_id = 0
trained_batch_id = 99
coh_lvl = 'coh0'
NUM_EXCI = 240

# Paul Tol's colorblind-friendly palette for scientific visualization
COLOR_PALETTE = [
    "#332288",  # indigo
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # cyan
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]

"""
# snippet of code to go thru a bunch of experiments that all contain a particular string
data_dirs = get_experiments(data_dir, experiment_string)
data_files = filenames(num_epochs, epochs_per_file)
fig, ax = plt.subplots(nrows=5, ncols=1)
# get your usual experiments first
for xdir in data_dirs:
"""

#ALL DUAL TRAINED TO BEGIN WITH:
spec_output_dirs = ["run-batch30-specout-onlinerate0.1-savey","run-batch30-dualloss-silence","run-batch30-dualloss-swaplabels"]
spec_input_dirs = ["run-batch30-dualloss-specinput0.3-rewire"]
spec_nointoout_dirs = ["run-batch30-dualloss-specinput0.2*noinoutrewire"]

def plot_all_weight_dists(): # just for dual-training for now
    # fall set (spec output)
    fig, ax = plt.subplots(nrows=3,ncols=2)
    for exp_string in spec_output_dirs:
        if not 'fall_data_dirs' in locals():
            fall_data_dirs = get_experiments(data_dir, exp_string)
        else:
        fall_data_dirs = np.hstack([fall_data_dirs,get_experiments(data_dir, exp_string)])
    # go through all dirs and grab the weight distributions of the first and last epochs
    data_files = filenames(num_epochs, epochs_per_file) # useful for plotting evolution over the entire course of training
    fall_in_naive = np.array([])
    fall_in_trained = np.array([])
    fall_rec_naive = np.array([])
    fall_rec_trained = np.array([])
    fall_out_naive = np.array([])
    fall_out_trained = np.array([])
    for xdir in fall_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        fall_in_naive.append(naive_data['tv0.postweights'][0])
        fall_in_trained.append(trained_data['tv0.postweights'][99])

        fall_rec_naive.append(naive_data['tv1.postweights'][0])
        fall_rec_trained.append(trained_data['tv1.postweights'][99])

        fall_out_naive.append(naive_data['tv2.postweights'][0])
        fall_out_trained.append(trained_data['tv2.postweights'][99])

    # plot e and i separately
    ax[0,0].hist(fall_in_naive[:,:,0:e_end],density=True)
    ax[0,0].hist(fall_in_naive[:,:,e_end:i_end],density=True)
    ax[0,0].legend(['e edges','i edges'])
    ax[0,0].set_title('naive input weights',fontname='Ubuntu')

    ax[0,1].hist(fall_in_trained[:,0:e_end,0:e_end],density=True)
    ax[0,1].hist(fall_in_trained[:,e_end:i_end,e_end:i_end],density=True)
    ax[0,1].legend(['e edges','i edges'])
    ax[0,1].set_title('trained input weights',fontname='Ubuntu')

    # plot layers separately
    ax[1,0].hist(fall_rec_naive[:,0:e_end,:],density=True)
    ax[1,0].hist(fall_rec_naive[:,e_end:i_end,:],density=True)
    ax[1,0].legend(['e edges','i edges'])
    ax[1,0].set_title('naive recurrent weights',fontname='Ubuntu')

    ax[1,1].hist(fall_rec_trained[:,0:e_end],density=True)
    ax[1,1].hist(fall_rec_trained[:,e_end:i_end],density=True)
    ax[1,1].legend(['e edges','i edges'])
    ax[1,1].set_title('trained recurrent weights',fontname='Ubuntu')

    ax[2,0].hist(fall_out_naive[:,0:e_end],density=True)
    ax[2,0].hist(fall_out_naive[:,e_end:i_end],density=True)
    ax[2,0].legend(['e edges','i edges'])
    ax[2,0].set_title('naive output weights',fontname='Ubuntu')

    ax[2,1].hist(fall_out_trained[:,0:e_end],density=True)
    ax[2,1].hist(fall_out_trained[:,e_end:i_end],density=True)
    ax[2,1].legend(['e edges','i edges'])
    ax[2,1].set_title('trained output weights',fontname='Ubuntu')

    plt.suptitle('Experiments with only specified output layer',fontname='Ubuntu')

    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.7)
    save_fname = savepath+'/set_plots/fall_weights_test.png'
    plt.savefig(save_fname,dpi=300)


    # eventually set the proper font for tick labels too
    """for tick in ax[0].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Ubuntu")"""

    # winter set (spec input)


    # spring set (spec no in to out lines)

def plot_input_channel_rates():
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]

    # for each of 600 trials
    for i in range(0,np.shape(y)[0]):
    # for each of 4080 time steps
    # determine if coherence 1 or 0
        coh0_idx = np.where(y[i]==0)[0]
        coh1_idx = np.where(y[i]==1)[0]
    # take average rates across that trial's timepoints for the same coherence level and append
        if len(coh0_idx)>0:
            if not 'coh0_channel_trial_rates' in locals():
                coh0_channel_trial_rates = np.average(x[i][coh0_idx],0)
            else:
                coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(x[i][coh0_idx],0)])

        if len(coh1_idx)>0:
            if not 'coh1_channel_trial_rates' in locals():
                coh1_channel_trial_rates = np.average(x[i][coh1_idx],0)
            else:
                coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(x[i][coh1_idx],0)])

    coh0_rates = np.average(coh0_channel_trial_rates,0)
    coh1_rates = np.average(coh1_channel_trial_rates,0)

    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].hist(coh0_channel_trial_rates,bins=20,histtype='step', density=True, stacked=True)
    ax[0].set_title('coherence 0', fontname="Ubuntu")
    ax[0].set_xlabel('spike rate (Hz)', fontname="Ubuntu")
    ax[0].set_ylabel('density', fontname="Ubuntu")
    ax[0].set_ylim([0,6])
    ax[1].hist(coh1_channel_trial_rates,bins=20,histtype='step', density=True, stacked=True)
    ax[1].set_title('coherence 1', fontname="Ubuntu")
    ax[1].set_xlabel('spike rate (Hz)', fontname="Ubuntu")
    ax[1].set_ylabel('density', fontname="Ubuntu")
    ax[1].set_ylim([0,6])
    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Ubuntu")
    #ax[1,0].hist(late_in[i,:],bins=50,histtype='step')
    #ax[1,1].hist(trained_in[i,:],bins=50,histtype='step')

    plt.suptitle("Spike rates of 16 input channels", fontname="Ubuntu")

    # Draw and save
    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    save_fname = savepath+'/set_plots/input_rates_final.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

    #return [coh0_rates,coh1_rates]
