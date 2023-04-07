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
spec_output = ["*run-batch30-specout-onlinerate0.1-savey*","*run-batch30-dualloss-silence*","*run-batch30-dualloss-swaplabels*"]
spec_input = ["*run-batch30-dualloss-specinput0.3-rewire*"]
spec_nointoout = ["*run-batch30-dualloss-specinput0.2*noinoutrewire*"]

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
    # take average rates and append
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
    ax[0].hist(coh0_channel_trial_rates,histtype='step')
    ax[0].set_title('responses to coherence 0')
    ax[0].set_xlabel('spike rates')
    ax[1].hist(coh1_channel_trial_rates,bins=50,histtype='step')
    ax[1].set_title('responses to coherence 1')
    ax[1].set_xlabel('spike rates')
    #ax[1,0].hist(late_in[i,:],bins=50,histtype='step')
    #ax[1,1].hist(trained_in[i,:],bins=50,histtype='step')

    plt.suptitle("Input channels' rates")

    # Draw and save
    plt.draw()
    save_fname = savepath+'/specinput0.2/input_rates.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

    return [coh0_rates,coh1_rates]
