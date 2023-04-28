"""final plot generation for series of completed experiments; synthesis of things in analyze_final"""

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

from utils.misc import *
from utils.extools.analyze_structure import get_degrees
from utils.extools.fn_analysis import *
from utils.extools.MI import *
from utils.extools.analyze_structure import _nets_from_weights
from utils.extools.analyze_dynamics import *
from utils.extools.analyze_final import *

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

# Paul Tol's colorblind-friendly palettes for scientific visualization
vibrant = [
    '#0077BB',#blue
    '#33BBEE',#cyan
    '#009988',#teal
    '#EE7733',#orange
    '#CC3311',#red
    '#EE3377',#magenta
    '#BBBBBB'#grey
]

muted = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    '#999933',  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]

light = [
    '#77AADD',#light blue
    '#99DDFF',#light cyan
    '#44BB99',#mint
    '#BBCC33',#pear
    '#AAAA00',#olive
    '#EEDD88',#light yellow
    '#EE8866',#orange
    '#FFAABB',#pink
    '#DDDDDD',# pale grey
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
spec_nointoout_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
save_inz_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
spec_nointoout_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
spec_nointoout_dirs_task = ["run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
all_spring_dual_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]


# this is all a general sort of thing, once you do one (mostly figure out shading and dist comparisons) it'll come easily

"""
def dists_of_all_weights(dual_exp_dir=spec_nointoout_dirs,task_exp_dir=spec_nointoout_dirs_task,rate_exp_dir=spec_nointoout_dirs_rate,exp_season='spring'):
    # aggregate over all experiments of this type
    # plot distributions in naive state
    # plot distributions in trained state
    # input layer to e and i; rec ee ei ie ii,
    # (((later create comparable plot according to tuning)))
    # do for rate trained as well
    # do for task trained as well
    # make everything on comparable axes
    # look for the 5x input multiplier; do something about that (plot separately / not at all?)
    # return means and stds (even if not proper) of the weight distributions, or save in some file

def dists_of_all_rates(exp_dir=spec_input_dirs,exp_season='winter'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

def dists_of_all_synch(exp_dir=spec_input_dirs,exp_season='winter'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

def all_losses_over_training(exp_dir=spec_nointoout_dirs,exp_season='spring'):
    # plot mean with shading over time course of training for task and rate loss
    # have a better way to describe this eventually
    # split into change and no-change trials to describe performance? that might be worthwhile


# SEE IF YOU CAN COMPLETE ALL THE BELOW TODAY
# ONE PER HOUR, SUPER DOABLE

def dists_of_input_rates(exp_dirs=[save_inz_dirs,save_inz_dirs_rate],exp_season='spring'):
    # also bring in the rate trained ones too. just anything that contains saveinz; also the original CNN outputs too
    # plot the distribution of rates of each channel in response to the two coherence levels
    # do so for all 16 channels together (and measure, compare their distributions)
    # do so for each channel separately and describe just how close their firing rates are to each coherence level

    # from CNN output
    # determine which coherence level the input units prefer based on original CNN output file
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    CNN_x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    CNN_y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]

    exp_dirs = np.unique(exp_dirs.flatten())

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)

            data = np.load(filepath)

        #np_dir = os.path.join(data_dir,xdir,"npz-data")
        #naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
        #trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))

        # do for both naive and trained
            in_spikes = trained_data['inputs']
            true_y = trained_data['true_y']

            if '-swaplabels' in xdir: # not unswaplabels
                true_y = ~true_y.astype(int) + 2
"""

"""
def input_layer_over_training_by_coherence(dual_exp_dir=save_inz_dirs,rate_exp_dir=save_inz_dirs_rate,exp_season='spring'):
    # characterize the connectivity from the input layer to recurrent
    # plot over the course of training with shaded error bars
    # for not-save-inz experiments, get the information about input channels' coherence tunings from the original CNN output file
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
    # default based on CNN outputs originally, these are the input channels that have "tuning"
    default_coh1_idx = np.where(coh1_rates>coh0_rates)[0]
    default_coh0_idx = np.where(coh1_rates<coh0_rates)[0]

    # from actual experiment now
    for exp_string in dual_exp_dir:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        # determine experimentally the input channels that are most responsive to coh 0 or coh 1
        # or this can be part of a different prior function...

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)

            data = np.load(filepath)
            spikes = data['spikes']
            true_y = data['true_y']

            # aggregate the mean connectivity strength from the two tuned input populations to e and i units
            # maybe it's too much to do it over more than just the last batch trial for each file
            # that's still 100 datapoints
            for i in range(0,np.shape(true_y)[0]):
                for j in range(0,np.shape(true_y)[1]):

            # ratio of weights; get naive vs. trained distributions and also see how they evolve over training too
            coh0in_to_e/coh0in_to_i
            coh1in_to_e/coh1in_to_i
            # aggregate over all experiments

    # get a number distribution to quantify this, maybe for each experiment
    # the ratio between avg weight from input coh0 and coh1 to e and i recurrent units at the beginning and at the end of training
    # 0_to_e/0_to_i = 1 at beginning
    # 1_to_e/1_to_i = 1 at beginning
    # 0_to_e/0_to_i < 1 at end
    # 1_to_e/1_to_e > 1 at end
    # that's a good start

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)
"""


def characterize_tuned_rec_populations(exp_dirs=spec_nointoout_dirs_rate,exp_season='spring',mix_tuned_indices=False,plot_counts=True):
    # determine tuning of each recurrent unit across each of these experiments
    # according to trials of single coherence level only
    # include save inz as well into these spring experimental categories, okay
    # count up how many e and i units are tuned to each coherence level
    # quantify the extent they are tuned - again their relative rates to coh 0 and coh 1; plot together
    # look at naive state as well
    # [the above points to maybe additional analyses / quantifications to get later]
    # plot the rate distributions of these populations
    # compare between dual, task, rate training
    # as best you can

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    exp_data_dirs = np.unique(exp_data_dirs)

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    coh1_e_ct = []
    coh1_i_ct = []
    coh0_e_ct = []
    coh0_i_ct = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        """
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)"""

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))

        # do for both naive and trained (but trained first here)
        spikes = trained_data['spikes']
        true_y = trained_data['true_y']

        # find the relative tuning of all e and i units
        # find which units respond more to input of a certain coh level across batches and trials
        coh0_rec_rates = []
        coh1_rec_rates = []

        for i in range(0,np.shape(true_y)[0]): # batch
            for j in range(0,np.shape(true_y)[1]): # trial
                if true_y[i][j][0]==true_y[i][j][seq_len-1]: # no change trials only
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        ######{{{{   GET INDICES OF COH0 and COH1 TUNED UNITS   }}}}#######
        coh1_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        coh1_e_ct.append(len(coh1_rec_idx[coh1_rec_idx<e_end]))
        coh1_i_ct.append(len(coh1_rec_idx[coh1_rec_idx>=e_end]))

        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        coh0_e_ct.append(len(coh0_rec_idx[coh0_rec_idx<e_end]))
        coh0_i_ct.append(len(coh0_rec_idx[coh0_rec_idx>=e_end]))

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        ######{{{{   GET RATES OF TRAINED-TUNED-INDEXED E AND I UNITS IN TRAINED TRIALS   }}}}#######
        if not 'all_0e_to_0_rates' in locals():
            all_0e_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_0_rates = np.hstack([all_0e_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        if not 'all_0e_to_1_rates' in locals():
            all_0e_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_1_rates = np.hstack([all_0e_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_0_rates' in locals():
            all_1e_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_0_rates = np.hstack([all_1e_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_1_rates' in locals():
            all_1e_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_1_rates = np.hstack([all_1e_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        # now for the i units
        if not 'all_0i_to_0_rates' in locals():
            all_0i_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_0_rates = np.hstack([all_0i_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_0i_to_1_rates' in locals():
            all_0i_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_1_rates = np.hstack([all_0i_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_0_rates' in locals():
            all_1i_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_0_rates = np.hstack([all_1i_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_1_rates' in locals():
            all_1i_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_1_rates = np.hstack([all_1i_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        """
        # get all these e and i units' actual rates in response to trials of each coherence level
        if not 'all_coh0_e_rates' in locals():
            all_coh0_e_rates = coh0_rec_rates[:,:e_end].flatten()
        else:
            all_coh0_e_rates = np.hstack([all_coh0_e_rates,coh0_rec_rates[:,:e_end].flatten()])

        if not 'all_coh0_i_rates' in locals():
            all_coh0_i_rates = coh0_rec_rates[:,e_end:].flatten()
        else:
            all_coh0_i_rates = np.hstack([all_coh0_i_rates,coh0_rec_rates[:,e_end:].flatten()])

        if not 'all_coh1_e_rates' in locals():
            all_coh1_e_rates = coh1_rec_rates[:,:e_end].flatten()
        else:
            all_coh1_e_rates = np.hstack([all_coh1_e_rates,coh1_rec_rates[:,:e_end].flatten()])

        if not 'all_coh1_i_rates' in locals():
            all_coh1_i_rates = coh1_rec_rates[:,e_end:].flatten()
        else:
            all_coh1_i_rates = np.hstack([all_coh1_i_rates,coh1_rec_rates[:,e_end:].flatten()])"""

    trained_ct = [coh1_e_ct,coh1_i_ct,coh0_e_ct,coh0_i_ct]

    fig, ax = plt.subplots(nrows=2,ncols=2)
    ax = ax.flatten()

    """
    ax[0].hist(np.array(all_coh0_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh0_e_ct)))+' avg units)')
    ax[0].set_title('coh 0 tuned e units',fontname='Ubuntu')
    ax[1].hist(np.array(all_coh0_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh0_i_ct)))+' avg units)')
    ax[1].set_title('coh 0 tuned i units',fontname='Ubuntu')
    ax[2].hist(np.array(all_coh1_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh1_e_ct)))+' avg units)')
    ax[2].set_title('coh 1 tuned e units',fontname='Ubuntu')
    ax[3].hist(np.array(all_coh1_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh1_i_ct)))+' avg units)')
    ax[3].set_title('coh 1 tuned i units',fontname='Ubuntu')
    """

    ######{{{{   ADD THE TRAINED E AND I RATES TO SUBPLOTS   }}}}#######

    ######{{{{   SUBPLOT FOR E COHERENCE 0 TRIALS   }}}}#######
    ax[0].hist(all_0e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[0].hist(all_1e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR E COHERENCE 1 TRIALS   }}}}#######
    ax[1].hist(all_0e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[1].hist(all_1e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR I COHERENCE 0 TRIALS   }}}}#######
    ax[2].hist(all_0i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[2].hist(all_1i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR I COHERENCE 1 TRIALS   }}}}#######
    ax[3].hist(all_0i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[3].hist(all_1i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')


    ######{{{{   REPEAT FOR NAIVE   }}}}#######
    ######{{{{   EVENTUALLY FOR THE OTHER PLOT YOU ENVISIONED YOU WILL NEED TO RENAME AND SAVE THE BELOW FOR TRAINED   }}}}#######
    coh1_e_ct = []
    coh1_i_ct = []
    coh0_e_ct = []
    coh0_i_ct = []

    """
    del all_coh1_e_rates
    del all_coh1_i_rates
    del all_coh0_e_rates
    del all_coh0_i_rates
    """

    del all_0e_to_0_rates
    del all_0e_to_1_rates
    del all_1e_to_0_rates
    del all_1e_to_1_rates
    del all_0i_to_0_rates
    del all_0i_to_1_rates
    del all_1i_to_0_rates
    del all_1i_to_1_rates


    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        """
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)"""

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        naive_data = np.load(os.path.join(np_dir,"1-10.npz"))

        # do for both naive and trained (but trained first here)
        spikes = naive_data['spikes']
        true_y = naive_data['true_y']

        # find the relative tuning of all e and i units
        # find which units respond more to input of a certain coh level across batches and trials
        coh0_rec_rates = []
        coh1_rec_rates = []

        ######{{{{   GET UNITWISE RATES FOR NAIVE TRIALS OF COH1 AND COH0   }}}}#######
        for i in range(0,np.shape(true_y)[0]): # batch
            for j in range(0,np.shape(true_y)[1]): # trial
                if true_y[i][j][0]==true_y[i][j][seq_len-1]: # no change trials only
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                        # these are entirely rates in response to 0 coherence level
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        ######{{{{   FIND THE INDICES OF E AND I UNITS THAT ARE TUNED IN THE NAIVE STATE   }}}}#######
        # specify that these are the naive tuned indices; we do not care about them in the comparison plots for now

        coh1_naive_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        coh0_naive_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]

        coh0_e_ct.append(len(coh0_naive_rec_idx[coh0_naive_rec_idx<e_end]))
        coh0_i_ct.append(len(coh0_naive_rec_idx[coh0_naive_rec_idx>=e_end]))

        coh1_e_ct.append(len(coh1_naive_rec_idx[coh1_naive_rec_idx<e_end]))
        coh1_i_ct.append(len(coh1_naive_rec_idx[coh1_naive_rec_idx>=e_end]))

        ######{{{{   OPTION TO EITHER USE THE TRAINED TUNING INDICES TO INDEX THE NAIVE TRIALS FOR RATE CALCULATIONS OR THE NAIVE TUNING INDICES THEMSELVES   }}}}#######
        if mix_tuned_indices:
            coh0_rec_idx = coh0_naive_rec_idx
            coh1_rec_idx = coh1_naive_rec_idx

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        ######{{{{   GET RATES OF TRAINED-TUNED-INDEXED E AND I UNITS IN NAIVE TRIALS   }}}}#######
        # get all these e and i units' actual rates in response to trials of each coherence level
        # using indices of units that are tuned in their trained states
        # look at their original responses in the naive state
        if not 'all_0e_to_0_rates' in locals():
            all_0e_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_0_rates = np.hstack([all_0e_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        # responses of all coh0-tuned e units to coherence 1
        if not 'all_0e_to_1_rates' in locals():
            all_0e_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_1_rates = np.hstack([all_0e_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        # repeat for all_1e_to_0_rates, all_1e_to_1_rates, all_0i_to_0_rates, all_0i_to_1_rates, all_1i_to_0_rates, all_1i_to_1_rates

        if not 'all_1e_to_0_rates' in locals():
            all_1e_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_0_rates = np.hstack([all_1e_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_1_rates' in locals():
            all_1e_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_1_rates = np.hstack([all_1e_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        # now for the i units
        if not 'all_0i_to_0_rates' in locals():
            all_0i_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_0_rates = np.hstack([all_0i_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_0i_to_1_rates' in locals():
            all_0i_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_1_rates = np.hstack([all_0i_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_0_rates' in locals():
            all_1i_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_0_rates = np.hstack([all_1i_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_1_rates' in locals():
            all_1i_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_1_rates = np.hstack([all_1i_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])


    ######{{{{   ADD THE NAIVE E AND I RATES TO EXISTING SUBPLOTS   }}}}#######

    ######{{{{   SUBPLOT FOR E COHERENCE 0 TRIALS   }}}}#######
    ax[0].hist(all_0e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[0].hist(all_1e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[0].set_title('E rates to coherence 0 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR E COHERENCE 1 TRIALS   }}}}#######
    ax[1].hist(all_0e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[1].hist(all_1e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[1].set_title('E rates to coherence 1 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR I COHERENCE 0 TRIALS   }}}}#######
    ax[2].hist(all_0i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[2].hist(all_1i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[2].set_title('I rates to coherence 0 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR I COHERENCE 1 TRIALS   }}}}#######
    ax[3].hist(all_0i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[3].hist(all_1i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[3].set_title('I rates to coherence 1 trials',fontname='Ubuntu')

        ######{{{{   CHARACTERIZE WITH NUMBERS COMPARING DISTRIBUTIONS   }}}}#######
        # main comparison is: WITHIN TRIALS AND UNITS (E OR I) OF A CERTAIN TYPE
        # how different are NAIVE VS TRAINED
        # how different are COH1-TUNED RATES VS COH0-TUNED RATES

        # plot distributions together for: TRAINED responses of 0-selective and 1-selective e and i units to coherence 0
        # NAIVE
        # in trained state, e and i units should differ in their responses according to their selectivity (0 more than 1 in this case)
        # should use the same units for NAIVE instead of the other way around, but you do want to know how many units are selective vs not

        # follow identity then, units selective in the naive case - their responses in the trained state
        # units selective in the trained state - their responses in the naive state

        # but in the end, maybe we show that the differences aren't so large

        # TRAINED responses of 0-selective and 1-selective e and i units to coherence 1
        # NAIVE
        # in the trained state, e and i units should differ in their responses according to their tuning (1 more than 0 in this case)

        # should we define selectivity better in some way?
        # maybe you'll decide you want to plot different groups together

        # definitely want to characterize how similar and different are the responses of 0-selective vs 1-selective units to a given stimulus

        # separate plot and set of numbers showing just how many e and i units on average (dist) are tuned to 0 or 1 in the naive and the trained states

    """
        if not 'all_coh0_i_rates' in locals():
            all_coh0_i_rates = coh0_rec_rates[:,e_end:].flatten()
        else:
            all_coh0_i_rates = np.hstack([all_coh0_i_rates,coh0_rec_rates[:,e_end:].flatten()])

        if not 'all_coh1_e_rates' in locals():
            all_coh1_e_rates = coh1_rec_rates[:,:e_end].flatten()
        else:
            all_coh1_e_rates = np.hstack([all_coh1_e_rates,coh1_rec_rates[:,:e_end].flatten()])

        if not 'all_coh1_i_rates' in locals():
            all_coh1_i_rates = coh1_rec_rates[:,e_end:].flatten()
        else:
            all_coh1_i_rates = np.hstack([all_coh1_i_rates,coh1_rec_rates[:,e_end:].flatten()])
    """

    naive_ct = [coh1_e_ct,coh1_i_ct,coh0_e_ct,coh0_i_ct]
    # the natural question that arises is: is it the SAME units?
    # are the Most Important ones the same units?

    """
    ax[0].hist(np.array(all_coh0_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh0_e_ct)))+' avg units)')
    # ACTUALLY, we want to plot not all units' rates, but actually separate based on tuning
    # so instead of coh0_rec_rates[:,:e_end], we want coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]]
    # and coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]]
    # to compare the spike rates of coh0 and coh1 tuned recurrent e units in response to coh0 trials
    # within the same plot we also want to see naive vs. trained, or maybe side by side
    # we would expect the two naive distributions to pretty much overlap, even though there is some separability
    # there should be more separability in the trained case

    # and in another plot we want to see the spike rates of coh0 and coh1 tuned recurrent i units in response to coh0 trials
    # coh0 and coh1 tuned recurrent e units to coh1 trials
    # coh0 and coh1 tuned recurrent i units to coh1 trials

    # essentially in the next plot we want to see their weights changing over time; this is rate
    # maybe you can do average rate of the tuned units as just a line thing over time.... yes okay
    # try out a couple different ways of visualization. that's tomorrow.
    # why am I reinventing the wheel? take a lot of what you've done before

    ax[1].hist(np.array(all_coh0_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh0_i_ct)))+' avg units)')
    ax[2].hist(np.array(all_coh1_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh1_e_ct)))+' avg units)')
    ax[3].hist(np.array(all_coh1_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh1_i_ct)))+' avg units)')
    """

    for j in range(0,len(ax)):
        ax[j].set_ylabel('density',fontname='Ubuntu')
        ax[j].set_xlabel('rates (Hz)',fontname='Ubuntu')
        ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    if mix_tuned_indices:
        save_fname = spath+'/characterize_tuning_mixedidx_rate_test.png'
        plt.suptitle('Rates of tuned recurrent units; tuning defined within state',fontname='Ubuntu')
    else:
        save_fname = spath+'/characterize_tuning_trainedidx_rate_test.png'
        plt.suptitle('Rates of tuned recurrent units; tuning defined by trained state',fontname='Ubuntu')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()


    if plot_counts:
        ######{{{{   CREATE NEW PLOT SHOWING COUNTS OF TUNED E AND I UNITS IN NAIVE AND TRAINED STATES   }}}}#######
        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()

        # want to visually compare the avg number of tuned units in naive and trained cases

        ax[0].hist(np.array(trained_ct[2]).flatten(),alpha=0.7,density=True,label='coh0-tuned trained')
        ax[0].hist(np.array(trained_ct[0]).flatten(),alpha=0.7,density=True,label='coh1-tuned trained')
        ax[0].hist(np.array(naive_ct[2]).flatten(),alpha=0.7,density=True,label='coh0-tuned naive')
        ax[0].hist(np.array(naive_ct[0]).flatten(),alpha=0.7,density=True,label='coh1-tuned naive')
        ax[0].set_title('Number of E units that are tuned',fontname='Ubuntu')

        ax[1].hist(np.array(trained_ct[3]).flatten(),alpha=0.7,density=True,label='coh0-tuned trained')
        ax[1].hist(np.array(trained_ct[1]).flatten(),alpha=0.7,density=True,label='coh1-tuned trained')
        ax[1].hist(np.array(naive_ct[3]).flatten(),alpha=0.7,density=True,label='coh0-tuned naive')
        ax[1].hist(np.array(naive_ct[1]).flatten(),alpha=0.7,density=True,label='coh1-tuned naive')
        ax[1].set_title('Number of I units that are tuned',fontname='Ubuntu')

        plt.suptitle('Quantities of tuned recurrent units',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('density',fontname='Ubuntu')
            ax[j].set_xlabel('number of units',fontname='Ubuntu')
            ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        save_fname = spath+'/count_tuning_rate_test.png'
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()


    ######{{{{   FOR LATER: CREATE NEW PLOT SHOWING THE RATE DISTRIBUTIONS OF NAIVE-TUNED-INDEXED E AND I UNITS IN BOTH NAIVE AND TRAINED TRIALS   }}}}#######

    # want a way to also quantify the extent to which units are tuned
    # like the ratio of their mean responses to i vs their mean responses to e
    # in both the trained and naive states for both coherence level trials
    # and for different varieties of training
    # this is the task for tomorrow

    return [trained_ct,naive_ct]


# plot over the course of training how MANY units become tuned

"""
def tuned_rec_layer_over_training():
    # plot over the course of training with shaded error bars
    # plot the average weight within and between coherence tuning of recurrent layer units
    # make sure all axes are comparable
    # get the numbers (avg and std weight for all of these connection types? shape tho?) for the weight distributions at the beginning and end of training



# below this line are NOT priorities for now

#def input_amp_or_supp_based_on_training(dual_exp_dir,task_exp_dir,rate_exp_dir):
# this is the one where we were looking at single trials
# you require a way to characterize

def labeled_lines(exp_dir=spec_input_dirs,exp_season='winter'):
    # demonstrate the relationship between sum of input weights and sum of output weights
    # do so across all experiments of this type
    # plot some sort of line? or something? to show the relationship?
"""
