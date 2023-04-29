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

data_files = filenames(num_epochs, epochs_per_file)

#ALL DUAL TRAINED TO BEGIN WITH:
spec_output_dirs = ["run-batch30-specout-onlinerate0.1-savey","run-batch30-dualloss-silence","run-batch30-dualloss-swaplabels"]
spec_input_dirs = ["run-batch30-dualloss-specinput0.3-rewire"]
spec_nointoout_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
save_inz_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
spec_nointoout_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
spec_nointoout_dirs_task = ["run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
all_spring_dual_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
all_save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]

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
"""

# SEE IF YOU CAN COMPLETE ALL THE BELOW TODAY
# ONE PER HOUR, SUPER DOABLE

def dists_of_input_rates(exp_dirs=all_save_inz_dirs,exp_season='spring',make_plots=True):
    # also bring in the rate trained ones too. just anything that contains saveinz; also the original CNN outputs too

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    # from CNN (original CNN output file)
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]
    y = np.squeeze(y)

    # from actual data
    # from actual experiment now
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)

            data = np.load(filepath)
            # simply too many if we don't just take the final batch
            in_spikes = data['inputs'][99]
            true_y = data['true_y'][99]
            if '-swaplabels' in xdir: # not unswaplabels
                true_y = ~true_y.astype(int) + 2

            true_y = np.reshape(true_y,[np.shape(true_y)[0],seq_len])
            in_spikes = np.reshape(in_spikes,[np.shape(in_spikes)[0],seq_len,np.shape(in_spikes)[2]])

            y=np.vstack([y,true_y])
            x=np.vstack([x,in_spikes])

    # for each of ALL trials (from CNN and experimental)
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

    # average across all trials for a given channel (16)
    coh0_rates = np.average(coh0_channel_trial_rates,0)
    coh1_rates = np.average(coh1_channel_trial_rates,0)

    coh0_channels = np.where(np.mean(coh1_rates,0)<np.mean(coh0_rates,0))[0]
    coh1_channels = np.where(np.mean(coh1_rates,0)>np.mean(coh0_rates,0))[0]

    return [coh0_channels, coh1_channels]

    # plot the distribution of rates of each channel in response to the two coherence levels
    # do so for all 16 channels together (and measure, compare their distributions)
    # do so for each channel separately and describe just how close their firing rates are to each coherence level

    if make_plots:
        # PLOT RATES OF TUNED CHANNELS TOGETHER FOR COH0 AND COH1 TRIALS
        # stack over subplots: coh0 tuned on top, coh 1 tuned on bottom
        # adjust axes to be the same

        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()

        ax[0].hist(coh0_channel_trial_rates[:,coh0_channels],density=True,bins=30,alpha=0.7,label='rates to coh 0 trials')
        ax[0].hist(coh1_channel_trial_rates[:,coh0_channels],density=True,bins=30,alpha=0.7,label='rates to coh 1 trials')
        ax[0].set_title('Coherence 0 tuned input channels',fontname='Ubuntu')

        ax[1].hist(coh0_channel_trial_rates[:,coh1_channels],density=True,bins=30,alpha=0.7,label='rates to coh 0 trials')
        ax[1].hist(coh1_channel_trial_rates[:,coh1_channels],density=True,bins=30,alpha=0.7,label='rates to coh 1 trials')
        ax[1].set_title('Coherence 1 tuned input channels',fontname='Ubuntu')

        # hopefully can visually (and numerically) see that input channels don't differ all that much in their responses
        # even though they are sliiiightly tuned

        # PLOT RATES OF ALL CHANNELS TO COH0 AND COH1 (regardless of tuning)
        ax[2].hist(coh0_channel_trial_rates,density=True,bins=30,alpha=0.7)
        ax[2].set_title("All channels' rates to coh 0 trials",fontname='Ubuntu')

        ax[3].hist(coh1_channel_trial_rates,density=True,bins=30,alpha=0.7)
        ax[3].set_title("All channels' rates to coh 1 trials",fontname='Ubuntu')

        plt.suptitle('Responses of 16 input channels to different coherences',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('density',fontname='Ubuntu')
            ax[j].set_xlabel('rates (Hz)',fontname='Ubuntu')
            if j < 2:
                ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        save_fname = spath+'/input_channel_rates_test.png'

        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()

    return [coh0_channels,coh1_channels]

def get_input_tuning_single_exp(xdir):

    for filename in data_files:
        filepath = os.path.join(data_dir, xdir, "npz-data", filename)
        data = np.load(filepath)
        input_z = data['inputs']
        # shaped [100 batches x 30 trials x 4080 timesteps x 16 units]
        true_y = data['true_y'] # shaped [100 batches x 30 trials x 4080 timesteps]
        for i in range(0,np.shape(true_y)[0]): # for batch
            for j in range(0,np.shape(true_y)[1]): # for trial
                coh0_idx = np.where(true_y[i][j]==0)[0]
                coh1_idx = np.where(true_y[i][j]==1)[0]
                if len(coh0_idx)>0:
                    if not 'coh0_channel_trial_rates' in locals():
                        coh0_channel_trial_rates = np.average(input_z[i][j][coh0_idx],0)
                    else:
                        coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(input_z[i][j][coh0_idx],0)])

                if len(coh1_idx)>0:
                    if not 'coh1_channel_trial_rates' in locals():
                        coh1_channel_trial_rates = np.average(input_z[i][j][coh1_idx],0)
                    else:
                        coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(input_z[i][j][coh1_idx],0)])

    coh1_channel_rates = np.array(np.mean(coh1_channel_trial_rates,0))
    coh0_channel_rates = np.array(np.mean(coh0_channel_trial_rates,0))
    coh1_idx = np.where(coh1_channel_rates>coh0_channel_rates)[0]
    coh0_idx = np.where(coh1_channel_rates<coh0_channel_rates)[0]

    return [coh0_idx,coh1_idx]

def input_layer_over_training_by_coherence(dual_exp_dir=save_inz_dirs,rate_exp_dir=save_inz_dirs_rate,exp_season='spring'):
    # characterize the connectivity from the input layer to recurrent
    # plot over the course of training with shaded error bars
    # compare for rate- and dual-trained

    # ACTUALLY YOU NEED TO DO THIS FOR INDIVIDUAL EXPERIMENTS BECAUSE WE ARE FOLLOWING LABELS NOT ACTUAL COHERENCE

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

    # aggregate over all experiments

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        coh1_e_exp = []
        coh1_i_exp = []
        coh0_e_exp = []
        coh0_i_exp = []
        epoch_task_loss_exp = []
        epoch_rate_loss_exp = []

        [coh0_idx, coh1_idx] = get_input_tuning_single_exp(xdir)

        # get the truly naive weights
        filepath = os.path.join(data_dir,xdir,"npz-data","input_preweights.npy")
        input_w = np.load(filepath)
        coh1_e_exp.append(np.mean(input_w[coh1_idx,:e_end]))
        coh1_i_exp.append(np.mean(input_w[coh1_idx,e_end:]))
        coh0_e_exp.append(np.mean(input_w[coh0_idx,:e_end]))
        coh0_i_exp.append(np.mean(input_w[coh0_idx,e_end:]))

        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_w = data['tv0.postweights'][0] # just the singular for now; too much data and noise otherwise
            epoch_task_loss_exp.append(np.mean(data['step_task_loss']))
            epoch_rate_loss_exp.append(np.mean(data['step_rate_loss']))
            #for i in range(0,np.shape(input_w)[0]): # 100 trials
            # weights of each type to e units and to i units
            coh1_e_exp.append(np.mean(input_w[coh1_idx,:e_end]))
            coh1_i_exp.append(np.mean(input_w[coh1_idx,e_end:]))
            coh0_e_exp.append(np.mean(input_w[coh0_idx,:e_end]))
            coh0_i_exp.append(np.mean(input_w[coh0_idx,e_end:]))

        if not "coh1_e" in locals():
            coh1_e = coh1_e_exp
        else:
            coh1_e = np.vstack([coh1_e, coh1_e_exp])

        if not "coh1_i" in locals():
            coh1_i = coh1_i_exp
        else:
            coh1_i = np.vstack([coh1_i, coh1_i_exp])

        if not "coh0_e" in locals():
            coh0_e = coh0_e_exp
        else:
            coh0_e = np.vstack([coh0_e, coh0_e_exp])

        if not "coh0_i" in locals():
            coh0_i = coh0_i_exp
        else:
            coh0_i = np.vstack([coh0_i, coh0_i_exp])

        if not "epoch_task_loss" in locals():
            epoch_task_loss = epoch_task_loss_exp
        else:
            epoch_task_loss = np.vstack([epoch_task_loss, epoch_task_loss_exp])

        if not "epoch_rate_loss" in locals():
            epoch_rate_loss = epoch_rate_loss_exp
        else:
            epoch_rate_loss = np.vstack([epoch_rate_loss, epoch_rate_loss_exp])

    fig, ax = plt.subplots(nrows=3, ncols=1)

    print('shape of coh1_e (coherence 1 tuned input channels to recurrent excitatory units): ')
    print(np.shape(coh1_e))
    coh1_e_mean = np.mean(coh1_e,0)
    coh1_e_std = np.std(coh1_e,0)
    coh0_e_mean = np.mean(coh0_e,0)
    coh0_e_std = np.std(coh0_e,0)

    ax[0].plot(coh1_e_mean, label='coh 1 tuned inputs', color='slateblue')
    ax[0].fill_between(coh1_e_mean, coh1_e_mean-coh1_e_std, coh1_e_mean+coh1_e_std, alpha=0.4, facecolor='slateblue')
    ax[0].plot(coh0_e_mean, label='coh 0 tuned inputs', color='mediumseagreen')
    ax[0].fill_between(coh0_e_mean, coh0_e_mean-coh0_e_std, coh0_e_mean+coh0_e_std, alpha=0.4, facecolor='mediumseagreen')
    ax[0].set_title('input weights to excitatory units',fontname='Ubuntu')

    coh1_i_mean = np.mean(coh1_i,0)
    coh1_i_std = np.mean(coh1_i,0)
    coh0_i_mean = np.mean(coh0_i,0)
    coh0_i_std = np.mean(coh0_i,0)

    ax[1].plot(coh1_i_mean, label='coh 1 tuned inputs', color='darkorange')
    ax[1].fill_between(coh1_i_mean, coh1_i_mean-coh1_i_std, coh1_i_mean+coh1_i_std, alpha=0.4, facecolor='slateblue')
    ax[1].plot(coh0_i_mean, label='coh 0 tuned inputs', color='orangered')
    ax[1].fill_between(coh0_i_mean, coh0_i_mean-coh0_i_std, coh0_i_mean+coh0_i_std, alpha=0.4, facecolor='mediumseagreen')
    ax[1].set_title('input weights to inhibitory units',fontname='Ubuntu')

    task_mean = np.mean(epoch_task_loss,0)
    task_error = np.std(epoch_task_loss,0)
    ax[2].plot(task_mean, label='task loss', color='darkorange')
    ax[2].fill_between(task_mean, task_mean-task_error, task_mean+task_error, alpha=0.4, facecolor='darkorange')

    rate_mean = np.mean(epoch_rate_loss,0)
    rate_error = np.std(epoch_rate_loss,0)
    ax[2].plot(rate_mean, label='rate loss', color='orangered')
    ax[2].fill_between(rate_mean, rate_mean+rate_error, rate_mean+rate_error, alpha=0.4, facecolor='orangered') #other options include edgecolor='#CC4F1B', facecolor='#FF9848'

    ax[2].set_ylabel('loss',fontname='Ubuntu')
    #ax[2].legend(['task loss','rate loss'],fontsize="11",prop={"family":"Ubuntu"})

    for j in range(0,len(ax)):
        ax[j].set_ylabel('average weights',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Evolution of input weights over training',fontname='Ubuntu')
    plt.subplots_adjust(wspace=1.0, hspace=1.0)
    plt.draw()

    save_fname = spath+'/inputs_to_ei_test.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

    """
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
    """

    # get a number distribution to quantify this, maybe for each experiment
    # the ratio between avg weight from input coh0 and coh1 to e and i recurrent units at the beginning and at the end of training
    # 0_to_e/0_to_i = 1 at beginning
    # 1_to_e/1_to_i = 1 at beginning
    # 0_to_e/0_to_i < 1 at end
    # 1_to_e/1_to_e > 1 at end
    # that's a good start


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

def tuned_rec_layer_over_training(exp_dirs=all_spring_dual_dirs,exp_season='spring'):
    # plot over the course of training with shaded error bars
    # plot the average weight within and between coherence tuning of recurrent layer units
    # make sure all axes are comparable
    # get the numbers (avg and std weight for all of these connection types? shape tho?) for the weight distributions at the beginning and end of training

    # look at tuning to coherence level
    # look at connections between units in accordance to tuning to coherence level
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs:
        coh0_ee_ = []
        coh0_ei_ = []
        coh0_ie_ = []
        coh0_ii_ = []

        coh1_ee_ = []
        coh1_ei_ = []
        coh1_ie_ = []
        coh1_ii_ = []

        # coh 0 to 1
        het_ee_ = []
        het_ei_ = []
        het_ie_ = []
        het_ii_ = []

        # coh 1 to 0
        ero_ee_ = []
        ero_ei_ = []
        ero_ie_ = []
        ero_ii_ = []

        print('begin new exp')
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        data = np.load(os.path.join(np_dir,"991-1000.npz")) # define tuning based on trained trials

        # go thru final epoch trials
        true_y = data['true_y']
        spikes = data['spikes']
        w = data['tv1.postweights']

        # collect weights over all of training
        temporal_w = []
        data_files = filenames(num_epochs, epochs_per_file)

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            temp_data = np.load(filepath)
            temporal_w.append(temp_data['tv1.postweights'][99])

        # find which units respond more to input of a certain coh level across batches and trials
        coh0_rec_rates = []
        coh1_rec_rates = []

        for i in range(0,np.shape(true_y)[0]):
            for j in range(0,np.shape(true_y)[1]):
                if true_y[i][j][0]==true_y[i][j][seq_len-1]:
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        # find which of the 300 recurrent units respond more on average to one coherence level over the other
        coh1_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        """
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx<e_end]))+' coh1-tuned e units')
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx>=e_end]))+' coh1-tuned i units')
        """
        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        """
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx<e_end]))+' coh0-tuned e units')
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx>=e_end]))+' coh0-tuned i units')
        """

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        # just average weights to begin with?
        coh1_e = np.array(coh1_rec_idx[coh1_rec_idx<e_end])
        coh1_i = np.array(coh1_rec_idx[coh1_rec_idx>=e_end])
        coh0_e = np.array(coh0_rec_idx[coh0_rec_idx<e_end])
        coh0_i = np.array(coh0_rec_idx[coh0_rec_idx>=e_end])

        # name them as homo and hetero lol
        # plot weights based on coh tuning over time

        for i in range(0,np.shape(temporal_w)[0]): # again over all training time, but now just one per file (100) instead of craziness (10000)
            coh0_ee_.append(np.mean(temporal_w[i][coh0_e,:][:,coh0_e]))
            coh0_ei_.append(np.mean(temporal_w[i][coh0_e,:][:,coh0_i]))
            coh0_ie_.append(np.mean(temporal_w[i][coh0_i,:][:,coh0_e]))
            coh0_ii_.append(np.mean(temporal_w[i][coh0_i,:][:,coh0_i]))

            coh1_ee_.append(np.mean(temporal_w[i][coh1_e,:][:,coh1_e]))
            coh1_ei_.append(np.mean(temporal_w[i][coh1_e,:][:,coh1_i]))
            coh1_ie_.append(np.mean(temporal_w[i][coh1_i,:][:,coh1_e]))
            coh1_ii_.append(np.mean(temporal_w[i][coh1_i,:][:,coh1_i]))

            het_ee_.append(np.mean(temporal_w[i][coh0_e,:][:,coh1_e]))
            het_ei_.append(np.mean(temporal_w[i][coh0_e,:][:,coh1_i]))
            het_ie_.append(np.mean(temporal_w[i][coh0_i,:][:,coh1_e]))
            het_ii_.append(np.mean(temporal_w[i][coh0_i,:][:,coh1_i]))

            ero_ee_.append(np.mean(temporal_w[i][coh1_e,:][:,coh0_e]))
            ero_ei_.append(np.mean(temporal_w[i][coh1_e,:][:,coh0_i]))
            ero_ie_.append(np.mean(temporal_w[i][coh1_i,:][:,coh0_e]))
            ero_ii_.append(np.mean(temporal_w[i][coh1_i,:][:,coh0_i]))

        if not "coh0_ee" in locals():
            coh0_ee = coh0_ee_
        else:
            coh0_ee = np.vstack([coh0_ee, coh0_ee_])

        if not "coh0_ei" in locals():
            coh0_ei = coh0_ei_
        else:
            coh0_ei = np.vstack([coh0_ei, coh0_ei_])

        if not "coh0_ie" in locals():
            coh0_ie = coh0_ie_
        else:
            coh0_ie = np.vstack([coh0_ie, coh0_ie_])

        if not "coh0_ii" in locals():
            coh0_ii = coh0_ii_
        else:
            coh0_ii = np.vstack([coh0_ii, coh0_ii_])

        if not "coh1_ee" in locals():
            coh1_ee = coh1_ee_
        else:
            coh1_ee = np.vstack([coh1_ee, coh1_ee_])

        if not "coh1_ei" in locals():
            coh1_ei = coh1_ei_
        else:
            coh1_ei = np.vstack([coh1_ei, coh1_ei_])

        if not "coh1_ie" in locals():
            coh1_ie = coh1_ie_
        else:
            coh1_ie = np.vstack([coh1_ie, coh1_ie_])

        if not "coh1_ii" in locals():
            coh1_ii = coh1_ii_
        else:
            coh1_ii = np.vstack([coh1_ii, coh1_ii_])

        if not "het_ee" in locals():
            het_ee = het_ee_
        else:
            het_ee = np.vstack([het_ee, het_ee_])

        if not "het_ei" in locals():
            het_ei = het_ei_
        else:
            het_ei = np.vstack([het_ei, het_ei_])

        if not "het_ie" in locals():
            het_ie = het_ie_
        else:
            het_ie = np.vstack([het_ie, het_ie_])

        if not "het_ii" in locals():
            het_ii = het_ii_
        else:
            het_ii = np.vstack([het_ii, het_ii_])

        if not "ero_ee" in locals():
            ero_ee = ero_ee_
        else:
            ero_ee = np.vstack([ero_ee, ero_ee_])

        if not "ero_ei" in locals():
            ero_ei = ero_ei_
        else:
            ero_ei = np.vstack([ero_ei, ero_ei_])

        if not "ero_ie" in locals():
            ero_ie = ero_ie_
        else:
            ero_ie = np.vstack([ero_ie, ero_ie_])

        if not "ero_ii" in locals():
            ero_ii = ero_ii_
        else:
            ero_ii = np.vstack([ero_ii, ero_ii_])


    fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(8,10))

    coh0_ee_mean = np.mean(coh0_ee,0)
    coh0_ee_std = np.std(coh0_ee,0)
    ax[0].plot(coh0_ee_mean,color='slateblue',label='ee')
    ax[0].fill_between(coh0_ee_mean, coh0_ee_mean-coh0_ee_std, coh0_ee_mean+coh0_ee_std, alpha=0.4, facecolor='slateblue')

    coh0_ei_mean = np.mean(coh0_ei,0)
    coh0_ei_std = np.std(coh0_ei,0)
    ax[0].plot(coh0_ei_mean,color='mediumseagreen',label='ei')
    ax[0].fill_between(coh0_ei_mean, coh0_ei_mean-coh0_ei_std, coh0_ei_mean+coh0_ei_std, alpha=0.4, facecolor='mediumseagreen')

    coh0_ie_mean = np.mean(coh0_ie,0)
    coh0_ie_std = np.std(coh0_ie,0)
    ax[0].plot(coh0_ie_mean,color='darkorange',label='ie')
    ax[0].fill_between(coh0_ie_mean, coh0_ie_mean-coh0_ie_std, coh0_ie_mean+coh0_ie_std, alpha=0.4, facecolor='darkorange')

    coh0_ii_mean = np.mean(coh0_ii,0)
    coh0_ii_std = np.std(coh0_ii,0)
    ax[0].plot(coh0_ii_mean,color='orangered',label='ii')
    ax[0].fill_between(coh0_ii_mean, coh0_ii_mean-coh0_ii_std, coh0_ii_mean+coh0_ii_std, alpha=0.4, facecolor='orangered')

    ax[0].set_title('coherence 0 tuned recurrent connections',fontname='Ubuntu')
    ax[0].set_ylabel('average weight',fontname='Ubuntu')


    ax[1].plot(np.mean(coh1_ee,0),color='slateblue',label='ee')
    ax[1].fill_between(np.mean(coh1_ee,0), np.mean(coh1_ee,0)-np.std(coh1_ee,0), np.mean(coh1_ee,0)+np.std(coh1_ee,0), alpha=0.4, facecolor='slateblue')

    ax[1].plot(np.mean(coh1_ei,0),color='mediumseagreen',label='ei')
    ax[1].fill_between(np.mean(coh1_ei,0), np.mean(coh1_ei,0)-np.std(coh1_ei,0), np.mean(coh1_ei,0)+np.std(coh1_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[1].plot(np.mean(coh1_ie,0),color='darkorange',label='ie')
    ax[1].fill_between(np.mean(coh1_ie,0), np.mean(coh1_ie,0)-np.std(coh1_ie,0), np.mean(coh1_ie,0)+np.std(coh1_ie,0), alpha=0.4, facecolor='darkorange')

    ax[1].plot(np.mean(coh1_ii,0),color='orangered',label='ii')
    ax[1].fill_between(np.mean(coh1_ii,0), np.mean(coh1_ii,0)-np.std(coh1_ii,0), np.mean(coh1_ii,0)+np.std(coh1_ii,0), alpha=0.4, facecolor='orangered')

    ax[1].set_title('coherence 1 tuned recurrent connections',fontname='Ubuntu')
    ax[1].set_ylabel('average weight',fontname='Ubuntu')


    ax[2].plot(np.mean(het_ee,0),color='slateblue',label='ee')
    ax[2].fill_between(np.mean(het_ee,0), np.mean(het_ee,0)-np.std(het_ee,0), np.mean(het_ee,0)+np.std(het_ee,0), alpha=0.4, facecolor='slateblue')

    ax[2].plot(np.mean(het_ei,0),color='mediumseagreen',label='ei')
    ax[2].fill_between(np.mean(het_ei,0), np.mean(het_ei,0)-np.std(het_ei,0), np.mean(het_ei,0)+np.std(het_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[2].plot(np.mean(het_ie,0),color='darkorange',label='ie')
    ax[2].fill_between(np.mean(het_ie,0), np.mean(het_ie,0)-np.std(het_ie,0), np.mean(het_ie,0)+np.std(het_ie,0), alpha=0.4, facecolor='darkorange')

    ax[2].plot(np.mean(het_ii,0),color='orangered',label='ii')
    ax[2].fill_between(np.mean(het_ii,0), np.mean(het_ii,0)-np.std(het_ii,0), np.mean(het_ii,0)+np.std(het_ii,0), alpha=0.4, facecolor='orangered')

    ax[2].set_title('coherence 0 to coherence 1 tuned recurrent connections',fontname='Ubuntu')
    ax[2].set_ylabel('average weight',fontname='Ubuntu')


    ax[3].plot(np.mean(ero_ee,0),color='slateblue',label='ee')
    ax[3].fill_between(np.mean(ero_ee,0), np.mean(ero_ee,0)-np.std(ero_ee,0), np.mean(ero_ee,0)+np.std(ero_ee,0), alpha=0.4, facecolor='slateblue')

    ax[3].plot(np.mean(ero_ei,0),color='mediumseagreen',label='ei')
    ax[3].fill_between(np.mean(ero_ei,0), np.mean(ero_ei,0)-np.std(ero_ei,0), np.mean(ero_ei,0)+np.std(ero_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[3].plot(np.mean(ero_ie,0),color='darkorange',label='ie')
    ax[3].fill_between(np.mean(ero_ie,0), np.mean(ero_ie,0)-np.std(ero_ie,0), np.mean(ero_ie,0)+np.std(ero_ie,0), alpha=0.4, facecolor='darkorange')

    ax[3].plot(np.mean(ero_ii,0),color='orangered',label='ii')
    ax[3].fill_between(np.mean(ero_ii,0), np.mean(ero_ii,0)-np.std(ero_ii,0), np.mean(ero_ii,0)+np.std(ero_ii,0), alpha=0.4, facecolor='orangered')

    ax[3].set_title('coherence 1 to coherence 0 tuned recurrent connections',fontname='Ubuntu')
    ax[3].set_ylabel('average weight',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylim(bottom=-1.5,top=0.3)
        ax[j].set_xlabel('training epoch')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('recurrent connectivity by coherence tuning',fontname='Ubuntu')
    save_fname = spath+'/rec_weights_by_tuning_over_training_test.png'
    plt.subplots_adjust(hspace=0.8,wspace=0.8)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

"""
# below this line are NOT priorities for now

#def input_amp_or_supp_based_on_training(dual_exp_dir,task_exp_dir,rate_exp_dir):
# this is the one where we were looking at single trials
# you require a way to characterize

def labeled_lines(exp_dir=spec_input_dirs,exp_season='winter'):
    # demonstrate the relationship between sum of input weights and sum of output weights
    # do so across all experiments of this type
    # plot some sort of line? or something? to show the relationship?
"""
