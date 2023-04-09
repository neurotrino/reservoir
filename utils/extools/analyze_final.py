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


def determine_delays():
    # principled way to decide what constitutes the duration of a delay between coherence changes
    # plot to see delay markers on a few trials
    # get the experiments you want to begin with
    for exp_string in spec_output_dirs:
        if not 'fall_data_dirs' in locals():
            fall_data_dirs = get_experiments(data_dir, exp_string)
        else:
            fall_data_dirs = np.hstack([fall_data_dirs,get_experiments(data_dir, exp_string)])

    # go through the experiments and files
    for xdir in fall_data_dirs:
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir, xdir, "npz-data")
        #naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        # go through final epoch trials
        true_y = trained_data['true_y'][99]
        pred_y = trained_data['pred_y'][99]
        delay_durs = []
        change_times = []
        change_ys = []
        change_preds = []

        for i in range(0,len(true_y)):
            # check to see if there is a change in this trial
            if true_y[i][0] != true_y[i][seq_len-1]:
                change_ys.append(true_y[i])
                change_preds.append(pred_y[i])
                # find time of change
                diffs = np.diff(true_y[i],axis=0)
                # t_change is the first timestep of the new coherence level
                t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
                change_times.append(t_change)
                # find average pred before and after
                #pre_avg = np.average(pred_y[i][:t_change])
                post_avg = np.average(pred_y[i][t_change:])
                # determine the duration after coherence change until we first exceed the after-change average
                t_crossing = np.where(pred_y[i][t_change:]>post_avg)[0][0]
                # append
                delay_durs.append(t_crossing)

        """# plot the distribution of delays
        fig, ax = plt.hist(np.array(delay_durs).flatten())
        plt.xlabel('delay duration (ms)',fontname='Ubuntu')
        plt.ylabel('count',fontname='Ubuntu')
        plt.title('Delay Durations',fontname='Ubuntu')
        for tick in ax.get_xticklabels():
            tick.set_fontname('Ubuntu')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Ubuntu')
        plt.draw()
        save_fname = savepath+'/set_plots/fall/'+str(exp_path)+'_delay_dist_test.png'
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()"""

        # take average duration as The Delay
        delay = np.average(delay_durs)

        # select a few trials and plot randomly with The Delay to visually inspect
        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()
        trials = np.random.randint(0,len(delay_durs),size=[4])
        for i in range(0,len(trials)):
            ax[i].plot(change_ys[trials[i]],color='dodgerblue')
            ax[i].plot(change_preds[trials[i]],color='mediumblue')
            ax[i].vlines(change_times[trials[i]],ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='orangered')
            ax[i].vlines(change_times[trials[i]]+delay,ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='orangered')
            ax[i].vlines(change_times[trials[i]]+delay_durs[trials[i]],ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='darkorange')
            ax[i].set_xlabel('time (ms)',fontname='Ubuntu')
            ax[i].set_ylabel('output',fontname='Ubuntu')
            ax[i].legend(['true y','pred y','time of change','avg delay','trial delay'],fontname='Ubuntu')
            ax[i].set_title('trial '+str(trials[i]),fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname('Ubuntu')
            for tick in ax[i].get_yticklabels():
                tick.set_fontname('Ubuntu')

        plt.suptitle('Example Trials with Trialwise and Average Delays',fontname='Ubuntu')
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.7)

        save_fname = savepath+'/set_plots/fall/'+str(exp_path)+'_delay_trials_test.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


#def delay_MI_gen():
    # determine delays of all experiments
    # only look at the delay period of trained experiments
    # generate based on each batch of 30 trials
    # first look at just the final batch
    # generate based on pre-delay (coh 0 or 1 separate), delay (to coh 0 or to coh 1 separate), post-delay (coh 0 or 1 separate)
    # generate ummmmmmmmm recruitment graphs?


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
    fall_in_naive = []
    fall_in_trained = []
    fall_rec_naive = []
    fall_rec_trained = []
    fall_out_naive = []
    fall_out_trained = []
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

    # convert to numpy arrays
    fall_in_naive = np.array(fall_in_naive)
    fall_in_trained = np.array(fall_in_trained)
    fall_rec_naive = np.array(fall_rec_naive)
    fall_rec_trained = np.array(fall_rec_trained)
    fall_out_naive = np.array(fall_out_naive)
    fall_out_trained = np.array(fall_out_trained)

    # plot e and i separately
    ax[0,0].hist(fall_in_naive[:,:,0:e_end].flatten(),density=True)
    ax[0,0].hist(fall_in_naive[:,:,e_end:i_end].flatten(),density=True)
    ax[0,0].legend(['e edges','i edges'])
    ax[0,0].set_title('naive input weights',fontname='Ubuntu')

    ax[0,1].hist(fall_in_trained[:,0:e_end,0:e_end].flatten(),density=True)
    ax[0,1].hist(fall_in_trained[:,e_end:i_end,e_end:i_end].flatten(),density=True)
    ax[0,1].legend(['e edges','i edges'])
    ax[0,1].set_title('trained input weights',fontname='Ubuntu')

    # plot layers separately
    ax[1,0].hist(fall_rec_naive[:,0:e_end,:].flatten(),density=True)
    ax[1,0].hist(fall_rec_naive[:,e_end:i_end,:].flatten(),density=True)
    ax[1,0].legend(['e edges','i edges'])
    ax[1,0].set_title('naive recurrent weights',fontname='Ubuntu')

    ax[1,1].hist(fall_rec_trained[:,0:e_end].flatten(),density=True)
    ax[1,1].hist(fall_rec_trained[:,e_end:i_end].flatten(),density=True)
    ax[1,1].legend(['e edges','i edges'])
    ax[1,1].set_title('trained recurrent weights',fontname='Ubuntu')

    ax[2,0].hist(fall_out_naive[:,0:e_end].flatten(),density=True)
    ax[2,0].hist(fall_out_naive[:,e_end:i_end].flatten(),density=True)
    ax[2,0].legend(['e edges','i edges'])
    ax[2,0].set_title('naive output weights',fontname='Ubuntu')

    ax[2,1].hist(fall_out_trained[:,0:e_end].flatten(),density=True)
    ax[2,1].hist(fall_out_trained[:,e_end:i_end].flatten(),density=True)
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
