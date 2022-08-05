"""Dynamic (spike) analysis for series of completed experiments"""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import networkx as nx

sys.path.append('../')
sys.path.append('../../')

# internal ----
from utils.misc import filenames
from utils.misc import get_experiments
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree

data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-savey'
num_epochs = 1000
epochs_per_file = 10
e_end = 240
i_end = 300
savepath = '/data/results/experiment1/'

def plot_rates_over_time():
    # separate into coherence level 1 and coherence level 0
    experiments = get_experiments(data_dir, experiment_string)
    # plot for each experiment, one rate value per coherence level per batch update
    # this means rates are averaged over entire runs (or section of a run by coherence level) and 30 trials for each update
    # do rates of e units only
    # do rates of i units only
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()
    # subplot 0: coherence level 0, e units, avg rate (for batch of 30 trials) over training time
    # subplot 1: coherence level 1, e units, avg rate (for batch of 30 trials) over training time
    # subplot 2: coherence level 0, i units, avg rate (for batch of 30 trials) over training time
    # subplot 3: coherence level 1, i units, avg rate (for batch of 30 trials) over training time
    for xdir in experiments:
        e_0_rate = []
        e_1_rate = []
        i_0_rate = []
        i_1_rate = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            spikes = data['spikes']
            y = data['true_y']
            y.resize([np.shape(y)[0],np.shape(y)[1],np.shape(y)[2]])
            for i in range(np.shape(y)[0]): # each file contains 100 batch updates
                # find indices for coherence level 0 and for 1
                # do this for each of 30 trials bc memory can't accommodate the whole batch
                # this also circumvents continuity problems for calculating branching etc
                # then calculate rate of spikes for each trial according to coherence level idx
                batch_e_0_rate = []
                batch_e_1_rate = []
                batch_i_0_rate = []
                batch_i_1_rate = []
                for j in range(np.shape(y)[1]):
                    coh_0_idx = np.argwhere(y[i][j]==0)
                    coh_1_idx = np.argwhere(y[i][j]==1)
                    spikes_trial = np.transpose(spikes[i][j])
                    if np.size(coh_0_idx)!=0:
                        batch_e_0_rate.append(np.mean(spikes_trial[0:e_end,coh_0_idx]))
                        batch_i_0_rate.append(np.mean(spikes_trial[e_end:i_end,coh_0_idx]))
                    if np.size(coh_1_idx)!=0:
                        batch_e_1_rate.append(np.mean(spikes_trial[0:e_end,coh_1_idx]))
                        batch_i_1_rate.append(np.mean(spikes_trial[e_end:i_end,coh_1_idx]))
                e_0_rate.append(np.mean(batch_e_0_rate))
                e_1_rate.append(np.mean(batch_e_1_rate))
                i_0_rate.append(np.mean(batch_i_0_rate))
                i_1_rate.append(np.mean(batch_i_1_rate))
        ax[0].plot(e_0_rate)
        ax[1].plot(e_1_rate)
        ax[2].plot(i_0_rate)
        ax[3].plot(i_1_rate)
    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('rate')
    ax[0].set_title('e units, coherence 0')
    ax[1].set_title('e units, coherence 1')
    ax[2].set_title('i units, coherence 0')
    ax[3].set_title('i units, coherence 1')
    # Create and save the final figure
    fig.suptitle('experiment set 1.5 rates according to coherence level')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_rates.png"),dpi=300)
    plt.clf()
    plt.close()

def plot_synch_over_time():
    # use the fast measure

"""
def branching_param(bin_size,spikes): # spikes in shape of [units, time]
    run_time = np.shape(spikes)[1]
    nbins = np.round(run_time/bin_size)
    branch_over_time = np.zeros([nbins])

    # determine number of neurons which spiked at least once
    Nactive = 0
    for i in range(np.shape(spikes)[0]):
        if np.size(np.argwhere(spikes[i,:]==1))!=0:
            Nactive+=1

    # for every pair of timesteps, determine the number of ancestors and the number of descendants
    numA = np.zeros([nbins-1]); # number of ancestors for each bin
    numD = np.zeros([nbins-1]); # number of descendants for each ancestral bin
    d = np.zeros([nbins-1]); # the ratio of electrode descendants per ancestor

    for i in range(nbins-1):
        numA[i] = np.size(spikes[:,i]==1)
        numD[i] = np.size(spikes[:,i+bin]==1)
        d[i] = np.round(numD[i]/numA[i])

    # filter out infs and nans
    d = d[~numpy.isnan(d)]
    d = d[~numpy.isinf(d)]
    dratio = np.unique(d)
    pd = np.zeros([np.size(dratio)])
    Na = np.sum(numA)
    norm = (Nactive-1)/(Nactive-Na) # correction for refractoriness
    # and then what do I do with this?

    i = 1
    for ii in range(np.size(dratio)):
        idx = np.argwhere(d==dratio[ii])
        if np.size(idx)!=0:
            nad = np.sum(numA[idx])
            pd[i] = (nad/Na)
        i+=1

    net_bscore = np.sum(dratio*pd)
    return net_bscore
"""

def simple_branching_param(bin_size, spikes): # spikes in shape of [units, time]
    run_time = np.shape(spikes)[1]
    nbins = np.round(run_time/bin_size)

    # for every pair of timesteps, determine the number of ancestors and the number of descendants
    numA = np.zeros([nbins-1]); # number of ancestors for each bin
    numD = np.zeros([nbins-1]); # number of descendants for each ancestral bin

    for i in range(nbins):
        numA[i] = np.size(spikes[:,i]==1)
        numD[i] = np.size(spikes[:,i+bin]==1)

    # the ratio of descendants per ancestor
    d = numD/numA
    # if we get a nan, that means there were no ancestors in the previous time point
    # in that case it probably means our choice of bin size is wrong
    # but to handle it for now we should probably just disregard
    # if we get a 0, that means there were no descendants in the next time point
    # 0 in that case is correct, because branching 'dies'
    # however, that's also incorrect because it means we are choosing our bin size wrong for actual synaptic effects!
    # will revisit this according to time constants
    bscore = np.nanmean(d)

    return bscore


def plot_branching_over_time():
    # count spikes in adjacent time bins
    # or should they be not adjacent?
    bin_size = 1 # for now, adjacent pre-post bins are just adjacent ms
    # separate into coherence level 1 and coherence level 0
    experiments = get_experiments(data_dir, experiment_string)
    # plot for each experiment, one branching value per coherence level per batch update
    # this means branching params are averaged over entire runs (or section of a run by coherence level) and 30 trials for each update
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()
    # subplot 0: coherence level 0, e units, avg branching (for batch of 30 trials) over training time
    # subplot 1: coherence level 1, e units, avg branching (for batch of 30 trials) over training time
    # subplot 2: coherence level 0, i units, avg branching (for batch of 30 trials) over training time
    # subplot 3: coherence level 1, i units, avg branching (for batch of 30 trials) over training time
    for xdir in experiments:
        e_0_branch = []
        e_1_branch = []
        i_0_branch = []
        i_1_branch = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            spikes = data['spikes']
            y = data['true_y']
            y.resize([np.shape(y)[0],np.shape(y)[1],np.shape(y)[2]]) # remove trailing dimension
            for i in range(np.shape(y)[0]): # each file contains 100 batch updates
                # find indices for coherence level 0 and for 1
                # do this for each of 30 trials bc memory can't accommodate the whole batch
                # this also circumvents continuity problems for calculating branching etc
                # then calculate rate of spikes for each trial according to coherence level idx
                batch_e_0_branch = []
                batch_e_1_branch = []
                batch_i_0_branch = []
                batch_i_1_branch = []
                for j in range(np.shape(y)[1]):
                    coh_0_idx = np.argwhere(y[i][j]==0)
                    coh_1_idx = np.argwhere(y[i][j]==1)
                    spikes_trial = np.transpose(spikes[i][j])
                    if np.size(coh_0_idx)!=0:
                        batch_e_0_branch.append(simple_branching_param(bin_size,spikes_trial[0:e_end,coh_0_idx]))
                        batch_i_0_branch.append(simple_branching_param(bin_size,spikes_trial[e_end:i_end,coh_0_idx]))
                    if np.size(coh_1_idx)!=0:
                        batch_e_1_branch.append(simple_branching_param(bin_size,spikes_trial[0:e_end,coh_1_idx]))
                        batch_i_1_branch.append(simple_branching_param(bin_size,spikes_trial[e_end:i_end,coh_1_idx]))
                e_0_branch.append(np.mean(batch_e_0_branch))
                e_1_branch.append(np.mean(batch_e_1_branch))
                i_0_branch.append(np.mean(batch_i_0_branch))
                i_1_branch.append(np.mean(batch_i_1_branch))
        ax[0].plot(e_0_branch)
        ax[1].plot(e_1_branch)
        ax[2].plot(i_0_branch)
        ax[3].plot(i_1_branch)
    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('branching parameter')
    ax[0].set_title('e units, coherence 0')
    ax[1].set_title('e units, coherence 1')
    ax[2].set_title('i units, coherence 0')
    ax[3].set_title('i units, coherence 1')
    # Create and save the final figure
    fig.suptitle('experiment set 1.5 branching according to coherence level')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_branching.png"),dpi=300)
    plt.clf()
    plt.close()
