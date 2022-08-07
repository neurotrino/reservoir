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
from utils.misc import generic_filenames
from utils.misc import get_experiments
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree
from utils.extools.MI import simple_confMI

data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-savey'
num_epochs = 1000
epochs_per_file = 10
e_end = 240
i_end = 300
savepath = '/data/results/experiment1/'

e_only = True
positive_only = False

# next thing is to do this for just the units that project to output, rather than the whole e network

def plot_fn_quad_metrics():
    # even though it's so simple (just 4 values per xdir),
    # it'll be useful to compare across xdirs and training
    metric_mat = calculate_fn_quad_metrics()
    np.save(os.path.join(savepath,'set_fn_quad_metrics_withnegative.npy'), metric_mat)
    # shaped [4-metrics,number-of-experiments,2-coherence-levels,4-epochs]
    labels=['average weight','density','reciprocity','weighted clustering']
    epochs = [0,10,100,1000]
    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax=ax.flatten()
    for i in range(4): # for each of the four metrics
        for j in range(np.shape(metric_mat)[0]): # for each experiment
            # plot for both coherence levels
            ax[i].plot(epochs,metric_mat[i][j][0])
            ax[i+4].plot(epochs,metric_mat[i][j][1])
        ax[i].set_title(labels[i]+' coherence level 0')
        ax[i+4].set_title(labels[i]+' coherence level 1')
        ax[i].set_ylabel(labels[i])
        ax[i+4].set_ylabel(labels[i])
    for i in range(8):
        ax[i].set_xlabel('epoch')
    fig.suptitle('functional graph metrics for just 4 epochs')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_fn_quad_metrics_withnegative.png"),dpi=300)
    plt.clf()
    plt.close()

def calculate_fn_quad_metrics():
    # for now, metrics are mean weight, density, reciprocity, clustering
    # each metric is sized [number-of-experiments,2-coherence-levels,4-epochs,]
    w_mat = []
    dens_mat = []
    recips_mat = []
    ccs_mat = []
    experiments = get_experiments(data_dir, experiment_string)
    for xdir in experiments:
        xdir_quad_fn = generate_quad_fn(xdir, e_only, positive_only)
        n_epochs = np.shape(xdir_quad_fn)[0]
        ws = np.zeros([2,n_epochs])
        dens = np.zeros([2,n_epochs])
        recips = np.zeros([2,n_epochs])
        ccs = np.zeros([2,n_epochs])
        for i in range(np.shape(xdir_quad_fn)[1]): # for each of the 2 coherence levels:
            for j in range(n_epochs): # for each of the four epochs
                # flipping indices from epoch,coherence to coherence,epoch
                # calculate mean weight
                ws[i][j]=np.average(xdir_quad_fn[j][i])
                # calculate density
                dens[i][j]=calc_density(xdir_quad_fn[j][i])
                # calculate reciprocity
                recips[i][j]=reciprocity(xdir_quad_fn[j][i])
                # calculate weighted clustering coefficient
                # nx clustering is still not supported for negative values (it will create complex cc values)
                # so, use absolute values for now
                G = nx.from_numpy_array(np.abs(xdir_quad_fn[j][i]),create_using=nx.DiGraph)
                ccs[i][j]=nx.average_clustering(G,nodes=G.nodes,weight='weight')
        w_mat.append(ws)
        dens_mat.append(dens)
        recips_mat.append(recips)
        ccs_mat.append(ccs)
    return [w_mat,dens_mat,recips_mat,ccs_mat]

def plot_fn_w_dist_experiments():
    # 4 subplots
    experiments = get_experiments(data_dir, experiment_string)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        plt_string = ['epoch0','epoch10','epoch100','epoch1000']

        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath,exp_path)):
            os.makedirs(os.path.join(savepath,exp_path))

        xdir_quad_fn = generate_quad_fn(xdir, e_only, positive_only)
        # sized [4,2,240,240]

        for i in range(np.shape(xdir_quad_fn)[0]):
            plt.figure()
            # plot coherence level 0 fn weights
            sns.histplot(data=np.ravel(xdir_quad_fn[i][0]), bins=30, color='blue', label='coherence 0', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot coherence level 1 fn weights
            sns.histplot(data=np.ravel(xdir_quad_fn[i][1]), bins=30, color='red', label='coherence 1', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('functional weight distribution')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_fn_wdist.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()

def generate_quad_fn(xdir, e_only, positive_only):
    # simply generate four functional networks for a given xdir within experiments
    # can then be used to make quick plots
    # epochs = ['epoch0','epoch10','epoch100','epoch1000']
    xdir_quad_fn = []
    data_files = []
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))
    spikes = []
    true_y = []
    data = np.load(data_files[0])
    # batch 0 (completely naive) spikes
    spikes.append(data['spikes'][0])
    true_y.append(data['true_y'][0])
    for i in range(3): # load other spikes
        data = np.load(data_files[i])
        spikes.append(data['spikes'][99])
        true_y.append(data['true_y'][99])
    for i in range(4):
        [fn_coh0, fn_coh1] = generate_batch_ccd_functional_graphs(spikes[i],true_y[i],e_only,positive_only)
        xdir_quad_fn.append([fn_coh0, fn_coh1])
    # returns [epoch,coherence,FN]
    # sized [4,2,240,240]
    return xdir_quad_fn

def generate_batch_ccd_functional_graphs(spikes,true_y,e_only,positive_only):
    # calculate 2 FNs (1 for each coherence level) for each batch of 30 trials
    if e_only:
        spikes_coh0 = np.empty([e_end,0])
        trialends_coh0 = []
        spikes_coh1 = np.empty([e_end,0])
        trialends_coh1 = []
        for trial in range(np.shape(true_y)[0]): # each of 30 trials per batch update
            trial_y = np.squeeze(true_y[trial])
            # separate spikes according to coherence level
            coh_0_idx = np.squeeze(np.where(trial_y==0))
            coh_1_idx = np.squeeze(np.where(trial_y==1))
            spikes_trial = np.transpose(spikes[trial])
            # get all the spikes for each coherence level strung together
            # get indices of trial_ends
            if np.size(coh_0_idx)!=0:
                spikes_coh0 = np.hstack([spikes_coh0,spikes_trial[0:e_end,coh_0_idx]])
                trialends_coh0.append(np.shape(spikes_coh0)[1]-1)
            if np.size(coh_1_idx)!=0:
                spikes_coh1 = np.hstack([spikes_coh1,spikes_trial[0:e_end,coh_1_idx]])
                trialends_coh1.append(np.shape(spikes_coh1)[1]-1)
        # pipe into confMI calculation
        fn_coh0 = simple_confMI(spikes_coh0,trialends_coh0,positive_only,lag=1)
        fn_coh1 = simple_confMI(spikes_coh1,trialends_coh1,positive_only,lag=1)
        return [fn_coh0, fn_coh1]

# this function generates functional graphs (using confMI) to save
# so that we'll have them on hand for use in all the analyses we need
def generate_all_functional_graphs(experiment_string, overwrite=False, e_only=True, positive_only=False):
    # do not overwrite already-saved files that contain generated functional networks
    # currently working only with e units
    # positive_only=False means we DO include negative confMI values (negative correlations)
    # previously we had always removed those, but now we'll try to make sense of negative correlations as we go
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    save_files = generic_filenames(num_epochs, epochs_per_file)
    MI_savepath = os.path.join(savepath,"MI_graphs")
    for xdir in experiments:
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(MI_savepath,exp_path)):
            os.makedirs(os.path.join(MI_savepath,exp_path))
        # check if FN folders have already been generated
        # for every batch update, there should be 2 FNs for the 2 coherence levels
        # currently we are only generating FNs for e units, not yet supporting i units
        if e_only:
            subdir_names = ['e_coh0','e_coh1']
            for subdir in subdir_names:
                if not os.path.isdir(os.path.join(MI_savepath,exp_path,subdir)):
                    os.makedirs(os.path.join(MI_savepath,exp_path,subdir))
            for file_idx in range(np.size(data_files)):
                filepath = os.path.join(data_dir, xdir, 'npz-data', data_files[file_idx])
                # check if we haven't generated FNs already
                if not os.path.isfile(os.path.join(MI_savepath,exp_path,subdir_names[0],save_files[file_idx])) or overwrite:
                    data = np.load(filepath)
                    spikes = data['spikes']
                    true_y = data['true_y']
                    # generate MI graph from spikes for each coherence level
                    fns_coh0 = []
                    fns_coh1 = []
                    for batch in range(np.shape(true_y)[0]): # each file contains 100 batch updates
                    # each batch update has 30 trials
                    # those spikes and labels are passed to generate FNs batch-wise
                        [batch_fn_coh0, batch_fn_coh1] = generate_batch_ccd_functional_graphs(spikes[batch],true_y[batch],e_only,positive_only)
                        fns_coh0.append(batch_fn_coh0)
                        fns_coh1.append(batch_fn_coh1)
                    # saving convention is same as npz data files
                    # within each subdir (e_coh0 or e_coh1), save as 1-10.npy for example
                    # the data is sized [100 batch updates, 240 pre e units, 240 post e units]
                    np.save(os.path.join(MI_savepath,exp_path,subdir_names[0],save_files[file_idx]), fns_coh0)
                    np.save(os.path.join(MI_savepath,exp_path,subdir_names[1],save_files[file_idx]), fns_coh1)


def generate_recruitment_graphs(save_graphs=True):
    # load in functional graphs
    # load in synaptic graphs
    # load in spikes
    # find active units in synaptic graph
    # take their weights from the functional graph
    # save recruitment graphs
    savedir = os.path.join(savepath,"recruitment_graphs")

def plot_rates_over_time(output_only=True):
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
            output_w = data['tv2.postweights']
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
                # find the units that have nonzero projections to the output
                if output_only:
                    batch_w = np.reshape(output_w[i],np.shape(output_w[i])[0])
                    e_out_idx = np.squeeze(np.where(batch_w>0))
                    i_out_idx = np.squeeze(np.where(batch_w<0))
                for j in range(np.shape(y)[1]): # each of 30 trials per batch update
                    batch_y = np.squeeze(y[i][j])
                    coh_0_idx = np.squeeze(np.where(batch_y==0))
                    coh_1_idx = np.squeeze(np.where(batch_y==1))
                    spikes_trial = np.transpose(spikes[i][j])
                    if np.size(coh_0_idx)!=0:
                        if output_only:
                            batch_e_0_rate.append(np.mean(spikes_trial[e_out_idx[:,None],coh_0_idx[None,:]]))
                            batch_i_0_rate.append(np.mean(spikes_trial[i_out_idx[:,None],coh_0_idx[None,:]]))
                        else:
                            batch_e_0_rate.append(np.mean(spikes_trial[0:e_end,coh_0_idx]))
                            batch_i_0_rate.append(np.mean(spikes_trial[e_end:i_end,coh_0_idx]))
                    if np.size(coh_1_idx)!=0:
                        if output_only:
                            batch_e_1_rate.append(np.mean(spikes_trial[e_out_idx[:,None],coh_1_idx[None,:]]))
                            batch_i_1_rate.append(np.mean(spikes_trial[i_out_idx[:,None],coh_1_idx[None,:]]))
                        else:
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
    ax[0].set_title('e-to-output units, coherence 0')
    ax[1].set_title('e-to-output units, coherence 1')
    ax[2].set_title('i-to-output units, coherence 0')
    ax[3].set_title('i-to-output units, coherence 1')
    # Create and save the final figure
    fig.suptitle('experiment set 1.5 rates according to coherence level')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_output_rates.png"),dpi=300)
    plt.clf()
    plt.close()

#def plot_synch_over_time():
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
    return net_bscore"""

def simple_branching_param(bin_size, spikes): # spikes in shape of [units, time]
    run_time = np.shape(spikes)[1]
    nbins = int(np.round(run_time/bin_size))

    # for every pair of timesteps, determine the number of ancestors and the number of descendants
    numA = np.zeros([nbins-1]); # number of ancestors for each bin
    numD = np.zeros([nbins-1]); # number of descendants for each ancestral bin

    for i in range(nbins-1):
        numA[i] = np.size(np.argwhere(spikes[:,i]==1))
        numD[i] = np.size(np.argwhere(spikes[:,i+bin_size]==1))

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
