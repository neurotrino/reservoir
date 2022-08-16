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
bin = 10

# next thing is to do this for just the units that project to output, rather than the whole e network

def plot_fn_quad_metrics(load_saved=True):
    # even though it's so simple (just 4 values per xdir),
    # it'll be useful to compare across xdirs and training
    metric_data_file = os.path.join(savepath,'set_fn_quad_metrics_withnegative.npy')
    if not load_saved:
        metric_mat = calculate_fn_quad_metrics()
        np.save(metric_data_file, metric_mat)
    else:
        metric_mat = np.load(metric_data_file)
    # shaped [4-metrics,number-of-experiments,2-coherence-levels,4-epochs]
    labels=['average weight','density','reciprocity','clustering']
    epochs = [0,10,100,1000]
    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax=ax.flatten()
    ax_idx = [0,2,4,6]
    for i in range(4): # for each of the four metrics
        for j in range(np.shape(metric_mat)[0]): # for each experiment
            # plot for both coherence levels
            ax[ax_idx[i]].plot(epochs,metric_mat[i][j][0])
            ax[ax_idx[i]+1].plot(epochs,metric_mat[i][j][1])
        ax[ax_idx[i]].set_title('coherence level 0')
        ax[ax_idx[i]+1].set_title('coherence level 1')
        ax[ax_idx[i]].set_ylabel(labels[i])
        ax[ax_idx[i]+1].set_ylabel(labels[i])
    for i in range(8):
        ax[i].set_xlabel('epoch')
    fig.suptitle('functional graph metrics for just 4 epochs')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=1.5)
    plt.savefig(os.path.join(savepath,"set_fn_quad_metrics_withnegative.png"),dpi=300)
    plt.clf()
    plt.close()

def calculate_fn_quad_metrics(e_only,positive_only=False):
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

"""
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
"""

def batch_recruitment_graphs(w,fn,spikes,trialends,threshold):
    # w = synaptic graph
    # fn = functional network for a particular batch (and coherence level)
    # spikes = binned spikes for 30 trials of a particular batch (and coherence level)
    # trialends = the indices for which spikes reach a discontinuity
    # thus, the returned recruitment graphs will be shaped [trial(segment), timestep, pre units, post units]
    # threshold = 0.25 meaning we only use the top quartile of functional weights
    # otherwise we have a fully dense graph

    # threshold the functional graph
    # find the value of the top quartile for the fn
    # do so for both negative and positive values, so taking absolute value of the graph
    sorted_pos_fn = np.abs(np.unique(fn[fn!=0])) # sorted unique elements, not including 0
    threshold_idx = int((1-threshold)*np.size(sorted_pos_fn))
    threshold_val = sorted_pos_fn[threshold_idx]

    # return fn value wherever its absolute value is greater than threshold value, otherwise return 0
    upper_fn = np.where(np.abs(fn)>=threshold_val,fn,0)
    # so we could still get negative values

    # mask of 0's and 1's for whether actual synaptic connections exist
    w_bool = np.where(w!=0,1,0)

    trialstarts = np.concatenate(([0],trialends[:-1]))
    recruit_graphs = []

    # for each trial segment (determined by trialends):
    for i in range(np.size(trialstarts)):
        # aggregate recruitment graphs for this segment
        segment_dur = trialends[i] - trialstarts[i]
        recruit_segment = np.zeros([segment_dur,np.shape(upper_fn)[0],np.shape(upper_fn)[1]])
        # for each timestep within that segment:
        for t in range(trialstarts[i],trialends[i]):
            # find which units spiked in this timestep
            spike_idx = np.argwhere(spikes[:,t]==1)
            if spike_idx.size>0: # at least one unit spiked
                # find nonzero synaptic connections between the active units
                w_idx = []
                for u in spike_idx:
                    for v in spike_idx:
                        if w_bool[u,v]==1:
                            w_idx.append([u,v])
                w_idx = np.squeeze(w_idx)
                if w_idx.size>0: # we found at least one nonzero synaptic connection between the active units
                    # fill in recruitment graph at those existing active indices using values from tresholded functional graph
                    recruit_segment[t,w_idx,w_idx] = upper_fn[w_idx,w_idx] # for each trial segment (less than 30)
        recruit_graphs.append(recruit_segment) # aggregate for the whole batch, though the dimensions (i.e. duration of each trial segment) will be ragged

    return recruit_graphs

# calculate MI functional graphs and associated binned recruitment graphs for a single batch update of 30 trials
def bin_batch_MI_graphs(w,spikes,true_y,bin,sliding_window_bins,threshold,e_only,positive_only):
    # calculate 8 FNs (2 coherence levels x 4 types of connectivity)
    if e_only:
        n_units = e_end
    else:
        n_units = i_end
    trial_dur = np.shape(true_y)[1]
    fn_coh0 = np.empty([n_units,0])
    fn_coh1 = np.empty([n_units,0])
    binned_spikes_coh0 = np.empty([n_units,0])
    trialends_coh0 = []
    binned_spikes_coh1 = np.empty([n_units,0])
    trialends_coh1 = []
    for trial in range(np.shape(true_y)[0]): # each of 30 trials per batch update
        spikes_trial = np.transpose(spikes[trial])
        trial_y = np.squeeze(true_y[trial])
        # separate spikes according to coherence level
        coh_0_idx = np.squeeze(np.where(trial_y==0))
        coh_1_idx = np.squeeze(np.where(trial_y==1))

        if np.size(coh_0_idx)>0:
            # if the start of a new coherence level happens in the middle of the trial
            if not (0 in coh_0_idx):
                # remove the first 50 ms
                coh_0_idx = coh_0_idx[50:]
            z_coh0 = spikes_trial[:,coh_0_idx]
            # bin spikes into 10 ms, discarding trailing ms
            trial_n_bins = int(np.math.floor(np.shape(z_coh0)[1]/bin)) # (count of bins for this coherence level's spikes)
            trial_binned_z = np.zeros([n_units,trial_n_bins]) # holder for this trial's binned spikes
            for t in range(trial_n_bins): # for each 10-ms bin
                # the only spikes we are looking at (within this 10-ms bin)
                z_in_bin = z_coh0[:,t*bin:(t+1)*bin-1]
                for j in range(n_units): # for each neuron
                    if (1 in z_in_bin[j,:]):
                        # if spiked at all, put in a 1
                        trial_binned_z[j,t] = 1
            binned_spikes_coh0 = np.hstack([binned_spikes_coh0,trial_binned_z])
            # get all the spikes for each coherence level strung together
            trialends_coh0.append(np.shape(binned_spikes_coh0)[1]-1)
            # keep sight of new trial_end_indices relative to newly binned spikes
        if np.size(coh_1_idx)>0:
            if not (0 in coh_1_idx):
                coh_1_idx = coh_1_idx[50:]
            z_coh1 = spikes_trial[:,coh_1_idx]
            trial_n_bins = int(np.math.floor(np.shape(z_coh1)[1]/bin))
            trial_binned_z = np.zeros([n_units,trial_n_bins])
            for t in range(trial_n_bins):
                z_in_bin = z_coh1[:,t*bin:(t+1)*bin-1]
                for j in range(n_units):
                    if (1 in z_in_bin[j,:]):
                        trial_binned_z[j,t] = 1
            binned_spikes_coh1 = np.hstack([binned_spikes_coh1,trial_binned_z])
            trialends_coh1.append(np.shape(binned_spikes_coh1)[1]-1)

    # pipe into confMI calculation
    fn_coh0 = simple_confMI(binned_spikes_coh0,trialends_coh0,positive_only)
    fn_coh1 = simple_confMI(binned_spikes_coh1,trialends_coh1,positive_only)

    # make recruitment graphs
    rn_coh0 = batch_recruitment_graphs(w,fn_coh0,binned_spikes_coh0,trialends_coh0,threshold)
    rn_coh1 = batch_recruitment_graphs(w,fn_coh1,binned_spikes_coh1,trialends_coh1,threshold)

    return [[fn_coh0,fn_coh1],[rn_coh0,rn_coh1]]

def generate_all_recruitment_graphs(experiment_string, overwrite=False, bin=10, sliding_window_bins=False, threshold=0.25, e_only=False, positive_only=False):
    # experiment_string is the data we want to turn into recruitment graphs
    # do not overwrite already-saved files that contain generated networks
    # bin functional networks into 10ms (as 'consecutive' bins)
    # for the sake of efficiency, these are discrete bins rather than sliding window through each ms
    # threshold = 0.25 means we take just the top quartile of FN weights to calculate recruitment graphs
    # generate separately for e-e, e-i, i-e, and i-i units (if e_only=True, only do e-e)
    # positive_only=False means we DO include negative confMI values (negative correlations)
    # previously we had always removed those, but now we'll try to make sense of negative correlations as we go

    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # networks will be saved as npz files (each containing multiple arrays), so the same names as data_files

    recruit_savepath = os.path.join(savepath,"recruitment_graphs_bin10_quartile")
    if not os.path.isdir(recruit_savepath):
        os.makedirs(recruit_savepath)

    # we desire to save new MI functional graphs as well
    MI_savepath = os.path.join(savepath,"MI_graphs_bin10")
    if not os.path.isdir(MI_savepath):
        os.makedirs(MI_savepath)

    for xdir in experiments:
        exp_path = xdir[-9:-1]
        # check if MI and recruitment graph folders have already been generated
        if not os.path.isdir(os.path.join(recruit_savepath,exp_path)):
            os.makedirs(os.path.join(recruit_savepath,exp_path))
        if not os.path.isdir(os.path.join(MI_savepath,exp_path)):
            os.makedirs(os.path.join(MI_savepath,exp_path))
        if not e_only:
            # for each batch update, there should be 2 functional networks (1 for each coherence level)
            for file_idx in range(np.size(data_files)):
                filepath = os.path.join(data_dir, xdir, 'npz-data', data_files[file_idx])
                # check if we haven't generated FNs already
                if not os.path.isfile(os.path.join(MI_savepath,exp_path,data_files[file_idx])) or overwrite:
                    data = np.load(filepath)
                    spikes = data['spikes']
                    true_y = data['true_y']
                    w = data['tv1.postweights']
                    # generate MI and recruitment graphs from spikes for each coherence level
                    fns_coh0 = []
                    fns_coh1 = []
                    rns_coh0 = []
                    rns_coh1 = []
                    for batch in range(np.shape(true_y)[0]): # each file contains 100 batch updates
                    # each batch update has 30 trials
                    # those spikes and labels are passed to generate graphs batch-wise
                    # each w is actually a postweight, so corresponds to the next batch
                        if batch==0 and file_idx==0:
                        # at the very beginning of the experiment, the naive network is loaded in
                            w_naive = np.load(os.path.join(data_dir, xdir, 'npz-data', 'main_preweights.npy'))
                            batch_w = w_naive
                        elif batch==0 and file_idx!=0:
                        # if we are at the starting batch of a file (but not the starting file of the experiment),
                        # load in the previous file's final (99th) batch's postweights
                            prev_data = np.load(os.path.join(data_dir, xdir, 'npz-data', data_files[file_idx-1]))
                            batch_w = prev_data['tv1.postweights'][99]
                        elif batch!=0:
                        # not at the starting (0th) batch (of any file), so just use the previous batch's postweights
                            batch_w = w[batch-1]
                        # generate batch-wise MI and recruitment graphs
                        [batch_fns, batch_rns] = bin_batch_MI_graphs(batch_w,spikes[batch],true_y[batch],bin,sliding_window_bins,threshold,e_only,positive_only)
                        # batch_fns is sized [2, 300, 300]
                        # batch_rns is sized [2, 408, 300, 300]
                        # aggregate functional networks to save
                        fns_coh0.append(batch_fns[0])
                        fns_coh1.append(batch_fns[1])
                        # aggregate recruitment networks to save
                        rns_coh0_ee.append(batch_rns[0])
                        rns_coh0_ei.append(batch_rns[1])
                    # do not save in separate directories, instead save all these in the same files by variable name
                    # saving convention is same as npz data files (save as 1-10.npz for example)
                    # for example, fns_coh0 is sized [100 batch updates, 300 pre units, 300 post units]
                    # and rns_coh0 is sized [100 batch updates, # trial segments, # timesteps, 300 pre units, 300 post units]
                    # we will separate by connection type (ee, ei, ie, ee) in further analyses
                    np.savez(os.path.join(MI_savepath,exp_path,data_files[file_idx]),coh0=fns_coh0,coh1=fns_coh1)
                    np.savez(os.path.join(recruitment_savepath,exp_path,data_files[file_idx]),coh0=rns_coh0,coh1=rns_coh1)

"""
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
"""

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
