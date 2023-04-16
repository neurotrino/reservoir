# MI.py
# calculates confluent mutual information based on spikes from .npz files

import sys
import time

import numpy as np
import scipy
from scipy.sparse import load_npz
import glob
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

dt = 1
experiment = "ccd_save_spikes"
run_dur = 4080
batch_size = 10


def simplest_confMI(raster, correct_signs=True, lag=1):
    post_raster = raster[:,lag:]
    raster = raster[:,:-lag]
    post_raster = np.logical_or(raster,post_raster)
    neurons = raster.shape[0]
    mat = np.zeros([neurons, neurons])
    for pre in range(neurons):
        for post in range(neurons):
            if pre != post:
                mat[pre, post] = compute_MI(
                    raster[pre, :], post_raster[post, :]
                )
    if correct_signs:
        signed_graph = signed_MI(mat, raster)
        e_graph = pos(signed_graph[:241,:])
        i_graph = neg(signed_graph[241:,:])
        composite_graph = np.concatenate((e_graph,i_graph),axis=0)
        return composite_graph
    else:
        return mat

def simplest_asym_confMI(pre_raster,raster,correct_signs=False,lag=1):
    post_raster = raster[:,lag:]
    pre_raster = pre_raster[:,:-lag]
    # spike at the same timestep or one after counts as confluence
    post_raster = np.logical_or(raster,post_raster)
    inputs = pre_raster.shape[0]
    neurons = post_raster.shape[0]
    mat = np.zeros([inputs,neurons])
    for pre in range(inputs):
        for post in range(neurons):
            mat[pre, post] = compute_MI(
                pre_raster[pre,:], post_raster[post,:]
            )
    if correct_signs:
        signed_graph = signed_MI(mat, raster)
        e_graph = pos(signed_graph[:241,:])
        i_graph = neg(signed_graph[241:,:])
        composite_graph = np.concatenate((e_graph,i_graph),axis=0)
        return composite_graph
    else:
        return mat

def simple_confMI(raster, trial_ends, positive_only, lag=1):
    post_raster = raster[:, lag:]
    raster = raster[:, :-lag]
    post_raster = np.logical_or(raster, post_raster)  # confluence

    # remove trial ends
    raster = remove_trial_ends(raster, trial_ends[:-1])
    post_raster = remove_trial_ends(post_raster, trial_ends[:-1])

    # compute the actual MI
    neurons = raster.shape[0]
    mat = np.zeros([neurons, neurons])
    for pre in range(neurons):
        for post in range(neurons):
            if pre != post:
                mat[pre, post] = compute_MI(
                    raster[pre, :], post_raster[post, :]
                )
    if positive_only:
        signed_graph = signed_MI(mat, raster)
        pos_graph = pos(signed_graph)
        return pos_graph
    else:
        return mat


def signed_MI(graph, raster):
    neurons = np.shape(graph)[0]
    signed_graph = np.copy(graph)
    for pre in range(0, neurons):
        for post in range((pre + 1), neurons):
            corr_mat = np.corrcoef(raster[pre, :], raster[post, :])
            factor = np.sign(corr_mat[0, 1])
            if ~np.isnan(factor):
                signed_graph[pre, post] *= factor
                signed_graph[post, pre] *= factor
    return signed_graph


def pos(graph):
    # takes signed_MI MI graph, returns positive version
    pos_graph = np.copy(graph)
    pos_graph[graph < 0] = 0
    return pos_graph

def neg(graph):
    # takes signed_MI MI graph, returns negative version
    neg_graph = np.copy(graph)
    neg_graph[graph > 0] = 0
    return neg_graph


def construct_MI_mat(
    raster, trial_ends=[], MI_type="confluent", temporal_FN_union=False
):
    """
    construct a mutual information matrix from the given spike trains,
    excluding transitions over the given trial end indices (the last
    timebin of each trial.

    confluent, consecutive, or simultaneous mutual information

    alternatively, instead of filling in the matrix with MI values, fill
    it in with the union of temporal FNs (1 where a spike pair is
    present at all in the given raster, 0 where it is not)
    """
    assert MI_type in {"confluent", "consecutive", "simultaneous"}
    lag = 1  # currently the trial_end logic won't work with larger lags

    # lag the spike trains accordingly
    if MI_type == "confluent":
        post_raster = raster[:, lag:]
        raster = raster[:, :-lag]
        post_raster = np.logical_or(raster, post_raster)  # confluence
    elif MI_type == "consecutive":
        post_raster = raster[:, lag:]  # nothing more
        raster = raster[:, :-lag]
    elif MI_type == "simultaneous":
        post_raster = raster  # fine to not copy

    # remove trial ends if necessary (they don't affect consecutive MI)
    if MI_type == "confluent" or MI_type == "consecutive":
        raster = remove_trial_ends(raster, trial_ends)
        post_raster = remove_trial_ends(post_raster, trial_ends)

    # compute the actual MI
    neurons = raster.shape[0]
    mat = np.zeros([neurons, neurons])
    for pre in range(neurons):
        for post in range(neurons):
            if pre != post:
                if temporal_FN_union:
                    mat[pre, post] = potential_edge(
                        raster[pre, :], post_raster[post, :]
                    )
                else:
                    mat[pre, post] = compute_MI(
                        raster[pre, :], post_raster[post, :]
                    )

    return mat


def compute_MI(train_1, train_2):
    """
    compute the mutual information between two spike trains
    (if one has been lagged or made confluent appropriately, this
    will compute consecutive, confluent, etc. MI)
    """
    # marginal probs
    p_pre_1 = np.mean(train_1)
    p_post_1 = np.mean(train_2)
    p_pre_0 = 1 - p_pre_1
    p_post_0 = 1 - p_post_1

    # joint variables
    both_1 = np.logical_and(train_1, train_2)
    pre_0_post_1 = np.logical_and(np.logical_not(train_1), train_2)
    pre_1_post_0 = np.logical_and(train_1, np.logical_not(train_2))
    both_0 = np.logical_and(np.logical_not(train_1), np.logical_not(train_2))

    # and compute probabilities
    p_both_1 = np.mean(both_1)
    p_pre_0_post_1 = np.mean(pre_0_post_1)
    p_pre_1_post_0 = np.mean(pre_1_post_0)
    p_both_0 = np.mean(both_0)

    # sum up MI, making sure not to take the log of zero
    MI = 0
    if p_both_1 > 0:
        MI += p_both_1 * np.log2(p_both_1 / (p_pre_1 * p_post_1))
    if p_both_0 > 0:
        MI += p_both_0 * np.log2(p_both_0 / (p_pre_0 * p_post_0))
    if p_pre_0_post_1 > 0:
        MI += p_pre_0_post_1 * np.log2(p_pre_0_post_1 / (p_pre_0 * p_post_1))
    if p_pre_1_post_0 > 0:
        MI += p_pre_1_post_0 * np.log2(p_pre_1_post_0 / (p_pre_1 * p_post_0))

    return MI


def potential_edge(train_1, train_2):
    """
    see whether there is a potential edge expressed in the given spike trains
    that is, if they ever spike together
    """
    return np.any(np.logical_and(train_1, train_2))


# some refactoring by Hal
def remove_trial_ends(raster, trial_ends):
    """
    return the NxT with the time indices corresponding to the
    last timebins of each trial removed.

    trial_ends is an array-like of integer indices to be removed
    so the length of the returned spike train is len(spike_train) - len(trial_ends)
    assuming all trial end indices are unique
    """
    trial_ends_bin = np.ones(raster.shape[1], dtype=bool)
    trial_ends_bin[trial_ends] = False

    return raster[:, trial_ends_bin]


"""
def main(experiment,dt):
    # loop through and load all experiment npz files
    dir = '/home/macleanlab/experiments/' + experiment + '/npz-data/'
    files = glob.glob(dir+'*.npz')
    files_sorted = sorted(files, key=lambda x: int(x.split('-')[0]))
    mi_graphs = []
    for f in files_sorted:
        data = np.load(dir + files_sorted[f])
        spikes = data['spikes']
        for batch in spikes:
            batch_spikes = np.reshape(spikes[batch], [run_dur*np.shape(spikes[batch])[0],np.shape(spikes[batch])[2]])
            batch_raster = np.transpose(batch_spikes)
            batch_mi_graph = generate_mi_graph(batch_raster,dt)
    mi_graphs.append(batch_mi_graph)
    # well, then we can do anything we want
    savedir = '/home/macleanlab/experiments/' + experiment + '/analysis/'
    np.save(savedir + 'mi.npy', mi_graphs)

    # for now using postweights but ultimately we'd like to have preweights
    # because those are the weights that actually led to the activity in these trials
    #w_rec = data['tv1.postweights']
    # note that we are using pre-update weights, since these are the ones actually generating the spikes
    #recruitment_graph = intersect_functional_and_synaptic(functional_graph,w_rec)

def debug(experiment,dt):
    dir = '/home/macleanlab/experiments/' + experiment + '/npz-data/'
    begin_file = dir + '1-10.npz'
    data = np.load(begin_file)
    spikes = data['spikes']
    batch = 0
    batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
    raster = np.transpose(batch_spikes)
    MI_graph = confMI_mat(raster)
    signed_graph = signed_MI(MI_graph,raster)
    pos_graph = pos(signed_graph)
    reexpress_graph = reexpress_param(pos_graph)
    background_graph = background(reexpress_graph)
    return background_graph

def mi_beginning_end(experiment,dt):
    dir = '/home/macleanlab/experiments/' + experiment + '/npz-data/'
    begin_file = dir + '1-10.npz'
    end_file = dir + '111-120.npz'
    # save mi graphs from beginning 10 epochs and ending 10 epochs in separate files

    data = np.load(begin_file)
    mi_graphs = []
    spikes = data['spikes']
    for batch in range(0,np.shape(spikes)[0]):
        batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
        batch_raster = np.transpose(batch_spikes)
        batch_mi_graph = generate_mi_graph(batch_raster,dt)
        mi_graphs.append(batch_mi_graph)
    save_dir = '/home/macleanlab/experiments/' + experiment + '/analysis/'
    np.save(save_dir + 'start_mi.npy', mi_graphs)

    data = np.load(end_file)
    mi_graphs = []
    spikes = data['spikes']
    for batch in spikes:
        batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
        batch_raster = np.transpose(batch_spikes)
        batch_mi_graph = generate_mi_graph(batch_raster,dt)
        mi_graphs.append(batch_mi_graph)
    end_save_file = '/home/macleanlab/experiments/' + experiment + '/analysis/'
    np.save(save_dir + 'end_mi.npy', mi_graphs)

def ccd_skeleton(batch_raster, batch, coh_lvl):
    # for every batch (in each npz file there are 100)
    # collapsing epoch and batch
    #dir = '/home/macleanlab/experiments/' + experiment + '/npz-data/'
    #data_file = dir + '11-20.npz'
    #spikes = np.load(data_file)['spikes']

    # handle coherences for whole epoch first
    coh_data_file = '/home/macleanlab/CNN_outputs/coherences_mixed_limlifetime_abs.npz'
    coherences = load_npz(coh_data_file)
    y = np.array(coherences.todense())
    #y = np.array(coherences.todense().reshape((-1, run_dur)))[:, :, None] # shaped as [600, run_dur]
    y_epoch = np.tile(y[0:10],[10,1])
    # since we have 10 batches x 10 trials (40800 total duration for each batch)
    # we actually only need the first 10 out of 60 for each epoch
    # and we repeat epoch # of times (for each npz file, only need 10 repeats)
    # now their indices should correspond precisely with batch_raster

    #for batch in range(0,99):
    #batch = 99
    #batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
    #batch_raster = np.transpose(batch_spikes)
    batch_y = y_epoch[batch]

    # determine timepts where coh = 100 (1) and coh = 15 (0)
    if coh_lvl == 100:
        indices = np.argwhere(batch_y == 1)
    else:
        indices = np.argwhere(batch_y == 0)
    coh_lvl_raster = np.squeeze(batch_raster[:,indices])
    mi_graph = generate_mi_graph_ccd(coh_lvl_raster, indices)
    return mi_graph

def generate_mi_graph_ccd(raster, indices):
    MI_graph = confMI_mat_ccd(raster, indices)
    signed_graph = signed_MI(MI_graph,raster)
    pos_graph = pos(signed_graph)
    reexpress_graph = reexpress_param(pos_graph)
    background_graph = background(reexpress_graph)
    residual_graph = residual(background_graph,MI_graph)
    normed_MI_graph = normed_residual(residual_graph)
    return normed_MI_graph
"""

"""
def generate_mi_graph(raster, trial_ends=[], MI_type='confluent',
        temporal=False):
    # add time logging
    t = time.time()
    MI_graph = construct_MI_mat(raster, trial_ends, MI_type, temporal)
    print(f'generating MI graph: {time.time() - t}')
    t = time.time()
    signed_graph = signed_MI(MI_graph,raster)
    print(f'generating signed graph: {time.time() - t}')
    t = time.time()
    pos_graph = pos(signed_graph)
    print(f'generating pos graph: {time.time() - t}')
    t = time.time()
    reexpress_graph = reexpress_param(pos_graph)
    print(f'generating reexpressed graph: {time.time() - t}')
    t = time.time()
    background_graph = background(reexpress_graph)
    print(f'generating background graph: {time.time() - t}')
    t = time.time()
    residual_graph = residual(background_graph,MI_graph)
    print(f'generating residual graph: {time.time() - t}')
    t = time.time()
    normed_MI_graph = normed_residual(residual_graph)
    print(f'generating normed graph: {time.time() - t}')
    # for debugging
    #return MI_graph, reexpress_graph, background_graph, residual_graph, normed_MI_graph
    #return signed_graph
    return normed_MI_graph

def confMI_mat_ccd(raster, indices): # using all spikes during a particular coherence level for a batch's trials
    lag = 1
    alpha = 0
    neurons = np.shape(raster)[0]
    trial_ends = np.arange(run_dur-1, run_dur*10, run_dur) # in 40800 index
    # transform trial ends into new indices
    trial_ends_newidx = np.where(np.isin(indices,trial_ends))[0]
    # to grab the rest of the discontinuities (where a coherence level ends mid-trial),
    # check adjacent indices for whether they are increments in value
    level_end = []
    for i in range(0,np.size(indices)-1):
        if indices[i]+1 != indices[i+1]:
            level_end.append(i)
    trial_ends = np.union1d(trial_ends_newidx,level_end)
    mat = np.zeros([neurons,neurons])
    for pre in range(0,neurons):
        for post in range(0,neurons):
            if pre != post:
                mat[pre,post] = confMI(raster[pre,:],raster[post,:], lag, alpha, trial_ends)
    return mat

def confMI_mat_sinusoid(raster, slow=False, trial_ends=[]):
    lag = 1
    alpha = 0
    neurons = np.shape(raster)[0]
    #trial_ends = np.arange(run_dur-1, np.shape(raster)[1],run_dur)
    trial_ends = np.array(trial_ends)
    mat = np.zeros([neurons,neurons])
    if slow:
        for pre in range(0,neurons):
            for post in range(0,neurons):
                if pre != post:
                    mat[pre,post] = confMI(raster[pre,:],raster[post,:],lag,alpha,trial_ends)
    else:
        with np.nditer(mat, op_flags=['writeonly'], flags=['multi_index']) as it:
            for entry in it:
                if it.multi_index[0] != it.multi_index[1]:
                    entry[...] = confMI_fast(raster[it.multi_index[0],:], \
                            raster[it.multi_index[1],:], trial_ends)

    mat = np.array(mat)
    np.fill_diagonal(mat, 0.0)
    return mat
"""

"""
def confMI(train_1,train_2,lag,alpha,trial_ends):
    MI = 0
    states = [0,1]
    #mat[post,pre] = confMI(raster[pre,:],raster[post,:],lag,alpha)
    # meaning train_1 is definitely for pre and train_2 is for post
    # former matrix convention was j,i rather than i,j is all

    for i in range(0,np.size(states)):
        i_inds = np.argwhere(train_1 == states[i])
        p_i = np.shape(i_inds)[0]/np.shape(train_1)[0]
        if np.shape(i_inds)[0] > 0:
            for j in range(0,np.size(states)):
                j_inds = np.argwhere(train_2 == states[j])
                #print(j_inds.shape)
                j_inds_lagged = j_inds - lag
                if np.shape(j_inds)[0] > 0:
                    j_inds_lagged = j_inds_lagged[j_inds_lagged >= 0]
                    # none of the lagged indices (t-1) can be equal to the end of a trial,
                    # since that steps over a causal discontinuity
                    j_inds_lagged = j_inds_lagged[~np.isin(j_inds_lagged,trial_ends)]
                    # bug? this does the OR over 0 values as well...
                    # meaning p(j=0) + p(j=1) > 1!
                    if j == 1:
                        j_inds = np.union1d(j_inds,j_inds_lagged)
                    else:
                        j_inds = np.intersect1d(j_inds,j_inds_lagged)

                    if np.shape(j_inds)[0] < np.shape(train_2)[0]:
                    # because if they are equal in size, we will have p > 1 when subtracting lag
                        p_j = np.shape(j_inds)[0]/(np.shape(train_2)[0]-lag)
                        # verify that p_j is larger than it should be
                        print(f'i={i}, j={j}, p_j={p_j}')
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/(np.shape(train_1)[0]-lag)
                    else:
                        p_j = np.shape(j_inds)[0]/np.shape(train_2)[0]
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/np.shape(train_1)[0]
                    if alpha > 0:
                        MI = MI + alpha + (1-alpha) * p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                    elif p_i_and_j > 0:
                        #print(p_i)
                        #print(p_j)
                        #print(p_i_and_j)
                        update = p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                        #print(f'{i} {j} {update}')
                        MI += update
    return MI
"""

"""
def confMI_fast(train_1, train_2, trial_ends):
    '''
    compute confluent mutual information between two spike trains, with
    trial_ends listing indices where a trial ended (causal discontinuities)
    '''
    # the trial adjustment will need to be changed for a larger lag
    # since as-is it will only work properly with a lag of 1
    # so set it inside this function
    lag = 1

    # j-hat
    train_2_lagged = train_2[lag:]
    train_2 = train_2[:-lag]
    j_hat = np.logical_or(train_2, train_2_lagged)

    # marginal probs
    p_pre_1 = np.mean(train_1)
    p_post_1 = np.mean(j_hat)
    p_pre_0 = 1 - p_pre_1
    p_post_0 = 1 - p_post_1

    # for the overlaps to work
    train_1 = train_1[:-lag]

    # joint variables -- compute joint probabilities from these
    # after taking out the trial ends
    both_1 = np.logical_and(train_1, j_hat)
    pre_0_post_1 = np.logical_and(np.logical_not(train_1), j_hat)
    pre_1_post_0 = np.logical_and(train_1, np.logical_not(j_hat))
    both_0 = np.logical_and(np.logical_not(train_1), np.logical_not(j_hat))

    # find trial_end indices
    trial_ends = trial_ends[trial_ends < len(train_1)] # if one falls in the last lag bins, get rid of it
    print(trial_ends) #???
    trial_ends_bin = np.ones(len(train_1), dtype=bool)
    trial_ends_bin[trial_ends] = False

    # now get rid of them
    # the times are aligned with train_1, so at 0 indices, the t+1 in j_hat
    # is across the causal discontinuity, this will take them out
    # could update to make [lag] indices up to the trial ends 0 indices, to
    # take all of them out for longer lags
    # (that would involve changing the lines above)
    both_1 = both_1[trial_ends_bin]
    pre_0_post_1 = pre_0_post_1[trial_ends_bin]
    pre_1_post_0 = pre_1_post_0[trial_ends_bin]
    both_0 = both_0[trial_ends_bin]

    # and compute probabilities
    p_both_1 = np.mean(both_1)
    p_pre_0_post_1 = np.mean(pre_0_post_1)
    p_pre_1_post_0 = np.mean(pre_1_post_0)
    p_both_0 = np.mean(both_0)

    # do updates -- check before dividing by zero!
    MI = 0
    if p_both_1 > 0:
        MI += p_both_1 * np.log2(p_both_1 / (p_pre_1 * p_post_1))
    if p_both_0 > 0:
        MI += p_both_0 * np.log2(p_both_0 / (p_pre_0 * p_post_0))
    if p_pre_0_post_1 > 0:
        MI += p_pre_0_post_1 * np.log2(p_pre_0_post_1 / (p_pre_0 * p_post_1))
    if p_pre_1_post_0 > 0:
        MI += p_pre_1_post_0 * np.log2(p_pre_1_post_0 / (p_pre_1 * p_post_0))

    return MI

"""
"""
def null_confMI(train_1,train_2,lag,alpha):
    MI = 0
    states = [0,1]
    for i in range(0,np.size(states)):
        # find all indices (timepoints) where presynaptic neuron i is in states[i]
        i_inds = np.argwhere(train_1 == states[i])
        p_i = np.shape(i_inds)[0]/np.shape(train_1)[0]
        if np.shape(i_inds)[0] > 0:
            for j in range(0,np.size(states)):
                j_inds = []
                #trial_ends = np.arange(run_dur-1,np.shape(train_1)[0],run_dur)
                trial_ends = np.shape(train_1)[0]
                if np.shape(np.argwhere(train_2 == states[j]))[0] > 0: # if neuron j is ever in states[j], proceed
                    for idx in i_inds:
                        # for all t where neuron i is in states[i]
                        if not(idx in trial_ends): # if we are not at the end of a trial (a discontinuity)
                            if (train_2[idx] == states[j]) or (train_2[idx+lag] == states[j]):
                                # check if postsynaptic neuron j was in states[j] at time t or t+1
                                j_inds.append(idx) # if so, add this t index to the time points that neuron j is confluent with i
                        else: # if we are at the end of a trial, only check time point t
                            if (train_2[idx] == states[j]):
                                j_inds.append(idx)
                    p_j = np.shape(j_inds)[0]/(np.shape(train_2)[0])
                    numer = np.shape(np.intersect1d(i_inds,j_inds))[0]
                    denom = (np.shape(train_1)[0])
                    p_i_and_j = numer/denom
                    if alpha > 0:
                        MI = MI + alpha + (1-alpha) * p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                    elif p_i_and_j > 0:
                        MI += p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
    return MI
"""
"""

def reexpress_param(graph):
    #takes pos graph and returns redist graph
    steps = 100
    data = graph[:]
    #data = np.copy(graph)
    #data = data[find(x->x>0,data)]
    data = data[data > 0]
    upper_exp=10
    lower_exp=0.00000000000001
    upper_skew = scipy.stats.skew(data**upper_exp)
    lower_skew = scipy.stats.skew(data**lower_exp)
    for i in range(0,steps):
        new_exp = (upper_exp + lower_exp)/2
        new_skew = scipy.stats.skew(data**new_exp)
        if new_skew<0:
            lower_exp=new_exp;
            lower_skew=new_skew;
        else:
            upper_exp=new_exp;
            upper_skew=new_skew;
    if np.abs(upper_skew)<np.abs(lower_skew):
        reexpress = upper_exp
    else:
        reexpress = lower_exp
    reexpress_graph = np.copy(graph)
    reexpress_graph = reexpress_graph**reexpress
    return reexpress_graph

def background(graph):
    #takes reexpress graph returns background graph
    background = np.copy(graph)
    neurons = np.arange(0,np.shape(graph)[0])
    for pre in neurons:
        for post in neurons:
            if pre != post:
                #background[pre,post] = np.mean(graph[pre,neurons[neurons != post]])*np.mean(graph[neurons[np.logical_and((neurons != post), (neurons != pre))]])
                # Hal's edit: second term was incorrect
                background[pre,post] = np.mean(graph[pre,neurons[neurons != post]]) * \
                        np.mean(graph[neurons[neurons != pre], post])
    return background

def residual(background_graph,graph):
    residual = np.copy(graph)
    neurons = np.shape(graph)[0]
    lm = LinearRegression()
    #b,m = linreg(background_graph[:],graph[:])
    #model = lm.fit(background_graph[:],graph[:])
    # Hal's edit: above works in Matlab but not Python
    model = lm.fit(background_graph.reshape(-1, 1), graph.reshape(-1))
    b = model.intercept_
    m = model.coef_
    residual = graph - (m*background_graph + b)
    return residual

def normed_residual(graph):
    norm_residual = np.copy(graph)
    neurons = np.arange(0,np.shape(graph)[0])
    for pre in neurons:
        for post in neurons:
            if pre != post:
                norm_residual[pre,post] = np.std(graph[pre,neurons[neurons != post]]) * \
                        np.std(graph[neurons[neurons != pre], post])
                #norm_residual[pre,post] = np.std(graph[pre,neurons[neurons != post]]) * \
                        #np.std(graph[neurons[np.logical_and(neurons != pre, neurons != post)]])
    cutoff = np.median(norm_residual)
    neurons = np.shape(graph)[0]
    norm_residual = 1/(np.sqrt(np.maximum(norm_residual,np.ones([neurons,neurons])*cutoff)))
    return norm_residual*graph

def binary_raster_gen(spikes,dt):
    bins = np.arange(0,run_total,dt)
    raster = np.zeros(np.shape(spikes)[0],length(bins)-1)
    for i in range(0:np.shape(spikes)[0]):
        discrete_train = unique(round.(spike_set[i]/dt))
        discrete_train = discrete_train[discrete_train.>0]
        for j = 1:length(discrete_train)
            raster[i,Int(discrete_train[j])] = 1
    return raster
"""
