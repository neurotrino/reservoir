#MI.py
#calculates confluent mutual information based on spikes from .npz files

import sys

import numpy as np
import scipy
from scipy.sparse import load_npz
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

dt = 1
experiment = 'ccd_save_spikes'
run_dur = 4080
batch_size = 10

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

    """
    # for now using postweights but ultimately we'd like to have preweights
    # because those are the weights that actually led to the activity in these trials
    w_rec = data['tv1.postweights']
    # note that we are using pre-update weights, since these are the ones actually generating the spikes
    recruitment_graph = intersect_functional_and_synaptic(functional_graph,w_rec)
    """

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

def ccd_skeleton(experiment):
    # for every batch (in each npz file there are 100)
    # collapsing epoch and batch
    dir = '/home/macleanlab/experiments/' + experiment + '/npz-data/'
    data_file = dir + '11-20.npz'
    spikes = np.load(data_file)['spikes']

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
    batch = 99
    batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
    batch_raster = np.transpose(batch_spikes)
    batch_y = y_epoch[batch]

    # determine timepts where coh = 100 (1) and coh = 15 (0)
    index_100 = np.argwhere(batch_y == 1)
    '''trial_end_100 = []
    for i in range(0, np.size(index_100)-1):
        if index_100[i] + 1 != index_100[i+1]:
            trial_end_100.append(index_100[i])'''
    index_15 = np.argwhere(batch_y==0)
    '''trial_end_15 = []
    for i in range(0, np.size(index_15)-1):
        if index_15[i] + 1 != index_100[i+1]:
            trial_end_15.append(index_15[i])
    raster_15 = batch_raster[:,index_15]
    raster_100 = batch_raster[:,index_100]'''

    mi_graph_low = generate_mi_graph_ccd(batch_raster, index_15)
    mi_graph_high = generate_mi_graph_ccd(batch_raster, index_100)

def generate_mi_graph_ccd(raster, coh_data):
    MI_graph = confMI_mat_ccd(raster, indices)
    signed_graph = signed_MI(MI_graph,raster)
    pos_graph = pos(signed_graph)
    reexpress_graph = reexpress_param(pos_graph)
    background_graph = background(reexpress_graph)
    residual_graph = residual(background_graph,MI_graph)
    normed_MI_graph = normed_residual(residual_graph)
    return normed_MI_graph

def generate_mi_graph_generic(raster,dt):
    #raster = binary_raster_gen(spikes,dt)
    MI_graph = confMI_mat(raster)
    signed_graph = signed_MI(MI_graph,raster)
    pos_graph = pos(signed_graph)
    reexpress_graph = reexpress_param(pos_graph)
    background_graph = background(reexpress_graph)
    residual_graph = residual(background_graph,MI_graph)
    normed_MI_graph = normed_residual(residual_graph)
    return normed_MI_graph

"""
def test_confMI_methods():
    file = '/home/macleanlab/experiments/sinusoid_save_spikes/npz-data/101-110.npz'
    data = np.load(file)
    spikes = data['spikes']
    raster = np.transpose(spikes[0][0])
    toy_raster = np.array([[0,1,1,0,0,1,0],[0,0,0,0,1,1,1],[0,1,1,0,0,0,1],[0,0,1,1,0,0,0]]) # four neurons, seven timesteps
    former_method_mat = formerly_confMI_mat(toy_raster)
    new_method_mat = confMI_mat(toy_raster)
    if former_method_mat == new_method_mat:
        print('methods match')
    else:
        print('methods do not match')
"""

def confMI_mat_ccd(raster, indices): # using all spikes during a particular coherence level for a batch's trials
    lag = 1
    alpha = 0
    neurons = np.shape(raster)[0]
    trial_ends = np.arange(run_dur-1, np.shape(raster)[0],run_dur)
    mat = np.zeros([neurons,neurons])
    for pre in range(0,neurons):
        for post in range(0,neurons):
            if pre != post:
                mat[pre,post] = confMI_ccd(raster[pre,:],raster[post,:],lag,alpha,indices)
    return mat

def confMI_mat_sinusoid(raster):
    lag = 1
    alpha = 0
    neurons = np.shape(raster)[0]
    trial_ends = np.arange(run_dur-1, np.shape(raster)[0],run_dur)
    mat = np.zeros([neurons,neurons])
    for pre in range(0,neurons):
        for post in range(0,neurons):
            if pre != post:
                mat[pre,post] = confMI(raster[pre,:],raster[post,:],lag,alpha)
    return mat

def confMI_ccd(train_1,train_2,lag,alpha,coh_indices):
    MI = 0
    states = [0,1]

    for i in range(0,np.size(states)):
        i_inds = np.argwhere(train_1[coh_indices] == states[i])
        p_i = np.shape(i_inds)[0]/np.shape(train_1)[0]
        if np.shape(i_inds)[0] > 0:
            for j in range(0,np.size(states)):
                j_inds = np.argwhere(train_2[coh_indices] == states[j])
                j_inds_lagged = j_inds - lag
                if np.shape(j_inds)[0] > 0:
                    j_inds_lagged = j_inds_lagged[j_inds_lagged >= 0]
                    # none of the lagged indices (t-1) can be equal to the end of a trial,
                    # since that steps over a causal discontinuity
                    j_inds_lagged = j_inds_lagged[~np.isin(j_inds_lagged,trial_ends)]
                    j_inds = np.union1d(j_inds,j_inds_lagged)
                    if np.shape(j_inds)[0] < np.shape(train_2)[0]:
                    # because if they are equal in size, we will have p > 1 when subtracting lag
                        p_j = np.shape(j_inds)[0]/(np.shape(train_2)[0]-lag)
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/(np.shape(train_1)[0]-lag)
                    else:
                        p_j = np.shape(j_inds)[0]/np.shape(train_2)[0]
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/np.shape(train_1)[0]
                    if alpha > 0:
                        MI = MI + alpha + (1-alpha) * p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                    elif p_i_and_j > 0:
                        MI += p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
    return MI


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
                j_inds_lagged = j_inds - lag
                if np.shape(j_inds)[0] > 0:
                    j_inds_lagged = j_inds_lagged[j_inds_lagged >= 0]
                    # none of the lagged indices (t-1) can be equal to the end of a trial,
                    # since that steps over a causal discontinuity
                    j_inds_lagged = j_inds_lagged[~np.isin(j_inds_lagged,trial_ends)]
                    j_inds = np.union1d(j_inds,j_inds_lagged)
                    if np.shape(j_inds)[0] < np.shape(train_2)[0]:
                    # because if they are equal in size, we will have p > 1 when subtracting lag
                        p_j = np.shape(j_inds)[0]/(np.shape(train_2)[0]-lag)
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/(np.shape(train_1)[0]-lag)
                    else:
                        p_j = np.shape(j_inds)[0]/np.shape(train_2)[0]
                        p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/np.shape(train_1)[0]
                    if alpha > 0:
                        MI = MI + alpha + (1-alpha) * p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                    elif p_i_and_j > 0:
                        MI += p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
    return MI

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

def signed_MI(graph,raster):
    neurons = np.shape(graph)[0]
    signed_graph = np.copy(graph)
    for pre in range(0,neurons):
        for post in range ((pre+1),neurons):
            corr_mat = np.corrcoef(raster[pre,:],raster[post,:])
            #factor = sign(corr_mat[1,2])
            factor = np.sign(corr_mat[0,1])
            if ~np.isnan(factor):
                signed_graph[pre,post] *= factor
                signed_graph[post,pre] *= factor
    return signed_graph

def pos(graph):
    #takes signed_MI MI graph, returns positive version
    pos_graph = np.copy(graph)
    pos_graph[graph < 0] = 0
    return pos_graph

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
                background[pre,post] = np.mean(graph[pre,neurons[neurons != post]])*np.mean(graph[neurons[np.logical_and((neurons != post), (neurons != pre))]])
    return background

def residual(background_graph,graph):
    residual = np.copy(graph)
    neurons = np.shape(graph)[0]
    lm = LinearRegression()
    #b,m = linreg(background_graph[:],graph[:])
    model = lm.fit(background_graph[:],graph[:])
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
                norm_residual[pre,post] = np.std(graph[pre,neurons[neurons != post]])*np.std(graph[neurons[np.logical_and((neurons != post), (neurons != pre))]])
    cutoff = np.median(norm_residual)
    neurons = np.shape(graph)[0]
    norm_residual = 1/(np.sqrt(np.maximum(norm_residual,np.ones([neurons,neurons])*cutoff)))
    return norm_residual*graph


'''
def binary_raster_gen(spikes,dt):
    bins = np.arange(0,run_total,dt)
    raster = np.zeros(np.shape(spikes)[0],length(bins)-1)
    for i in range(0:np.shape(spikes)[0]):
        discrete_train = unique(round.(spike_set[i]/dt))
        discrete_train = discrete_train[discrete_train.>0]
        for j = 1:length(discrete_train)
            raster[i,Int(discrete_train[j])] = 1
    return raster
'''
