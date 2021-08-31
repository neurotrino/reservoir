#MI.py
#calculates confluent mutual information based on spikes from .npz files

import sys

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

dt = 1
experiment = ''
run_dur = 4080
# OR should be strung together for all trials in a batch
batch_size = 10
#run_total = run_dur * batch_size

def main(experiment,dt):
    # loop through and load all experiment npz files
    data = np.load('11-20.npz')
    spikes = data['spikes']

    # 10 epochs each file, 10 batches each epoch, 10 trials each batch
    # dimensions of 100 x 10 x 4080 x 100
    # arbitrarily choosing [0,0] for now, then transpose to get neurons x run_dur
    raster = np.transpose(spikes[0][0])
    mi_graph = generate_mi_graph(raster,dt)
    # can do processing to create final functional graph
    # such as taking only the top 25% of the mi graph or other such actions
    functional_graph = mi_graph

    w_rec = data['tv1.preweights']
    # note that we are using pre-update weights, since these are the ones actually generating the spikes
    recruitment_graph = intersect_functional_and_synaptic(functional_graph,w_rec)

def generate_mi_graph(raster,dt):
    #raster = binary_raster_gen(spikes,dt)
    MI_graph = confMI_mat(raster)
    signed_graph = signed_MI(MI_graph,raster)
    pos_graph = pos(signed_graph)
    reexpress_graph = reexpress_param(pos_graph)
    background_graph = background(reexpress_graph)
    residual_graph = residual(background_graph,MI_graph)
    normed_MI_graph = normed_residual(residual_graph)
    return normed_MI_graph

def confMI_mat(raster):
    lag = 1
    alpha = 0
    neurons = np.shape(raster)[0]
    mat = np.zeros([neurons,neurons])
    for post in range(0,neurons):
        for pre in range(0,neurons):
            if post != pre:
                mat[post,pre] = confMI(raster[pre,:],raster[post,:],lag,alpha)
    return mat

def confMI(train_1,train_2,lag,alpha):
    MI = 0
    states = [0,1]
    for i in range(0,np.size(states)):
        i_inds = np.argwhere(train_1 == states[i])
        p_i = np.shape(i_inds)[0]/np.shape(train_1)[0]
        if np.shape(i_inds)[0] > 0:
            for j in range(0,np.size(states)):
                j_inds = np.argwhere(train_2 == states[j])
                j_inds_lagged = j_inds + lag
                j_inds_lagged = j_inds_lagged[j_inds_lagged < run_dur]
                if np.shape(j_inds)[0] > 0:
                    j_inds_lagged = j_inds_lagged[j_inds_lagged>0]
                    j_inds = np.union1d(j_inds,j_inds_lagged)
                    p_j = np.shape(j_inds)[0]/(np.shape(train_2)[0])
                    p_i_and_j = np.shape(np.intersect1d(i_inds,j_inds))[0]/(np.shape(train_1)[0])
                    if alpha > 0:
                        MI = MI + alpha + (1-alpha) * p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
                    elif p_i_and_j > 0:
                        MI += p_i_and_j * np.log2(p_i_and_j/(p_i*p_j))
    return MI

def signed_MI(graph,raster):
    neurons = np.shape(graph)[0]
    signed_graph = np.copy(graph)
    for post in range(0,neurons):
        for pre in range ((post+1),neurons):
            corr_mat = np.corrcoef(raster[pre,:],raster[post,:])
            #factor = sign(corr_mat[1,2])
            factor = np.sign(corr_mat[0,1])
            if ~np.isnan(factor):
                signed_graph[post,pre] *= factor
                signed_graph[pre,post] *= factor

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
    upper_skew = scipy.stats.skewness(data**upper_exp)
    lower_skew = scipy.stats.skewness(data**lower_exp)
    for i in range(0,steps):
        new_exp = (upper_exp + lower_exp)/2
        new_skew = scipy.stats.skewness(data**new_exp)
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
    neurons = np.shape(graph)[0]
    for pre in range(0,neurons):
        for post in range(0,neurons):
            if post != pre:
                background[post,pre] = np.mean(graph[post,0:neurons[neurons != pre]])*np.mean(graph[0:neurons[(neurons != post) & (neurons != pre)]])
    return background

def residual(background_graph,graph):
    residual = np.copy(graph)
    neurons = np.shape(graph)[0]
    lm = LinearRegression()
    #b,m = linreg(background_graph[:],graph[:])
    model = lm.fit(background_graph[:],graph[:])
    b = {model.intercept_}
    m = {model.coef_}
    residual = graph - (m*background_graph + b)
    return residual

def normed_residual(graph):
    norm_residual = np.copy(graph)
    neurons = np.shape(graph)[0]
    for pre in range(0,neurons):
        for post in range(0,neurons):
            if post != pre:
                norm_residual[post,pre] = np.std(graph[post,0:neurons[neurons != pre]])*np.std(graph[0:neurons[(neurons != post) & (neurons != pre)]])
    cutoff = np.median(norm_residual)
    norm_residual = 1/(np.sqrt(np.maximum(norm_residual,np.ones(neurons,neurons)*cutoff)))
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
