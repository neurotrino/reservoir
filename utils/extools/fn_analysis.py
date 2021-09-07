#fn_analysis.py

import sys
import os
import matplotlib.pyplot as plt
import argparse
import glob
import logging
import numpy as np
import pickle
import seaborn as sns
from MI import *

start_file = '/home/macleanlab/experiments/sinusoid_save_spikes/npz-data/1-10.npz'
end_file = '/home/macleanlab/experiments/sinusoid_save_spikes/npz-data/111-120.npz'
start_batch = 10
end_batch = 99

def compare_begin_end(start_file,start_batch,end_file,end_batch):
    savedir = '/home/macleanlab/experiments/sinusoid_save_spikes/analysis/'
    plot_quad_compare(start_file,start_batch,savedir + 'fn_quad_epoch_1.png')
    plot_quad_compare(end_file,end_batch,savedir + 'fn_quad_epoch_120.png')


def plot_quad_compare(infile,batch,savefile):
    # load correct synaptic graphs
    dt = 1
    data = np.load(infile)
    syn_w = data['tv1.postweights'][batch-1]
    units = np.shape(syn_w)[0]
    density = calc_density(syn_w)
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    # plot synaptic heatmap
    syn_heatmap = gen_heatmap(syn_w, 'Synaptic Graph; density = ' + density, axis=ax1)

    # calculate fn
    spikes = data['spikes']
    batch_spikes = np.reshape(spikes[batch], [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]])
    batch_raster = np.transpose(batch_spikes)
    mi_graph = generate_mi_graph(batch_raster,dt)
    density = calc_density(mi_graph)
    # plot full mi graph
    mi_heatmap = gen_heatmap(mi_graph, 'Full FN; density = ' + density, axis=ax2)

    # calculate top quartile based on abs + weights
    thresh_e = np.quantile(np.abs(mi_graph[mi_graph>0]),0.75)
    top_quartile_mi = np.copy(mi_graph)
    for i in range(0,units):
        for j in range(0,units):
            if abs(top_quartile_mi[i,j]) < thresh:
                 top_quartile_mi[i,j] = 0
    density = calc_density(top_quartile_mi)
    # plot top quartile
    top_quartile_heatmap = gen_heatmap(top_quartile_mi, 'Top quartile FN; density = ' + density, axis=ax3)

    # separate e and i
    e_units = int(0.8*units)
    mi_e_i = np.copy(mi_graph)
    for i in range(0,e_units):
        for j in range(0,units):
            if mi_graph[i,j] < 0:
                mi_e_i[i,j] = 0
    for i in range(e_units,units):
        for j in range(0,units):
            if mi_graph[i,j] > 0:
                mi_e_i[i,j] = 0
    density = calc_density(mi_e_i)
    # plot separate/enforced e and i
    e_i_heatmap = gen_heatmap(mi_e_i, 'FN enforcing + and -; density = ' + density, axis=ax4)

    plt.savefig(savefile)
    plt.clf()

def calc_density(graph):
    ct_nonzero = np.size(graph[graph]!=0)
    ct_total = np.size(graph) - np.shape(graph)[0]
    density = ct_nonzero/ct_total
    return density

def gen_heatmap(weights, title, axis, show_value_bounds=True):
    weights = weights.copy()
    min_wgt = np.min(weights)
    max_wgt = np.max(weights)
    ticklabels = [' (−)', ' 0', ' (+)']

    if min_wgt < 0:
        weights[weights < 0] /= abs(min_wgt)
        if show_value_bounds:
            ticklabels[0] = f' {min_wgt:.4f}'
    if max_wgt > 0:
        weights[weights > 0] /= max_wgt
        if show_value_bounds:
            ticklabels[2] = f' {max_wgt:.4f}'

    heatmap = sns.heatmap(
        weights,
        cmap = 'seismic', vmin=-1, vmax=1,
        cbar_kws={
            'ticks': [-1, 0, 1],
            'label': 'synapse strength'
        },
        xticklabels=[], yticklabels=[],
        ax=axis
    )
    heatmap.collections[0].colorbar.set_ticklabels(ticklabels)

    heatmap.set_title(title)
    heatmap.set_xlabel('target neuron')
    heatmap.set_ylabel('projecting neuron')

    return heatmap

def reciprocity(graph):
    units = np.shape(graph)[0]
    reciprocal_ct = 0
    for i in range(0,units):
        for j in range(0,units):
            if i!=j and graph[i,j] !=0 and graph[j,i] !=0:
                reciprocal_ct += 1
    possible_reciprocal_ct = np.size(graph) - units
    return reciprocal_ct
