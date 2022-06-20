# Script for plotting / determining what happens in connectivity to cause spikes in loss during training

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from compare_losses import filenames

data_dir = '/data/experiments/'
num_epochs = 1000
epochs_per_file = 10
loss_of_interest="step_loss"

# begin with main lr 0.005, output lr 0.00001
experiment = "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower"
savepath = '/data/results/fwd/loss_spike_causes_0.005.png'
# move on to others as you desire

# create four subplots
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

def reciprocity(graph):
    units = np.shape(graph)[0]
    reciprocal_ct = 0
    for i in range(0,units):
        for j in range(i+1,units):
                if graph[i,j]!=0 and graph[j,i]!=0:
                    reciprocal_ct += 1
    possible_reciprocal_ct = np.size(graph) - units
    recip_ratio = reciprocal_ct/possible_reciprocal_ct
    return recip_ratio

def scatter_reasons():
    # comparison is always for the previous epoch's postweights (current preweights) and current epoch total loss
    data_files = filenames(num_epochs, epochs_per_file)
    losses = []
    eiratio_in = []
    eiratio_main = []
    eiratio_out = []
    recip_main = []
    
    for filename in data_files:
        filepath = os.path.join(data_dir, experiment, "npz-data", filename)
        data = np.load(filepath)
        losses += data[loss_of_interest].tolist()
        w_in = data['tv0.postweights']
        for i in range(0,len(w_in)):
            w = w_in[i]
            e = w[w>0]
            i = w[w<0]
            eiratio_in = np.append(eiratio_in, np.abs(np.sum(e)/np.sum(i)))
        w_main = data['tv1.postweights']
        for i in range(0,len(w_main)):
            w = w_main[i]
            recip_ratio = reciprocity(w)
            recip_main = np.append(recip_main, recip_ratio)
            e = w[w>0]
            i = w[w<0]
            eiratio_main = np.append(eiratio_main, np.abs(np.sum(e)/np.sum(i)))
        w_out = data['tv2.postweights']
        for i in range(0,len(w_out)):
            w = w_out[i]
            e = w[w>0]
            i = w[w<0]
            eiratio_out = np.append(eiratio_out, np.abs(np.sum(e)/np.sum(i)))

    # plot main e/i ratio (weighted)
    ax1.plt.scatter(loss[1:len(loss)],eiratio_in[0:len(eiratio_in)-1])
    ax1.set_xlabel('loss')
    ax1.set_ylabel('input e/i ratio')
    ax2.plt.scatter(loss[1:len(loss)],eiratio_main[0:len(eiratio_main)-1])
    ax2.set_xlabel('loss')
    ax2.set_ylabel('main e/i ratio')
    ax3.plt.scatter(loss[1:len(loss)],recip_main[0:len(recip_main)-1])
    ax3.set_xlabel('loss')
    ax3.set_ylabel('main reciprocity')
    ax4.plt.scatter(loss[1:len(loss)],eiratio_out[0:len(eiratio_out)-1])
    ax4.set_xlabel('loss')
    ax4.set_ylabel('output e/i ratio')
    fig.suptitle("main lr 0.005, output lr 0.00001")
    plt.draw()
    plt.savefig(savepath)
    plt.clf()
    plt.close()

    # plot output e/i ratio (weighted)
    # plot main density v loss
    # plot main reciprocity v loss
    # plot main number of transitions from zero t0 nonzero v loss
