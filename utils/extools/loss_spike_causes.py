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
#experiment = "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower"
experiment = 'fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger'
savepath = '/data/results/fwd/loss_causes_over_time_0.001.png'
# move on to others as you desire

# create four subplots
fig, axes = plt.subplots(5, figsize=(6, 8), sharex=True)

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

    """
    # plot main e/i ratio (weighted)
    ax1.scatter(losses[1:len(losses)],eiratio_in[0:len(eiratio_in)-1],s=1,marker='*')
    ax1.set_xlabel('loss')
    ax1.set_ylabel('input e/i ratio')
    ax1.set_title('Input Weighted E/I Ratio and Loss')
    ax2.scatter(losses[1:len(losses)],eiratio_main[0:len(eiratio_main)-1],s=1,marker='*')
    ax2.set_xlabel('loss')
    ax2.set_ylabel('main e/i ratio')
    ax2.set_title('Main Weighted E/I Ratio and Loss')
    ax3.scatter(losses[1:len(losses)],recip_main[0:len(recip_main)-1],s=1,marker='*')
    ax3.set_xlabel('loss')
    ax3.set_ylabel('main reciprocity')
    ax3.set_title('Main Reciprocity and Loss')
    ax4.scatter(losses[1:len(losses)],eiratio_out[0:len(eiratio_out)-1],s=1,marker='*')
    ax4.set_xlabel('loss')
    ax4.set_ylabel('output e/i ratio')
    ax4.set_title('Output E/I Ratio and Loss')
    """
    axes[0].plot(losses[1:len(losses)])
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('batch')

    axes[1].plot(eiratio_in[0:len(eiratio_in)-1])
    axes[1].set_ylabel('input layer e/i ratio')
    axes[1].set_xlabel('batch')

    axes[2].plot(eiratio_main[0:len(eiratio_main)-1])
    axes[2].set_ylabel('main layer e/i ratio')
    axes[2].set_xlabel('batch')

    axes[3].plot(recip_main[0:len(recip_main)-1])
    axes[3].set_ylabel('main layer reciprocity')
    axes[3].set_xlabel('batch')

    axes[4].plot(eiratio_out[0:len(eiratio_out)-1])
    axes[4].set_ylabel('output layer e/i ratio')
    axes[4].set_xlabel('batch')

    fig.suptitle("main lr 0.001, output lr 0.00001")
    #plt.subplots_adjust(left=0.15,bottom=0.1,right=0.95,top=0.9,wspace=0.4,hspace=0.4)
    plt.draw()
    plt.savefig(savepath,dpi=300)
    plt.clf()
    plt.close()

    # plot output e/i ratio (weighted)
    # plot main density v loss
    # plot main reciprocity v loss
    # plot main number of transitions from zero t0 nonzero v loss
