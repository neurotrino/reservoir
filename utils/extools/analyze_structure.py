"""Graph (structural) analysis for series of completed experiments"""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
sys.path.append('../../')

# internal ----
from utils.misc import filenames
from utils.misc import get_experiments
from utils.extools.loss_spike_causes import reciprocity
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree

data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-singlepreweight'
num_epochs = 1000
epochs_per_file = 10
e_end = 240
i_end = 300
savepath = '/data/results/experiment1/'

# Calculate and plot main rsnn reciprocity as it evolves over training time
# subplots each for e-e, e-i, i-e, and i-i
plot_reciprocity_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    recip_arr = []
    for xdir in experiments: # loop through all experiments of this set
        recip_ee = []
        recip_ei = [] # same as recip_ie
        recip_ii = []
        recip_all = []
        for filename = data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in np.shape(w)[0]: # loop through 100 batch updates within each npz file
                recip_ee.append(reciprocity(w[i][0:e_end,0:e_end], within_type = True))
                recip_ei.append(reciprocity(w[i][0:e_end,e_end:i_end], within_type = False))
                recip_ii.append(reciprocity(w[i][e_end:i_end,e_end:i_end], within_type = True))
                recip_all.append(reciprocity(w[i]), within_type = True)
        # plot each experiment over all training time
        ax[0].plot(recip_ee)
        ax[1].plot(recip_ei)
        ax[2].plot(recip_ii)
        # stack experiment (each over all training time) into rows for meaning later
        recip_arr = np.vstack([recip_arr,recip_all])
    for i in ax:
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('reciprocity')
    ax[0].set_title('within e')
    ax[1].set_title('between e and i')
    ax[2].set_title('within i')
    # plot whole-network mean reciprocity and std
    ax[3].set_title('whole-network reciprocity and std')
    recip_std = np.std(recip_arr, axis=0)
    recip_mean = np.mean(recip_arr, axis=0)
    ax[3].plot(recip_mean)
    ax[3].fill_between(recip_mean-recip_std, recip_mean+recip_std, alpha=0.5)

    # Create and save the final figure
    fig.suptitle('experiment set 1 reciprocity')
    plt.draw()
    plt.savefig(os.path.join(savepath,"set_reciprocity.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot in and out mean connection strength as they evolve over training time
plot_aux_w_over_time(savepath):
    xperiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # input to main
    # main e to output, main i to output (don't consider 0's)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    for xdir in experiments: # loop through all experiments of this set
        input = []
        e_out = []
        i_out = []
        for filename = data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            in_w = data['tv0.postweights']
            out_w = data['tv2.postweights']
            out_w[out_w==0] = np.nan # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in np.shape(w)[0]: # loop through 100 batch updates within each npz file
                input.append(np.mean(in_w[i]))
                e_out.append(np.nanmean(out_w[i][0:e_end,:]))
                i_out.append(np.nanmean(out_w[i][e_end:i_end,:]))
        # plot each experiment over all training time
        ax[0].plot(input)
        ax[1].plot(e_out)
        ax[2].plot(i_out)

    for i in ax:
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('mean weights')

    ax[0].set_title('input to main')
    ax[1].set_title('main e to output')
    ax[1].set_title('main i to output')

    # Create and save the final figure
    fig.suptitle('experiment set 1 input output weights')
    plt.draw()
    plt.savefig(os.path.join(savepath,"set_weights_aux.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot main mean connection strength as it evolves over training time
plot_main_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # main network e-e, e-i, i-e, and i-i (don't consider 0's)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for xdir in experiments: # loop through all experiments of this set
        ee = []
        ei = []
        ie = []
        ii = []

        for filename = data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            w[w==0] = np.nan # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in np.shape(w)[0]: # loop through 100 batch updates within each npz file
                ee.append(np.nanmean(w[i][0:e_end,0:e_end]))
                ei.append(np.nanmean(w[i][0:e_end,e_end:i_end]))
                ie.append(np.nanmean(w[i][e_end:i_end,0:e_end]))
                ii.append(np.nanmean(w[i][e_end:i_end,e_end:i_end]))

        # plot each experiment over all training time
        ax[0].plot(ee)
        ax[1].plot(ei)
        ax[2].plot(ie)
        ax[3].plot(ii)

    for i in ax:
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('mean weights')

    ax[0].set_title('within e')
    ax[1].set_title('from e to i')
    ax[2].set_title('from i to e')
    ax[3].set_title('within i')

    # Create and save the final figure
    fig.suptitle('experiment set 1 main weights')
    plt.draw()
    plt.savefig(os.path.join(savepath,"set_weights_main.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot in/out degree for main rsnn as they evolve over training time
# within e alone
# within i alone
# whole graph
# weighted and unweighted for all
def plot_degree_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for xdir in experiments: # loop through all experiments of this set
        ee_ratio = []
        ii_ratio = []
        all_ratio = []
        all_unweighted_ratio = []

        for filename = data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']

            for i in np.shape(w)[0]:
                ee_out = out_degree(w[i][0:e_end,0:e_end], weighted=True)
                ee_in = out_degree(np.transpose(w[i][0:e_end,0:e_end]), weighted=True)
                ee_ratio.append(ee_in/ee_out)

                ii_out = out_degree(w[i][e_end:i_end,e_end:i_end], weighted=True)
                ii_in = out_degree(np.transpose(w[i][e_end:i_end,e_end:i_end]), weighted=True)
                ii_ratio.append(ii_in/ii_out)

                all_out = out_degree(w[i], weighted=True)
                all_in = out_degree(np.transpose(w[i]), weighted=True)
                all_ratio.append(all_in/all_out)

                all_out = out_degree(w[i], weighted=False)
                all_in = out_degree(np.transpose(w[i]), weighted=False)
                all_unweighted_ratio.append(all_in/all_out)

        # plot each experiment over all training time
        ax[0].plot(ee_ratio)
        ax[1].plot(ii_ratio)
        ax[2].plot(all_weighted_ratio)
        ax[3].plot(all_ratio)

    ax[0].set_title('within e only')
    ax[1].set_title('within i only')
    ax[2].set_title('whole graph')
    ax[3].set_title('unweighted whole graph')

    for i in ax:
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('in/out-degree ratio')

    fig.suptitle('experiment set 1 weighted in/out degree ratios')
    plt.draw()
    plt.savefig(os.path.join(savepath,"set_degrees.png"),dpi=300)
    plt.clf()
    plt.close()

# Naive distribution, epoch 10 distribution, epoch 100 distribution, epoch 1000 distribution
# of in and out degree
def plot_degree_dist():
# of weights for in, main, out
def plot_aux_w_dist():
def plot_main_w_dist():
