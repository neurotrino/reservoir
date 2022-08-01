"""Graph (structural) analysis for series of completed experiments"""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

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
experiment_string = 'run-batch30-specout-onlinerate0.1-singlepreweight'
num_epochs = 1000
epochs_per_file = 10
e_end = 240
i_end = 300
savepath = '/data/results/experiment1/'

# Calculate and plot main rsnn reciprocity as it evolves over training time
# subplots each for e-e, e-i, i-e, and i-i
def plot_reciprocity_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    recip_arr = []
    for xdir in experiments: # loop through all experiments of this set
        recip_ee = []
        recip_ei = [] # same as recip_ie
        recip_ii = []
        recip_all = []
        for filename in data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(np.shape(w)[0]): # loop through 100 batch updates within each npz file
                recip_ee.append(reciprocity(w[i][0:e_end,0:e_end]))
                recip_ei.append(reciprocity_ei(w[i][0:e_end,e_end:i_end], w[i][e_end:i_end,0:e_end])
                recip_ii.append(reciprocity(w[i][e_end:i_end,e_end:i_end]))
                recip_all.append(reciprocity(w[i]))
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
def plot_aux_w_over_time(savepath):
    xperiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # input to main
    # main e to output, main i to output (don't consider 0's)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    for xdir in experiments: # loop through all experiments of this set
        input = []
        e_out = []
        i_out = []
        for filename in data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            in_w = data['tv0.postweights']
            out_w = data['tv2.postweights']
            out_w[out_w==0] = np.nan # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(np.shape(w)[0]): # loop through 100 batch updates within each npz file
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
def plot_main_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # main network e-e, e-i, i-e, and i-i (don't consider 0's)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for xdir in experiments: # loop through all experiments of this set
        ee = []
        ei = []
        ie = []
        ii = []

        for filename in data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            w[w==0] = np.nan # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(np.shape(w)[0]): # loop through 100 batch updates within each npz file
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

# Calculate and plot unweighted in/out degree difference for main nodes (Copeland score)
def plot_main_copeland_score_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    for xdir in experiments: # loop through all experiments of this set
        ee_score = []
        ii_score = []
        all_score = []

        for filename in data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']

            for i in range(np.shape(w)[0]):
                ee_out = np.mean(out_degree(w[i][0:e_end,0:e_end], weighted=False))
                ee_in = np.mean(out_degree(np.transpose(w[i][0:e_end,0:e_end]), weighted=False))
                ee_score.append(ee_out - ee_in)

                ii_out = np.mean(out_degree(w[i][e_end:i_end,e_end:i_end], weighted=False))
                ii_in = np.mean(out_degree(np.transpose(w[i][e_end:i_end,e_end:i_end]), weighted=False))
                ii_score.append(ii_out - ii_in)

                all_out = np.mean(out_degree(w[i], weighted=False))
                all_in = np.mean(out_degree(np.transpose(w[i]), weighted=False))
                all_ratio.append(all_out - all_in)

        # plot each experiment over all training time
        ax[0].plot(ee_score)
        ax[1].plot(ii_score)
        ax[2].plot(all_score)

    ax[0].set_title('within e only')
    ax[1].set_title('within i only')
    ax[2].set_title('whole graph')

    for i in ax:
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('Copeland score (out-degree minus in-degree)')

    fig.suptitle('experiment set 1 weighted in/out degree ratios')
    plt.draw()
    plt.savefig(os.path.join(savepath,"set_copelands.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot weighted in/out degree ratio for main rsnn as they evolve over training time
# within e alone
# within i alone
# whole graph
def plot_main_degree_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for xdir in experiments: # loop through all experiments of this set
        ee_ratio = []
        ii_ratio = []
        all_ratio = []
        all_unweighted_ratio = []

        for filename in data_files: # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']

            for i in range(np.shape(w)[0]): # for each graph (1 graph each for 100 batches per file), get mean degrees across graph
                ee_out = np.mean(out_degree(w[i][0:e_end,0:e_end], weighted=True))
                ee_in = np.mean(out_degree(np.transpose(w[i][0:e_end,0:e_end]), weighted=True))
                ee_ratio.append(ee_in/ee_out)

                ii_out = np.mean(out_degree(w[i][e_end:i_end,e_end:i_end], weighted=True))
                ii_in = np.mean(out_degree(np.transpose(w[i][e_end:i_end,e_end:i_end]), weighted=True))
                ii_ratio.append(ii_in/ii_out)

                all_out = np.mean(out_degree(w[i], weighted=True))
                all_in = np.mean(out_degree(np.transpose(w[i]), weighted=True))
                all_ratio.append(all_in/all_out)

                all_out = np.mean(out_degree(w[i], weighted=False))
                all_in = np.mean(out_degree(np.transpose(w[i]), weighted=False))
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
def plot_degree_dist_single_experiments():
    # 4 subplots
    experiment = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=4, ncols=1)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath,exp_path)):
            os.makedirs(os.path.join(savepath,exp_path))

        data_files = []
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/main_preweights.npy'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1,4): # load other weights
            data = np.load(data_files[i])
            w.append(data['tv1.postweights'][99])

        for i in range(4):
            d_out = out_degree(w[i], weighted=True)
            d_in = out_degree(np.transpose(w[i]), weighted=True)
            # plot distribution of degree ratios for all units in the graph of that particular batch
            ax[i] = sns.histplot(data=d_in/d_out, bins=20, stat='density', alpha=1, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))

        ax[0].set_title('naive')
        ax[1].set_title('epoch 10')
        ax[2].set_title('epoch 100')
        ax[3].set_title('epoch 1000')

        for i in ax:
            ax[i].set_xlabel('in/out-degree ratio for main rsnn')
            ax[i].set_ylabel('density')

        fig.suptitle('experiment set 1 weighted in/out degree ratios')
        plt.draw()
        plt.savefig(os.path.join(savepath,exp_path,"degree_dist_exp.png"),dpi=300) # saved in indiv exp folders
        plt.clf()
        plt.close()

# of weights for in, main, out
def plot_output_w_dist_experiments():
    # 4 subplots
    experiment = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=4, ncols=1)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath,exp_path)):
            os.makedirs(os.path.join(savepath,exp_path))

        data_files = []
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/output_preweights.npy'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1,4): # load other weights
            data = np.load(data_files[i])
            w.append(data['tv2.postweights'][99])

        for i in range(4):
            ax[i] = sns.histplot(data=w[i], bins=30, stat='density', alpha=1, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))

        ax[0].set_title('naive')
        ax[1].set_title('epoch 10')
        ax[2].set_title('epoch 100')
        ax[3].set_title('epoch 1000')

        for i in ax:
            ax[i].set_xlabel('weights for output layer')
            ax[i].set_ylabel('density')

        fig.suptitle('experiment set 1 output weights')
        plt.draw()
        plt.savefig(os.path.join(savepath,exp_path,"output_w_dist_exp.png"),dpi=300) # saved in indiv exp folders
        plt.clf()
        plt.close()

def plot_input_w_dist_experiments():
    # 4 subplots
    experiment = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=4, ncols=1)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath,exp_path)):
            os.makedirs(os.path.join(savepath,exp_path))

        data_files = []
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/input_preweights.npy'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1,4): # load other weights
            data = np.load(data_files[i])
            w.append(data['tv0.postweights'][99])

        for i in range(4):
            ax[i] = sns.histplot(data=w[i], bins=30, stat='density', alpha=1, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))

        ax[0].set_title('naive')
        ax[1].set_title('epoch 10')
        ax[2].set_title('epoch 100')
        ax[3].set_title('epoch 1000')

        for i in ax:
            ax[i].set_xlabel('weights for input layer')
            ax[i].set_ylabel('density')

        fig.suptitle('experiment set 1 input weights')
        plt.draw()
        plt.savefig(os.path.join(savepath,exp_path,"input_w_dist_exp.png"),dpi=300) # saved in indiv exp folders
        plt.clf()
        plt.close()

def plot_main_w_dist_experiments():
    # 4 subplots
    experiment = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=4, ncols=2)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath,exp_path)):
            os.makedirs(os.path.join(savepath,exp_path))

        data_files = []
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/main_preweights.npy'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
        data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1,4): # load other weights
            data = np.load(data_files[i])
            w.append(data['tv1.postweights'][99])

        for i in range(4):
            # plot distribution of excitatory (to e and i) weights
            ax[i] = sns.histplot(data=w[i][0:e_end,:], bins=30, stat='density', alpha=1, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
            # plot distribution of inhibitory (to e and i) weights
            ax[i+4] = sns.histplot(data=w[i][e_end:i_end,:], bins=30, stat='density', alpha=1, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))

        ax[0].set_title('e naive')
        ax[1].set_title('e epoch 10')
        ax[2].set_title('e epoch 100')
        ax[3].set_title('e epoch 1000')
        ax[4].set_title('i naive')
        ax[5].set_title('i epoch 10')
        ax[6].set_title('i epoch 100')
        ax[7].set_title('i epoch 1000')

        for i in ax:
            ax[i].set_xlabel('weights for recurrent layer')
            ax[i].set_ylabel('density')

        fig.suptitle('experiment set 1 main weights')
        plt.draw()
        plt.savefig(os.path.join(savepath,exp_path,"main_w_dist_exp.png"),dpi=300) # saved in indiv exp folders
        plt.clf()
        plt.close()