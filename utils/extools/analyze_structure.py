"""Graph (structural) analysis for series of completed experiments"""

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

def plot_eigvc_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ['epoch0','epoch10','epoch100','epoch1000']

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
            plt.figure()
            #G = nx.from_numpy_array(w[i],create_using=nx.DiGraph)
            Ge = nx.from_numpy_array(w[i][0:e_end,0:e_end],create_using=nx.DiGraph)
            Gi = nx.from_numpy_array(np.abs(w[i][e_end:i_end,e_end:i_end]),create_using=nx.DiGraph)
            # plot centrality between e units
            result = list(nx.eigenvector_centrality_numpy(Ge,weight='weight').items())
            e_eigvc = np.array(result)[:,1]
            sns.histplot(data=np.ravel(e_eigvc), bins=30, color='blue', label='within e units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot centrality between i units
            result = list(nx.eigenvector_centrality_numpy(Gi,weight='weight').items())
            i_eigvc = np.array(result)[:,1]
            sns.histplot(data=np.ravel(i_eigvc), bins=30, color='red', label='within i units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot whole network clustering
            #result = list(nx.clustering(G,nodes=G.nodes,weight='weight').items())
            #cc = np.array(result)[:,1]
            #sns.histplot(data=np.ravel(cc), bins=30, color='black', label='whole network', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('weighted eigenvector centrality for recurrent layer')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_eigvc_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()

def plot_clustering_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ['epoch0','epoch10','epoch100','epoch1000']

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
            plt.figure()
            #G = nx.from_numpy_array(w[i],create_using=nx.DiGraph)
            Ge = nx.from_numpy_array(w[i][0:e_end,0:e_end],create_using=nx.DiGraph)
            Gi = nx.from_numpy_array(np.abs(w[i][e_end:i_end,e_end:i_end]),create_using=nx.DiGraph)
            # plot clustering between e units
            result = list(nx.clustering(Ge,nodes=Ge.nodes,weight='weight').items())
            e_cc = np.array(result)[:,1]
            sns.histplot(data=np.ravel(e_cc), bins=30, color='blue', label='within e units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot clustering between i units
            result = list(nx.clustering(Gi,nodes=Gi.nodes,weight='weight').items())
            i_cc = np.array(result)[:,1]
            sns.histplot(data=np.ravel(i_cc), bins=30, color='red', label='within i units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot whole network clustering
            #result = list(nx.clustering(G,nodes=G.nodes,weight='weight').items())
            #cc = np.array(result)[:,1]
            #sns.histplot(data=np.ravel(cc), bins=30, color='black', label='whole network', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('absolute weighted clustering for recurrent layer')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_wcc_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()

def nx_plot_clustering_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()
    for xdir in experiments:
        cc_e = []
        cc_i = []
        cc_all = []
        loss = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            loss.append(np.add(data['step_task_loss'],data['step_rate_loss']).tolist())
            for i in range(np.shape(w)[0]):
                G = nx.from_numpy_array(np.abs(w[i]),create_using=nx.DiGraph)
                cc_all.append(nx.average_clustering(G,nodes=G.nodes,weight='weight'))
                Ge = nx.from_numpy_array(w[i][0:e_end,0:e_end],create_using=nx.DiGraph)
                cc_e.append(nx.average_clustering(Ge,nodes=Ge.nodes,weight='weight'))
                Gi = nx.from_numpy_array(np.abs(w[i][e_end:i_end,e_end:i_end]),create_using=nx.DiGraph)
                cc_i.append(nx.average_clustering(Gi,nodes=Gi.nodes,weight='weight'))
        ax[0].plot(cc_all)
        ax[1].plot(cc_e)
        ax[2].plot(cc_i)
        ax[3].plot(loss)
    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('absolute weighted clustering coefficient')
    ax[0].set_title('whole graph')
    ax[1].set_title('within e')
    ax[2].set_title('within i')
    ax[3].set_title('loss')
    ax[3].set_ylabel('total loss')
    fig.suptitle('experiment set 1 synaptic clustering')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_wcc.png"),dpi=300)
    plt.clf()
    plt.close()

def nx_plot_reciprocity_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()
    for xdir in experiments:
        recip_ee = []
        recip_ei = []
        recip_ii = []
        recip_all = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, 'npz-data', filename)
            data = np.load(filepath)
            w = data['tv1.postweights']
            for i in range(np.shape(w)[0]):
                G = nx.from_numpy_array(w[i],create_using=nx.DiGraph)
                recip_all.append(nx.reciprocity(G))
                Ge = nx.from_numpy_array(w[i][0:e_end,0:e_end],create_using=nx.DiGraph)
                recip_ee.append(nx.reciprocity(Ge))
                recip_ei.append(reciprocity_ei(w[i][0:e_end,e_end:i_end], w[i][e_end:i_end,0:e_end]))
                Gi = nx.from_numpy_array(w[i][e_end:i_end,e_end:i_end],create_using=nx.DiGraph)
                recip_ii.append(nx.reciprocity(Gi))
        ax[0].plot(recip_ee)
        ax[1].plot(recip_ei)
        ax[2].plot(recip_ii)
        ax[3].plot(recip_all)
    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('reciprocity')
    ax[0].set_title('within e')
    ax[1].set_title('between e and i')
    ax[2].set_title('within i')
    ax[3].set_title('whole network')
    # Create and save the final figure
    fig.suptitle('experiment set 1 reciprocity')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_reciprocity_nx.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot main rsnn reciprocity as it evolves over training time
# subplots each for e-e, e-i, i-e, and i-i
def plot_reciprocity_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()
    #recip_arr = []
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
                recip_ei.append(reciprocity_ei(w[i][0:e_end,e_end:i_end], w[i][e_end:i_end,0:e_end]))
                recip_ii.append(reciprocity(w[i][e_end:i_end,e_end:i_end]))
                recip_all.append(reciprocity(w[i]))
        # plot each experiment over all training time
        ax[0].plot(recip_ee)
        ax[1].plot(recip_ei)
        ax[2].plot(recip_ii)
        # stack experiment (each over all training time) into rows for meaning later
        #recip_arr = np.vstack([recip_arr,recip_all])
        ax[3].plot(recip_all)
    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('reciprocity')
    ax[0].set_title('within e')
    ax[1].set_title('between e and i')
    ax[2].set_title('within i')
    # plot whole-network mean reciprocity and std
    #ax[3].set_title('whole-network reciprocity and std')
    #recip_std = np.std(recip_arr, axis=0)
    #recip_mean = np.mean(recip_arr, axis=0)
    #ax[3].plot(recip_mean)
    #ax[3].fill_between(recip_mean-recip_std, recip_mean+recip_std, alpha=0.5)
    ax[3].set_title('whole network')

    # Create and save the final figure
    fig.suptitle('experiment set 1 reciprocity')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_reciprocity.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot in and out mean connection strength as they evolve over training time
def plot_aux_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
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
            for i in range(np.shape(in_w)[0]): # loop through 100 batch updates within each npz file
                input.append(np.mean(in_w[i]))
                e_out.append(np.nanmean(out_w[i][0:e_end,:]))
                i_out.append(np.nanmean(out_w[i][e_end:i_end,:]))
        # plot each experiment over all training time
        ax[0].plot(input)
        ax[1].plot(e_out)
        ax[2].plot(i_out)

    for i in range(3):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('mean weights')

    ax[0].set_title('input to main')
    ax[1].set_title('main e to output')
    ax[2].set_title('main i to output')

    # Create and save the final figure
    fig.suptitle('experiment set 1 input output weights')
    plt.draw()
    plt.subplots_adjust(hspace=1.0)
    plt.savefig(os.path.join(savepath,"set_weights_aux.png"),dpi=300)
    plt.clf()
    plt.close()

# Calculate and plot main mean connection strength as it evolves over training time
def plot_main_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # main network e-e, e-i, i-e, and i-i (don't consider 0's)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()

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

    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('mean weights')

    ax[0].set_title('within e')
    ax[1].set_title('from e to i')
    ax[2].set_title('from i to e')
    ax[3].set_title('within i')

    # Create and save the final figure
    fig.suptitle('experiment set 1 main weights')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
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
                all_score.append(all_out - all_in)

        # plot each experiment over all training time
        ax[0].plot(ee_score)
        ax[1].plot(ii_score)
        ax[2].plot(all_score)

    ax[0].set_title('within e only')
    ax[1].set_title('within i only')
    ax[2].set_title('whole graph')

    for i in range(3):
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
    ax=ax.flatten()

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
        ax[2].plot(all_ratio)
        ax[3].plot(all_unweighted_ratio)

    ax[0].set_title('within e only')
    ax[1].set_title('within i only')
    ax[2].set_title('whole graph')
    ax[3].set_title('unweighted whole graph')

    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('in/out-degree ratio')

    fig.suptitle('experiment set 1 weighted in/out degree ratios')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_degrees.png"),dpi=300)
    plt.clf()
    plt.close()

def plot_main_out_degree_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax=ax.flatten()

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
                ee_ratio.append(ee_out)

                ii_out = np.mean(out_degree(w[i][e_end:i_end,e_end:i_end], weighted=True))
                ii_ratio.append(ii_out)

                all_out = np.mean(out_degree(w[i], weighted=True))
                all_ratio.append(all_out)

                all_out = np.mean(out_degree(w[i], weighted=False))
                all_unweighted_ratio.append(all_out)

        # plot each experiment over all training time
        ax[0].plot(ee_ratio)
        ax[1].plot(ii_ratio)
        ax[2].plot(all_ratio)
        ax[3].plot(all_unweighted_ratio)

    ax[0].set_title('within e only')
    ax[1].set_title('within i only')
    ax[2].set_title('whole graph')
    ax[3].set_title('unweighted whole graph')

    for i in range(4):
        ax[i].set_xlabel('batch')
        ax[i].set_ylabel('out-degrees')

    fig.suptitle('experiment set 1 weighted out degrees')
    plt.draw()
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.savefig(os.path.join(savepath,"set_out_degrees.png"),dpi=300)
    plt.clf()
    plt.close()

# Naive distribution, epoch 10 distribution, epoch 100 distribution, epoch 1000 distribution
# of in and out degree
def plot_degree_dist_single_experiments():
    # 4 subplots
    experiments = get_experiments(data_dir, experiment_string)
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
        plt_string = ['naive','epoch10','epoch100','epoch1000']

        for i in range(4):
            plt.figure()
            # plot for within e units
            d_out = out_degree(w[i][0:e_end,0:e_end], weighted=True)
            d_in = out_degree(np.transpose(w[i][0:e_end,0:e_end]), weighted=True)
            # plot distribution of degree ratios for all units in the graph of that particular batch
            sns.histplot(data=np.divide(d_in,d_out), binwidth=0.1, color='blue', label='within e units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot for within i units
            d_out = out_degree(w[i][e_end:i_end,e_end:i_end], weighted=True)
            d_in = out_degree(np.transpose(w[i][e_end:i_end,e_end:i_end]), weighted=True)
            sns.histplot(data=np.divide(d_in,d_out), binwidth=0.1, color='red', label='within i units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('weighted in/out-degree ratio for main rsnn')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_degree_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300) # saved in indiv exp folders
            plt.clf()
            plt.close()

# of weights for in, main, out
# remove zeros for weight distributions, otherwise they take up too much of the density
def plot_output_w_dist_experiments():
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
            plt.figure()
            e_to_out = w[i][0:e_end,:]
            i_to_out = w[i][e_end:i_end,:]
            # plot nonzero e-to-output
            sns.histplot(data=np.ravel(e_to_out[e_to_out!=0]), binwidth=0.05, color='blue', label='from e units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot nonzero i-to-output
            sns.histplot(data=np.ravel(i_to_out[i_to_out!=0]), binwidth=0.05, color='red', label='from i units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('nonzero output weight distribution')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_output_w_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()

def plot_input_w_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ['epoch0','epoch10','epoch100','epoch1000']
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
            plt.figure()
            in_to_e = w[i][:,0:e_end]
            in_to_i = w[i][:,e_end:i_end]
            sns.histplot(data=np.ravel(in_to_e), bins=30, color='blue', label='to e units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            sns.histplot(data=np.ravel(in_to_i), bins=30, color='red', label='to i units', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('input weight distribution')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_input_w_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()

def plot_main_w_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ['epoch0','epoch10','epoch100','epoch1000']
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
            plt.figure()
            from_e = w[i][0:e_end,:]
            from_i = w[i][e_end:i_end,:]
            # plot distribution of excitatory (to e and i) weights
            sns.histplot(data=np.ravel(from_e[from_e!=0]), binwidth=0.5, color='blue', label='from e units to all', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            # plot distribution of inhibitory (to e and i) weights
            sns.histplot(data=np.ravel(from_i[from_i!=0]), binwidth=0.5, color='red', label='from i units to all', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel('nonzero weights for recurrent layer')
            plt.ylabel('density')
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i]+"_main_w_dist_exp.png"
            plt.savefig(os.path.join(savepath,exp_path,plt_name),dpi=300)
            plt.clf()
            plt.close()
