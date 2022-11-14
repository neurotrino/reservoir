"""Graph (structural) analysis for series of completed experiments"""

# ---- external imports -----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import networkx as nx


# ---- internal imports -----------------------------------------------
sys.path.append("../")
sys.path.append("../../")

from utils.misc import filenames
from utils.misc import get_experiments
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree

# ---- global variables -----------------------------------------------
data_dir = "/data/experiments/"
#experiment_string = "run-batch30-specout-onlinerate0.1-singlepreweight"
experiment_string = "run-batch30-specout-onlinerate0.1-savey"
task_experiment_string = "run-batch30-onlytaskloss"
rate_experiment_string = "run-batch30-onlyrateloss"
num_epochs = 1000
epochs_per_file = 10
e_end = 240
i_end = 300
savepath = "/data/results/experiment1/"

# Paul Tol's colorblind-friendly palette for scientific visualization
COLORS = [
    "#332288",  # indigo
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # cyan
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]


# ========= ========= ========= ========= ========= ========= =========
# Utilities
# ========= ========= ========= ========= ========= ========= =========

def _histplot_by_bin_count(x, c, lbl, bins=30):
    """Preconfigured histplot plotter."""
    sns.histplot(
        data=x,
        color=c,
        label=lbl,
        bins=bins,
        stat="density",
        alpha=0.5,
        kde=True,
        edgecolor="white",
        linewidth=0.5,
        line_kws=dict(
            color="black",
            alpha=0.5,
            linewidth=1.5
        ),
    )


def _histplot_by_bin_width(x, c, lbl, binwidth=0.1):
    """Another preconfigured histplot."""
    sns.histplot(
        data=x,
        color=c,
        label=lbl,
        binwidth=binwidth,
        stat="density",
        alpha=0.5,
        kde=True,
        edgecolor="white",
        linewidth=0.5,
        line_kws=dict(
            color="black",
            alpha=0.5,
            linewidth=1.5
        ),
    )


# ========= ========= ========= ========= ========= ========= =========
# Plot Stuff
# ========= ========= ========= ========= ========= ========= =========

def plot_input_channels():
    experiments = get_experiments(data_dir, rate_experiment_string)
    for xdir in experiments:
        # separately for each experiment
        exp_path = xdir[-9:-1]
        task_exp_path = 'rateloss_'+exp_path
        if not os.path.isdir(os.path.join(savepath, task_exp_path)):
            os.makedirs(os.path.join(savepath, task_exp_path))

        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if os.path.isfile(os.path.join(np_dir, "991-1000.npz")):

            fig, ax = plt.subplots(nrows=2, ncols=2)

            naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
            early_data = np.load(os.path.join(np_dir, "41-50.npz"))
            late_data = np.load(os.path.join(np_dir, "241-250.npz"))
            trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

            naive_in = naive_data['tv0.postweights'][0]
            early_in = early_data['tv0.postweights'][0]
            late_in = late_data['tv0.postweights'][0]
            trained_in = trained_data['tv0.postweights'][0]

            # plot each channel's distribution
            for i in range(np.shape(naive_in)[0]):
                """
                sns.histplot(
                    data=np.ravel(naive_in[i,:]),
                    bins=30,
                    stat="density",
                    alpha=0.5,
                    kde=True,
                    edgecolor="white",
                    linewidth=0.5,
                    line_kws=dict(color="black", alpha=0.5, linewidth=1.5),
                )"""
                ax[0,0].hist(naive_in[i,:],bins=50,histtype='step')
                ax[0,1].hist(early_in[i,:],bins=50,histtype='step')
                ax[1,0].hist(late_in[i,:],bins=50,histtype='step')
                ax[1,1].hist(trained_in[i,:],bins=50,histtype='step')

            ax[0,0].set_title('epoch 0')
            ax[0,0].set_xlabel('input weights')
            ax[0,1].set_title('epoch 50')
            ax[0,1].set_xlabel('input weights')
            ax[1,0].set_title('epoch 250')
            ax[1,0].set_xlabel('input weights')
            ax[1,1].set_title('epoch 1000')
            ax[1,1].set_xlabel('input weights')

            plt.suptitle("Evolution of 16 input channels' weights; rate loss only")
            plt.draw()
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            save_fname = savepath+task_exp_path+'/'+exp_path+'_input_channel_dist_quad.png'
            plt.savefig(save_fname,dpi=300)
            plt.clf()
            plt.close()

def get_degrees(arr, weighted):
    out_degree = []
    in_degree = []

    for i in range(arr.shape[0]): # calculate degrees for each unit
        if not weighted:
            out_degree.append(np.size(np.where(arr[i,:]!=0)))
            in_degree.append(np.size(np.where(arr[:,i]!=0)))
        else: # absolute weighted degree
            out_degree.append(np.sum(np.abs(arr[i,:])))
            in_degree.append(np.sum(np.abs(arr[:,i])))

    return [in_degree, out_degree]

def plot_recip_dist_experiments():
    """TODO: document function"""

    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/main_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv1.postweights"][99])

        for i in range(4):
            plt.figure()
            G = nx.from_numpy_array(w[i], create_using=nx.DiGraph)
            Ge = nx.from_numpy_array(
                w[i][0:e_end, 0:e_end], create_using=nx.DiGraph
            )
            Gi = nx.from_numpy_array(
                w[i][e_end:i_end, e_end:i_end], create_using=nx.DiGraph
            )

            # plot reciprocity between e units
            result = list(nx.reciprocity(Ge, Ge.nodes).items())
            e_recip = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(e_recip), "blue", "within e units")

            # plot reciprocity between i units
            result = list(nx.reciprocity(Gi, Gi.nodes).items())
            i_recip = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(i_recip), "red", "within i units")

            # plot whole-network reciprocity
            result = list(nx.reciprocity(G, G.nodes).items())
            recip = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(recip), "black", "whole network")

            plt.xlabel("node reciprocity for recurrent layer")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_recip_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()


def plot_eigvc_dist_experiments():
    """TODO: document function"""


    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/main_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv1.postweights"][99])

        for i in range(4):
            plt.figure()
            # G = nx.from_numpy_array(w[i],create_using=nx.DiGraph)
            Ge = nx.from_numpy_array(
                w[i][0:e_end, 0:e_end], create_using=nx.DiGraph
            )
            Gi = nx.from_numpy_array(
                np.abs(w[i][e_end:i_end, e_end:i_end]),
                create_using=nx.DiGraph,
            )
            # plot centrality between e units
            result = list(
                nx.eigenvector_centrality_numpy(Ge, weight="weight").items()
            )
            e_eigvc = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(e_eigvc), "blue", "within e units")
            # plot centrality between i units
            result = list(
                nx.eigenvector_centrality_numpy(Gi, weight="weight").items()
            )
            i_eigvc = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(i_eigvc), "red", "within i units")
            # plot whole network clustering
            # result = list(nx.clustering(G,nodes=G.nodes,weight='weight').items())
            # cc = np.array(result)[:,1]
            # sns.histplot(data=np.ravel(cc), bins=30, color='black', label='whole network', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel(
                "absolute weighted eigenvector centrality for recurrent layer"
            )
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_eigvc_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()


def plot_clustering_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/main_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv1.postweights"][99])

        for i in range(4):
            plt.figure()
            # G = nx.from_numpy_array(w[i],create_using=nx.DiGraph)
            Ge = nx.from_numpy_array(
                w[i][0:e_end, 0:e_end], create_using=nx.DiGraph
            )
            Gi = nx.from_numpy_array(
                np.abs(w[i][e_end:i_end, e_end:i_end]),
                create_using=nx.DiGraph,
            )
            # plot clustering between e units
            result = list(
                nx.clustering(Ge, nodes=Ge.nodes, weight="weight").items()
            )
            e_cc = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(e_cc), "blue", "within e units")
            # plot clustering between i units
            result = list(
                nx.clustering(Gi, nodes=Gi.nodes, weight="weight").items()
            )
            i_cc = np.array(result)[:, 1]
            _histplot_by_bin_count(np.ravel(i_cc), "red", "within i units")
            # plot whole network clustering
            # result = list(nx.clustering(G,nodes=G.nodes,weight='weight').items())
            # cc = np.array(result)[:,1]
            # sns.histplot(data=np.ravel(cc), bins=30, color='black', label='whole network', stat='density', alpha=0.5, kde=True, edgecolor='white', linewidth=0.5, line_kws=dict(color='black', alpha=0.5, linewidth=1.5))
            plt.xlabel("absolute weighted clustering for recurrent layer")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_wcc_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()


def nx_plot_clustering_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    for xdir in experiments:
        cc_e = []
        cc_i = []
        cc_all = []
        loss = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]
            loss.append(
                np.add(
                    data["step_task_loss"], data["step_rate_loss"]
                ).tolist()
            )
            for i in range(np.shape(w)[0]):
                G = nx.from_numpy_array(np.abs(w[i]), create_using=nx.DiGraph)
                cc_all.append(
                    nx.average_clustering(G, nodes=G.nodes, weight="weight")
                )
                Ge = nx.from_numpy_array(
                    w[i][0:e_end, 0:e_end], create_using=nx.DiGraph
                )
                cc_e.append(
                    nx.average_clustering(Ge, nodes=Ge.nodes, weight="weight")
                )
                Gi = nx.from_numpy_array(
                    np.abs(w[i][e_end:i_end, e_end:i_end]),
                    create_using=nx.DiGraph,
                )
                cc_i.append(
                    nx.average_clustering(Gi, nodes=Gi.nodes, weight="weight")
                )
        ax[0].plot(cc_all)
        ax[1].plot(cc_e)
        ax[2].plot(cc_i)
        ax[3].plot(loss)
    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("absolute weighted clustering coefficient")
    ax[0].set_title("whole graph")
    ax[1].set_title("within e")
    ax[2].set_title("within i")
    ax[3].set_title("loss")
    ax[3].set_ylabel("total loss")
    fig.suptitle("experiment set 1 synaptic clustering")
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_wcc.png"), dpi=300)
    plt.clf()
    plt.close()


def nx_plot_reciprocity_over_time(savepath):
    """TODO: document function"""

    # Load data
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)

    # Plotting setup
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()

    # Plot data
    for xdir in experiments:
        recip_ee = []
        recip_ei = []
        recip_ii = []
        recip_all = []

        # TODO: inline docs
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]

            # TODO: inline docs
            for w_i in w:
                G = nx.from_numpy_array(w_i, create_using=nx.DiGraph)
                recip_all.append(nx.reciprocity(G))
                Ge = nx.from_numpy_array(
                    w_i[0:e_end, 0:e_end], create_using=nx.DiGraph
                )
                recip_ee.append(nx.reciprocity(Ge))
                recip_ei.append(
                    reciprocity_ei(
                        w_i[0:e_end, e_end:i_end], w_i[e_end:i_end, 0:e_end]
                    )
                )
                Gi = nx.from_numpy_array(
                    w_i[e_end:i_end, e_end:i_end], create_using=nx.DiGraph
                )
                recip_ii.append(nx.reciprocity(Gi))

        # TODO: inline docs
        ax[0].plot(recip_ee)
        ax[1].plot(recip_ei)
        ax[2].plot(recip_ii)
        ax[3].plot(recip_all)

    # Title and label everything
    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("reciprocity")
    ax[0].set_title("within e")
    ax[1].set_title("between e and i")
    ax[2].set_title("within i")
    ax[3].set_title("whole network")
    fig.suptitle("experiment set 1 reciprocity")

    # Render and save
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_reciprocity_nx.png"), dpi=300)

    # Teardown
    plt.clf()
    plt.close()


# Calculate and plot main rsnn reciprocity as it evolves over training time
# subplots each for e-e, e-i, i-e, and i-i
def plot_reciprocity_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    # recip_arr = []
    for xdir in experiments:  # loop through all experiments of this set
        recip_ee = []
        recip_ei = []  # same as recip_ie
        recip_ii = []
        recip_all = []
        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(
                np.shape(w)[0]
            ):  # loop through 100 batch updates within each npz file
                recip_ee.append(reciprocity(w[i][0:e_end, 0:e_end]))
                recip_ei.append(
                    reciprocity_ei(
                        w[i][0:e_end, e_end:i_end], w[i][e_end:i_end, 0:e_end]
                    )
                )
                recip_ii.append(reciprocity(w[i][e_end:i_end, e_end:i_end]))
                recip_all.append(reciprocity(w[i]))
        # plot each experiment over all training time
        ax[0].plot(recip_ee)
        ax[1].plot(recip_ei)
        ax[2].plot(recip_ii)
        # stack experiment (each over all training time) into rows for meaning later
        # recip_arr = np.vstack([recip_arr,recip_all])
        ax[3].plot(recip_all)
    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("reciprocity")
    ax[0].set_title("within e")
    ax[1].set_title("between e and i")
    ax[2].set_title("within i")
    # plot whole-network mean reciprocity and std
    # ax[3].set_title('whole-network reciprocity and std')
    # recip_std = np.std(recip_arr, axis=0)
    # recip_mean = np.mean(recip_arr, axis=0)
    # ax[3].plot(recip_mean)
    # ax[3].fill_between(recip_mean-recip_std, recip_mean+recip_std, alpha=0.5)
    ax[3].set_title("whole network")

    # Create and save the final figure
    fig.suptitle("experiment set 1 reciprocity")
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_reciprocity.png"), dpi=300)
    plt.clf()
    plt.close()


# Calculate and plot in and out mean connection strength as they evolve over training time
def plot_aux_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # input to main
    # main e to output, main i to output (don't consider 0's)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    for xdir in experiments:  # loop through all experiments of this set
        input = []
        e_out = []
        i_out = []
        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            in_w = data["tv0.postweights"]
            out_w = data["tv2.postweights"]
            out_w[out_w == 0] = np.nan  # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(
                np.shape(in_w)[0]
            ):  # loop through 100 batch updates within each npz file
                input.append(np.mean(in_w[i]))
                e_out.append(np.nanmean(out_w[i][0:e_end, :]))
                i_out.append(np.nanmean(out_w[i][e_end:i_end, :]))
        # plot each experiment over all training time
        ax[0].plot(input)
        ax[1].plot(e_out)
        ax[2].plot(i_out)

    for i in range(3):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("mean weights")

    ax[0].set_title("input to main")
    ax[1].set_title("main e to output")
    ax[2].set_title("main i to output")

    # Create and save the final figure
    fig.suptitle("experiment set 1 input output weights")
    plt.draw()
    plt.subplots_adjust(hspace=1.0)
    plt.savefig(os.path.join(savepath, "set_weights_aux.png"), dpi=300)
    plt.clf()
    plt.close()


# Calculate and plot main mean connection strength as it evolves over training time
def plot_main_w_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # main network e-e, e-i, i-e, and i-i (don't consider 0's)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()

    for xdir in experiments:  # loop through all experiments of this set
        ee = []
        ei = []
        ie = []
        ii = []

        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]
            w[w == 0] = np.nan  # so we can ignore them in the mean
            # w is shaped 100 (batches x epochs) x 300 x 300
            for i in range(
                np.shape(w)[0]
            ):  # loop through 100 batch updates within each npz file
                ee.append(np.nanmean(w[i][0:e_end, 0:e_end]))
                ei.append(np.nanmean(w[i][0:e_end, e_end:i_end]))
                ie.append(np.nanmean(w[i][e_end:i_end, 0:e_end]))
                ii.append(np.nanmean(w[i][e_end:i_end, e_end:i_end]))

        # plot each experiment over all training time
        ax[0].plot(ee)
        ax[1].plot(ei)
        ax[2].plot(ie)
        ax[3].plot(ii)

    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("mean weights")

    ax[0].set_title("within e")
    ax[1].set_title("from e to i")
    ax[2].set_title("from i to e")
    ax[3].set_title("within i")

    # Create and save the final figure
    fig.suptitle("experiment set 1 main weights")
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_weights_main.png"), dpi=300)
    plt.clf()
    plt.close()


# Calculate and plot unweighted in/out degree difference for main nodes (Copeland score)
def plot_main_copeland_score_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    for xdir in experiments:  # loop through all experiments of this set
        ee_score = []
        ii_score = []
        all_score = []

        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]

            for i in range(np.shape(w)[0]):
                ee_out = np.mean(
                    out_degree(w[i][0:e_end, 0:e_end], weighted=False)
                )
                ee_in = np.mean(
                    out_degree(
                        np.transpose(w[i][0:e_end, 0:e_end]), weighted=False
                    )
                )
                ee_score.append(ee_out - ee_in)

                ii_out = np.mean(
                    out_degree(w[i][e_end:i_end, e_end:i_end], weighted=False)
                )
                ii_in = np.mean(
                    out_degree(
                        np.transpose(w[i][e_end:i_end, e_end:i_end]),
                        weighted=False,
                    )
                )
                ii_score.append(ii_out - ii_in)

                all_out = np.mean(out_degree(w[i], weighted=False))
                all_in = np.mean(
                    out_degree(np.transpose(w[i]), weighted=False)
                )
                all_score.append(all_out - all_in)

        # plot each experiment over all training time
        ax[0].plot(ee_score)
        ax[1].plot(ii_score)
        ax[2].plot(all_score)

    ax[0].set_title("within e only")
    ax[1].set_title("within i only")
    ax[2].set_title("whole graph")

    for i in range(3):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("Copeland score (out-degree minus in-degree)")

    fig.suptitle("experiment set 1 weighted in/out degree ratios")
    plt.draw()
    plt.savefig(os.path.join(savepath, "set_copelands.png"), dpi=300)
    plt.clf()
    plt.close()


# ========= ========= ========= ========= ========= ========= =========
# Plotting: Degree Metrics
# ========= ========= ========= ========= ========= ========= =========

# Calculate and plot weighted in/out degree ratio for main rsnn as they evolve over training time
# within e alone
# within i alone
# whole graph
def plot_main_degree_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()

    for xdir in experiments:  # loop through all experiments of this set
        ee_ratio = []
        ii_ratio = []
        all_ratio = []
        all_unweighted_ratio = []

        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]

            for i in range(
                np.shape(w)[0]
            ):  # for each graph (1 graph each for 100 batches per file), get mean degrees across graph
                ee_out = np.mean(
                    out_degree(w[i][0:e_end, 0:e_end], weighted=True)
                )
                ee_in = np.mean(
                    out_degree(
                        np.transpose(w[i][0:e_end, 0:e_end]), weighted=True
                    )
                )
                ee_ratio.append(ee_in / ee_out)

                ii_out = np.mean(
                    out_degree(w[i][e_end:i_end, e_end:i_end], weighted=True)
                )
                ii_in = np.mean(
                    out_degree(
                        np.transpose(w[i][e_end:i_end, e_end:i_end]),
                        weighted=True,
                    )
                )
                ii_ratio.append(ii_in / ii_out)

                all_out = np.mean(out_degree(w[i], weighted=True))
                all_in = np.mean(
                    out_degree(np.transpose(w[i]), weighted=True)
                )
                all_ratio.append(all_in / all_out)

                all_out = np.mean(out_degree(w[i], weighted=False))
                all_in = np.mean(
                    out_degree(np.transpose(w[i]), weighted=False)
                )
                all_unweighted_ratio.append(all_in / all_out)

        # plot each experiment over all training time
        ax[0].plot(ee_ratio)
        ax[1].plot(ii_ratio)
        ax[2].plot(all_ratio)
        ax[3].plot(all_unweighted_ratio)

    ax[0].set_title("within e only")
    ax[1].set_title("within i only")
    ax[2].set_title("whole graph")
    ax[3].set_title("unweighted whole graph")

    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("in/out-degree ratio")

    fig.suptitle("experiment set 1 weighted in/out degree ratios")
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_degrees.png"), dpi=300)
    plt.clf()
    plt.close()


def plot_main_out_degree_over_time(savepath):
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()

    for xdir in experiments:  # loop through all experiments of this set
        ee_ratio = []
        ii_ratio = []
        all_ratio = []
        all_unweighted_ratio = []

        for filename in data_files:  # loop through all 1000 npz files
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            w = data["tv1.postweights"]

            for i in range(
                np.shape(w)[0]
            ):  # for each graph (1 graph each for 100 batches per file), get mean degrees across graph
                ee_out = np.mean(
                    out_degree(w[i][0:e_end, 0:e_end], weighted=True)
                )
                ee_ratio.append(ee_out)

                ii_out = np.mean(
                    out_degree(w[i][e_end:i_end, e_end:i_end], weighted=True)
                )
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

    ax[0].set_title("within e only")
    ax[1].set_title("within i only")
    ax[2].set_title("whole graph")
    ax[3].set_title("unweighted whole graph")

    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("out-degrees")

    fig.suptitle("experiment set 1 weighted out degrees")
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_out_degrees.png"), dpi=300)
    plt.clf()
    plt.close()


# Naive distribution, epoch 10 distribution, epoch 100 distribution, epoch 1000 distribution
# of in and out degree
def plot_degree_dist_single_experiments():
    """TODO: document function"""


    # 4 subplots
    experiments = get_experiments(data_dir, experiment_string)
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/main_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv1.postweights"][99])
        plt_string = ["naive", "epoch10", "epoch100", "epoch1000"]

        for i in range(4):
            plt.figure()
            # plot for within e units
            d_out = out_degree(w[i][0:e_end, 0:e_end], weighted=True)
            d_in = out_degree(
                np.transpose(w[i][0:e_end, 0:e_end]), weighted=True
            )

            # plot distribution of degree ratios for all units in the
            # graph of that particular batch
            _histplot_by_bin_width(np.divide(d_in, d_out), "blue", "within e units")

            # plot for within i units
            d_out = out_degree(w[i][e_end:i_end, e_end:i_end], weighted=True)
            d_in = out_degree(
                np.transpose(w[i][e_end:i_end, e_end:i_end]), weighted=True
            )
            _histplot_by_bin_width(np.divide(d_in, d_out), "red", "within i units")

            plt.xlabel("weighted in/out-degree ratio for main rsnn")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_degree_dist_exp.png"
            plt.savefig(
                os.path.join(savepath, exp_path, plt_name), dpi=300
            )  # saved in indiv exp folders
            plt.clf()
            plt.close()


# ========= ========= ========= ========= ========= ========= =========
# Plot Stuff
# ========= ========= ========= ========= ========= ========= =========

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
        plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]

        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/output_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv2.postweights"][99])

        for i in range(4):
            plt.figure()
            e_to_out = w[i][0:e_end, :]
            i_to_out = w[i][e_end:i_end, :]

            # plot nonzero e-to-output
            _histplot_by_bin_width(
                x=np.ravel(e_to_out[e_to_out != 0]),
                c="blue",
                lbl="from e units",
                binwidth=0.05
            )

            # plot nonzero i-to-output
            _histplot_by_bin_width(
                x=np.ravel(i_to_out[i_to_out != 0]),
                c="red",
                lbl="from i units",
                binwidth=0.05
            )

            plt.xlabel("nonzero output weight distribution")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_output_w_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()


def plot_input_w_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/input_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv0.postweights"][99])

        for i in range(4):
            plt.figure()
            in_to_e = w[i][:, 0:e_end]
            in_to_i = w[i][:, e_end:i_end]
            _histplot_by_bin_count(np.ravel(in_to_e), "blue", "to e units")
            _histplot_by_bin_count(np.ravel(in_to_i), "red", "to i units")
            plt.xlabel("input weight distribution")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_input_w_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()


def plot_main_w_dist_experiments():
    experiments = get_experiments(data_dir, experiment_string)
    plt_string = ["epoch0", "epoch10", "epoch100", "epoch1000"]
    # first for naive distribution
    # second for epoch 10
    # third for epoch 100
    # fourth for epoch 1000

    for xdir in experiments:
        # create experiment-specific folder for saving if it doesn't exist yet
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(savepath, exp_path)):
            os.makedirs(os.path.join(savepath, exp_path))

        data_files = []
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/main_preweights.npy")
        )
        data_files.append(os.path.join(data_dir, xdir, "npz-data/1-10.npz"))
        data_files.append(os.path.join(data_dir, xdir, "npz-data/91-100.npz"))
        data_files.append(
            os.path.join(data_dir, xdir, "npz-data/991-1000.npz")
        )

        w = []
        # load naive weights
        w.append(np.load(data_files[0]))
        for i in range(1, 4):  # load other weights
            data = np.load(data_files[i])
            w.append(data["tv1.postweights"][99])

        for i in range(4):
            plt.figure()
            from_e = w[i][0:e_end, :]
            from_i = w[i][e_end:i_end, :]
            # plot distribution of excitatory (to e and i) weights
            _histplot_by_bin_width(
                x=np.ravel(from_e[from_e != 0]),
                c="blue",
                lbl="from e units to all",
                binwidth=0.5
            )
            # plot distribution of inhibitory (to e and i) weights
            _histplot_by_bin_width(
                x=np.ravel(from_i[from_i != 0]),
                c="red",
                lbl="from i units to all",
                binwidth=0.5
            )
            plt.xlabel("nonzero weights for recurrent layer")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()
            plt.draw()
            plt_name = plt_string[i] + "_main_w_dist_exp.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)
            plt.clf()
            plt.close()
