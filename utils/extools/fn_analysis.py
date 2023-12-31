"""Functional analyses on saved model output data."""

# ---- external imports -----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# ---- internal imports -----------------------------------------------
#from utils.extools.MI import generate_mi_graph
from utils.misc import filenames

# ---- global variables -----------------------------------------------
xdir = ""
data_dir = "/home/macleanlab/experiments/ccd_save_spikes/npz-data/"
start_file = os.path.join(data_dir, "1-10.npz")
mid_file = os.path.join(data_dir, "91-100.npz")
end_file = os.path.join(data_dir, "191-200.npz")
start_batch = 99
mid_batch = 99
end_batch = 99
batch = 99
run_dur = 4080
dt = 1
savedir = "/home/macleanlab/experiments/ccd_save_spikes/analysis/"


# ========= ========= ========= ========= ========= ========= =========
# Compute Stuff
# ========= ========= ========= ========= ========= ========= =========

def calc_density(graph):
    """TODO: document function"""
    ct_nonzero = np.size(graph[graph != 0])
    ct_total = np.size(graph) - np.shape(graph)[0]
    density = ct_nonzero / ct_total
    return density


def out_degree(graph, weighted):
    """TODO: document function

    Providing a transposed graph will provide the in-degree instead.

    Arguments:
        graph: NumPy array TODO: document parameter
        weightd: bool TODO: document parameter
    """
    axes = tuple(range(1, graph.ndim))

    if weighted:
        return np.sum(graph, axis=axes)
    else:
        return np.count_nonzero(graph, axis=axes)


def reciprocity(graph):
    """Calculate reciprocity.

    Reciprocity is calculated as the number of present reciprocal
    connections over the number of reciprocal connections possible.

    Arguments:
        graph<NumPy Array>: an adjacency matrix of the network whose
            reciprocity is to be calculated.
    """

    # Determine the possible number of reciprocal connections
    possible_reciprocal_ct = np.count_nonzero(graph)

    # If no reciprocal connections are even possible, reciprocity is
    # undefined
    if possible_reciprocal_ct <= 0:
        return 0  # TODO: should we use NaN/None for undef. ?

    # Count actual number of reciprocal connections
    #
    # Multiplying (elementwise) the adjacency matrix by its transpose
    # creates a symmetric matrix with zero-valued entries indicating
    # non-reciprocal connections (because g[i, j] * g.T[i, j] equals
    # g[i, j] * g[j, i], so if either are zero, the value will be zero
    # in the symmetric matrix). Ignoring the diagonal, we can count the
    # number of non-zero entries to get the number of reciprocal
    # connections (each pair of vertices counts as two reciprocal
    # connections). This only works because `graph` is an adjacency
    # matrix, and therefore square
    sym_mat = graph * graph.T                  # create sym. matrix
    np.fill_diagonal(sym_mat, 0)               # ignore the diagonal
    reciprocal_ct = np.count_nonzero(sym_mat)  # count r. connections

    # Return the fraction of actual reciprocal connections over the
    # number of reciprocal connections possible
    return reciprocal_ct / possible_reciprocal_ct


def reciprocity_ei(e_graph, i_graph):
    """Reciprocity for excitatory/inhbitory graphs."""

    reciprocal_ct = 0  # counter variable
    e_units = e_graph.shape[0]
    i_units = i_graph.shape[0]
    for i in range(0, e_units):
        for j in range(0, i_units):
            if e_graph[i, j] != 0 and i_graph[j, i] != 0:
                reciprocal_ct += 1

    possible_reciprocal_ct = np.size(e_graph[e_graph != 0])
    return reciprocal_ct / possible_reciprocal_ct


# ========= ========= ========= ========= ========= ========= =========
# Plot Stuff
# ========= ========= ========= ========= ========= ========= =========

def DISABLED_compare_syn_fn(data_dir, batch):  # was commented out
    """TODO: document function"""

    data_files = filenames(num_epochs=200, epochs_per_file=10)

    # only keeping final batch from each epoch group
    syn_fn_corr = []
    sign_constrained_corr = []
    abs_value_corr = []

    for filename in data_files:
        filepath = os.path.join(data_dir, xdir, "npz-data", filename)
        data = np.load(filepath)

        # get synaptic graph
        syn_w = data["tv1.postweights"][batch - 1]  # synapic weights
        units = np.shape(syn_w)[0]  # number of units

        # calculate mi graph
        spikes = data["spikes"]
        batch_spikes = np.reshape(  # [!] document
            spikes[batch],
            [
                run_dur * np.shape(spikes[batch])[0],
                np.shape(spikes[batch])[2],
            ],
        )
        batch_raster = np.transpose(batch_spikes)
        mi_graph = generate_mi_graph(batch_raster, dt)

        # make all mi self-connections 0
        np.fill_diagonal(mi_graph, 0)

        # save mi_graph
        np.save(os.path.join(savedir, "mi_graph_", group, ".npy", mi_graph))

        # calculate element-wise correlation between syn_w and
        # mi_graph - no constraints
        unconstrained_corr = np.corrcoef(
            np.reshape(syn_w, -1), np.reshape(mi_graph, -1)
        )[0, 1]
        syn_fn_corr.append(unconstrained_corr)

        # calculate corr only considering mi_graph values that align
        # with +/- according to projecting neuron
        e_units = int(0.8 * units)
        mi_e_i = np.copy(mi_graph)

        for i in range(0, e_units):
            for j in range(0, units):
                if mi_graph[i, j] < 0:
                    mi_e_i[i, j] = 0

        for i in range(e_units, units):
            for j in range(0, units):
                if mi_graph[i, j] > 0:
                    mi_e_i[i, j] = 0

        signed_corr = np.corrcoef(
            np.reshape(syn_w, -1), np.reshape(mi_e_i, -1)
        )[0, 1]
        sign_constrained_corr.append(signed_corr)

        # calculate corr considering absolute values of each synapse
        abs_corr = np.corrcoef(
            np.reshape(np.abs(syn_w), -1), np.reshape(np.abs(mi_graph), -1)
        )[0, 1]
        abs_value_corr.append(abs_corr)

    # plot correlations over epochs
    labels = [
        "all conns as they are",
        "only +/- matching functional conns",
        "absolute values of all conns",
    ]
    plt.plot(epochs, syn_fn_corr, label=labels[0])
    plt.plot(epochs, sign_constrained_corr, label=labels[1])
    plt.plot(epochs, abs_value_corr, label=labels[2])
    plt.title("Correlation between Synaptic and Functional Weights")
    plt.xlabel("training epoch")
    plt.ylabel("pearson's r")
    plt.legend()
    plt.draw()
    plt.savefig(savedir + "syn_fn_corr.png", dpi=300)



def presaved_compare_syn_fn(data_dir, batch):
    """TODO: document function"""
    epochs = np.arange(10, 201, 10)
    epoch_groups = [
        f"{i - 9}-{i}" for i in epochs  # ["1-10", ..., "191-200"]
    ]

    # only keeping final batch from each epoch group
    syn_fn_corr = []
    sign_constrained_corr = []
    abs_value_corr = []
    for group in epoch_groups:
        data = np.load(data_dir + group + ".npz")

        # get synaptic graph
        syn_w = data["tv1.postweights"][batch - 1]
        units = np.shape(syn_w)[0]

        # load pre-calculated mi graph
        mi_graph = np.load(savedir + "mi_graph_" + group + ".npy")

        # calculate element-wise correlation between syn_w and
        # mi_graph - no constraints
        unconstrained_corr = np.corrcoef(
            np.reshape(syn_w, -1), np.reshape(mi_graph, -1)
        )[0, 1]
        syn_fn_corr.append(unconstrained_corr)

        # calculate corr only considering mi_graph values that align
        # with +/- according to projecting neuron
        e_units = int(0.8 * units)
        mi_e_i = np.copy(mi_graph)
        for i in range(0, e_units):
            for j in range(0, units):
                if mi_graph[i, j] < 0:
                    mi_e_i[i, j] = 0
        for i in range(e_units, units):
            for j in range(0, units):
                if mi_graph[i, j] > 0:
                    mi_e_i[i, j] = 0
        signed_corr = np.corrcoef(
            np.reshape(syn_w, -1), np.reshape(mi_e_i, -1)
        )[0, 1]
        sign_constrained_corr.append(signed_corr)

        # calculate corr considering absolute values of each synapse
        abs_corr = np.corrcoef(
            np.reshape(np.abs(syn_w), -1), np.reshape(np.abs(mi_graph), -1)
        )[0, 1]
        abs_value_corr.append(abs_corr)

    # plot correlations over epochs
    labels = [
        "all conns as they are",
        "only +/- matching functional conns",
        "absolute values of all conns",
    ]
    plt.plot(epochs, syn_fn_corr, label=labels[0])
    plt.plot(epochs, sign_constrained_corr, label=labels[1])
    plt.plot(epochs, abs_value_corr, label=labels[2])
    plt.title("Correlation between Synaptic and Functional Weights")
    plt.xlabel("training epoch")
    plt.ylabel("pearson's r")
    plt.legend()
    plt.draw()
    plt.savefig(savedir + "syn_fn_corr.png", dpi=300)

def plot_quad_compare(infile, batch, savefile):
    """TODO: document function"""
    # load correct synaptic graphs
    dt = 1
    data = np.load(infile)
    syn_w = data["tv1.postweights"][batch - 1]
    units = np.shape(syn_w)[0]
    density = calc_density(syn_w)
    fig, ax = plt.subplots(2, 2)
    # plot synaptic heatmap
    syn_heatmap = gen_heatmap(
        syn_w,
        "Synaptic Graph; " + f"{density*100:.1f}" + "% conn",
        axis=ax[0, 0],
    )

    # calculate fn
    spikes = data["spikes"]
    batch_spikes = np.reshape(
        spikes[batch],
        [run_dur * np.shape(spikes[batch])[0], np.shape(spikes[batch])[2]],
    )
    batch_raster = np.transpose(batch_spikes)
    mi_graph = ccd_skeleton(batch_raster, batch, coh_lvl=100)
    # make all self-connections 0
    np.fill_diagonal(mi_graph, 0)
    density = calc_density(mi_graph)
    # plot full mi graph
    mi_heatmap = gen_heatmap(
        mi_graph, "Full FN; " + f"{density*100:.1f}" + "% conn", axis=ax[0, 1]
    )

    # calculate top quartile based on abs + weights
    thresh = np.quantile(np.abs(mi_graph[mi_graph > 0]), 0.75)
    top_quartile_mi = np.copy(mi_graph)
    for i in range(0, units):
        for j in range(0, units):
            if np.abs(top_quartile_mi[i, j]) < thresh:
                top_quartile_mi[i, j] = 0
    density = calc_density(top_quartile_mi)
    # plot top quartile
    top_quartile_heatmap = gen_heatmap(
        top_quartile_mi,
        "Top quartile FN; " + f"{density*100:.1f}" + "% conn",
        axis=ax[1, 0],
    )

    # separate e and i
    e_units = int(0.8 * units)
    mi_e_i = np.copy(mi_graph)
    for i in range(0, e_units):
        for j in range(0, units):
            if mi_graph[i, j] < 0:
                mi_e_i[i, j] = 0
    for i in range(e_units, units):
        for j in range(0, units):
            if mi_graph[i, j] > 0:
                mi_e_i[i, j] = 0
    density = calc_density(mi_e_i)
    # plot separate/enforced e and i
    e_i_heatmap = gen_heatmap(
        mi_e_i,
        "FN enforcing + and -; " + f"{density*100:.1f}" + "% conn",
        axis=ax[1, 1],
    )

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle("CCD task - 100% coherence")
    plt.savefig(savefile, dpi=300)
    plt.clf()


def compare_begin_end(
    savedir, start_file, start_batch, mid_file, mid_batch, end_file, end_batch
):
    """TODO: document function"""
    plot_quad_compare(
        start_file, start_batch, savedir + "100_fn_quad_epoch_10.png"
    )
    plot_quad_compare(
        mid_file, mid_batch, savedir + "100_fn_quad_epoch_100.png"
    )
    plot_quad_compare(
        end_file, end_batch, savedir + "100_fn_quad_epoch_200.png"
    )


def gen_heatmap(weights, title, axis, show_value_bounds=True):
    """Generate a heatmap of synaptic weights."""

    weights = weights.copy()
    min_wgt = np.min(weights)
    max_wgt = np.max(weights)
    ticklabels = [" (−)", " 0", " (+)"]

    if min_wgt < 0:
        weights[weights < 0] /= abs(min_wgt)
        if show_value_bounds:
            ticklabels[0] = f" {min_wgt:.4f}"
    if max_wgt > 0:
        weights[weights > 0] /= max_wgt
        if show_value_bounds:
            ticklabels[2] = f" {max_wgt:.4f}"

    heatmap = sns.heatmap(
        weights,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        cbar_kws={
            "ticks": [-1, 0, 1],
            #'label': 'synapse strength'
        },
        xticklabels=[],
        yticklabels=[],
        ax=axis,
    )
    heatmap.collections[0].colorbar.set_ticklabels(ticklabels)

    heatmap.set_title(title)
    heatmap.set_xlabel("target neuron")
    heatmap.set_ylabel("projecting neuron")

    return heatmap
