"""Dynamic (spike) analysis for series of completed experiments"""

# ---- external imports -------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy
import sys
import seaborn as sns
import networkx as nx

from scipy.sparse import load_npz

# ---- internal imports -------------------------------------------------------
sys.path.append("../")
sys.path.append("../../")

from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments
from utils.extools.analyze_structure import get_degrees
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree
from utils.extools.MI import simple_confMI

# ---- global variables -------------------------------------------------------
data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-savey'
task_experiment_string = 'run-batch30-onlytaskloss'
rate_experiment_string = 'run-batch30-onlyrateloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
i_end = 300

n_input = 16
seq_len = 4080

savepath = '/data/results/experiment1/'

e_only = True
positive_only = False
bin = 10


### ADDED (temporarily?) BY CHAD FOR RATE-ONLY GENERATION ###
RECRUIT_PATH = '/data/results/smith7/rateonly-fast-01/recruitment_graphs_bin10_full/'
naive_id = 0
trained_id = 99
save_name='recruit_bin10_full'
coh_lvl = 'coh0'
NUM_EXCI = 240
SAVE_NAME="recruit_bin10_full"
LOAD_DIR = "/data/experiments/"
SAVE_DIR = "/data/results/smith7/rateonly-5"
XSTR = "run-batch30-onlyrateloss"
MI_path = "/data/results/smith7/rateonly-fast-01/MI_graphs_bin10/"

# Paul Tol's colorblind-friendly palette for scientific visualization
COLOR_PALETTE = [
    "#332288",  # indigo
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # cyan
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
]


# =============================================================================
#  Utilities
# =============================================================================

def _nets_from_weights(w, num_exci=240):
    """Return connection networks for a weight matrix.

    Arguments:
        w <NumPy Array> : Two-dimensional array of synaptic weights,
            i.e. an NxN weighted adjacency matrix, where N is the
            number of units in the network. Excitatory units should
            occupy the lower indices of the matrix. For example, a
            100-unit matrix with 80 exitatory units must have those
            units correspond to indices [0, 1, ..., 78, 79] along
            each axis, while the indices [80, 81, ..., 98, 99] will
            correspond to inhibitory units.

        num_exci <int> : Number of excitatory units in `w`.

    Returns:
        Ge : Network of only excitatory connections from `w`.

        Gi : Network of only inhibitory connections from `w`.
    """

    # Select excitatory and inhibitory synapses
    exci_synapses = w[0:num_exci, 0:num_exci]
    inhi_synapses = w[num_exci:, num_exci:]

    # Generate network objects (whole-network, excitatory, inhibitory)
    exci_net = nx.from_numpy_array(exci_synapses, create_using=nx.DiGraph)
    inhi_net = nx.from_numpy_array(inhi_synapses, create_using=nx.DiGraph)

    return exci_net, inhi_net


def smallify(old_data):
    """Format experiment output so it's smaller on disk when saved.

    Item `old_data["cohX"][i][j]` is moved to `new_data["cohX-i-j"].
    Expects `old_data` to be formatted such that
    `old_data["cohX"][i][j]` is the NxN (where N is the num. of units,
    usually 300) matrix corresponding to the jth non-discarded bin for
    the ith trial at coherence level X (0 or 1).

    Example of reading this data back out:
    ```
    old_data = ...
    new_data = smallify(old_data)

    for cohX in new_data["shapes"]:
        for (i, (jmax, N, _)) in enumerate(new_data["shapes"][cohX]):
            for j in range(jmax):
                label = f"{cohX}-{i}-{j}"
                matrix = new_data[label].todense()

                assert(np.array_equal(matrix, old_data[cohX][i][j]))
    ```
    """
    new_data = {"shapes": {}}

	# for each coherence level..
    for cohX in old_data:
        new_data["shapes"][cohX] = []

		# for each set of NxN (N is typically 300) matrices..
        for i, ws in enumerate(old_data[cohX]):
            new_data["shapes"][cohX].append(ws.shape)  # ws.shape == (j, N, N)

			# for each NxN matrix..
            for j, w in enumerate(ws):
                new_label = f"{cohX}-{i}-{j}"
                new_data[new_label] = scipy.sparse.csr_matrix(w.astype(float))

    return new_data


def biggify(smallified_data):
    """Reverse the "smallify" operation."""
    shapes = smallified_data["shapes"][()]
    biggified_data = {}

    for cohX in shapes:
        m = []
        for (i, (jmax, N, _)) in enumerate(shapes[cohX]):
            n = []
            for j in range(jmax):
                label = f"{cohX}-{i}-{j}"
                n.append(np.array(smallified_data[label][()].todense()))
            m.append(np.stack(n))
        biggified_data[cohX] = np.array(m, dtype=object)

    return biggified_data


def safely_make_joint_dirpath(*args, **kwargs):
    """
    Runs `os.path.join()` with the given arguments, and creates a
    directory at the path location, iff it does not already exist.

    Returns the filepath, may create a directory as side effect.
    """
    path = os.path.join(*args, **kwargs)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


# =============================================================================
#  Plot Stuff
# =============================================================================


def plot_avalanche_dist(threshold=True,subsample=False):
    # choose a sample experiment (full run) to examine first
    # use all trials in the initial batch of 30 trials
    # use all trials in the final batch of 30 trials
    # begin by lumping together, but separate by coherence to really say sth
    # go through several choices of what counts as a period of silence
    # count up avalanche sizes (number of spikes in between silences)
    # plot log-log prob(S) vs avalanche size S (# spikes)
    np_dir='/data/experiments/run-batch30-onlytaskloss [2022-10-12 11.54.39]/npz-data'
    #np_dir='/data/experiments/run-batch30-onlyrateloss [2022-10-26 21.32.08]/npz-data'
    #np_dir='/data/experiments/run-batch30-specout-onlinerate0.1-savey [2022-08-15 02.58.19]/npz-data'

    _, ax = plt.subplots(nrows=3, ncols=2)

    naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
    trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))

    # just the e units for now
    naive_spikes = naive_data['spikes'][59]
    trained_spikes = trained_data['spikes'][99]

    silence_sizes = [1,5,10]
    silence_thresh = 0
    thresholds = [20,40,60] # total activity must exceed this percentile of spikes in a timestep to count as part of an avalanche
    sub_size = 60 # number of units in random subsample of each ms to determine silences

    if threshold:
        for th_idx in range(len(thresholds)):
            naive_avalanches = []
            trained_avalanches = []

            for i in range(0,np.shape(naive_spikes)[0]):
                # for each of 30 trials
                Z = naive_spikes[i][:,0:240] # e units only
                X = np.sum(Z,1)
                theta = np.percentile(X,thresholds[th_idx])

                # find where X exceeds theta
                above_thresh_idx = np.squeeze(np.argwhere(X>theta))

                # initialize counter with the number of spikes in the first threshold crossing timebin
                avalanche_counter = X[above_thresh_idx[0]]
                for j in range(1,len(above_thresh_idx)):
                    # if adjacent indices
                    if above_thresh_idx[j]-above_thresh_idx[j-1]==1:
                        # count as part of same avalanche
                        avalanche_counter+=(X[above_thresh_idx[j]]-theta)
                    else:
                        # append existing avalanche to array
                        naive_avalanches.append(avalanche_counter)
                        # start new avalanche counter
                        avalanche_counter=X[above_thresh_idx[j]]-theta

            # repeat for trained spikes
            for i in range(0,np.shape(trained_spikes)[0]):
                Z = trained_spikes[i][:,0:240]
                X = np.sum(Z,1)
                theta = np.percentile(X,thresholds[th_idx])

                above_thresh_idx = np.squeeze(np.argwhere(X>theta))

                avalanche_counter=X[above_thresh_idx[0]]
                for j in range(1,len(above_thresh_idx)):
                    if above_thresh_idx[j]-above_thresh_idx[j-1]==1:
                        avalanche_counter+=(X[above_thresh_idx[j]]-theta)
                    else:
                        trained_avalanches.append(avalanche_counter)
                        avalanche_counter=X[above_thresh_idx[j]]-theta

            # plot
            [naive_s, naive_p] = np.unique(naive_avalanches, return_counts=True)
            # plot p(s) vs s on log-log scale
            ax[th_idx,0].scatter(naive_s, naive_p, s=2)
            ax[th_idx,0].set_xscale("log")
            ax[th_idx,0].set_yscale("log")
            ax[th_idx,0].set_xlabel('S (avalanche size)')
            ax[th_idx,0].set_ylabel('P(S)')
            ax[th_idx,0].set_title('Naive; '+str(thresholds[th_idx])+' %ile threshold')

            [trained_s,trained_p] = np.unique(trained_avalanches, return_counts=True)
            # plot p(s) vs s on log-log scale
            ax[th_idx,1].scatter(trained_s, trained_p, s=2)
            ax[th_idx,1].set_xscale("log")
            ax[th_idx,1].set_yscale("log")
            ax[th_idx,1].set_xlabel('S (avalanche size)')
            ax[th_idx,1].set_ylabel('P(S)')
            ax[th_idx,1].set_title('Trained; '+str(thresholds[th_idx])+' %ile threshold')

        plt.suptitle('E avalanche size dist; task trained')
        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.96)
        save_fname = savepath+'/criticality/avalanches_e_tasktrained_epoch7_thresholds_11.54.39.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

    else:
        for s_idx in range(len(silence_sizes)):

            naive_avalanches = []
            trained_avalanches = []

            for i in range(0,np.shape(naive_spikes)[0]):
                # for each of the 30 naive trials
                trial_spikes = naive_spikes[i][:,0:240] # e units only; not transposing actually works better
                # find where there are zeros across the board
                # for each ms of each trial
                avalanche_counter = 0
                spike_counter = []
                for j in range(np.shape(trial_spikes)[0]):
                    if subsample:
                        # take random subsample of a quarter of the network
                        rand_idx = np.sort(np.random.randint(low=0,high=240,size=sub_size))
                        # append how many units spiked in this ms
                        spike_counter.append(len(np.argwhere(trial_spikes[j][rand_idx]!=0)))
                    else:
                        spike_counter.append(len(np.argwhere(trial_spikes[j]!=0)))
                    # at least silence_size steps into the trial before we can confidently count silences
                    if len(spike_counter)>silence_sizes[s_idx] and np.sum(spike_counter[-(silence_sizes[s_idx]+1):-1])<=silence_thresh:
                        # if not a single unit spiked in this ms and preceding silence_size steps
                        #if len(np.argwhere(trial_spikes[j]==0))==300:
                        if avalanche_counter>0: # previously, there have been spikes
                            naive_avalanches.append(avalanche_counter)
                            # now we reach the end of the avalanche; append previous total
                        avalanche_counter = 0 # either way, avalanche counter should be zeroed
                    else: # at least one unit spiked in this or any of the preceding silence_size steps
                        # count up spikes in current ms slice (may be 0) and add to current avalanche counter
                        if subsample:
                            avalanche_counter += len(np.argwhere(trial_spikes[j][rand_idx]!=0))
                        else:
                            avalanche_counter += len(np.argwhere(trial_spikes[j]!=0))

            [naive_s, naive_p] = np.unique(naive_avalanches, return_counts=True)
            # plot p(s) vs s on log-log scale
            ax[s_idx,0].scatter(naive_s, naive_p, s=2)
            ax[s_idx,0].set_xscale("log")
            ax[s_idx,0].set_yscale("log")
            ax[s_idx,0].set_xlabel('S (avalanche size)')
            ax[s_idx,0].set_ylabel('P(S)')
            ax[s_idx,0].set_title('Naive; '+str(silence_sizes[s_idx])+'ms quiescence')

            # same for trained
            for i in range(0,np.shape(trained_spikes)[0]):
                # for each of the 30 trials
                trial_spikes = trained_spikes[i][:,0:240] # not transposing actually works better
                # find where there are zeros across the board
                # for each ms of each trial
                avalanche_counter = 0
                spike_counter = []
                for j in range(np.shape(trial_spikes)[0]):
                    if subsample:
                        # take random subsample of a quarter of the network
                        rand_idx = np.sort(np.random.randint(low=0,high=240,size=sub_size))
                        # append how many of the randomly subsampled units spiked in this ms
                        spike_counter.append(len(np.argwhere(trial_spikes[j][rand_idx]!=0)))
                    else:
                        spike_counter.append(len(np.argwhere(trial_spikes[j]!=0)))
                    # at least silence_size steps into the trial before we can confidently count silences
                    if len(spike_counter)>silence_sizes[s_idx] and np.sum(spike_counter[-(silence_sizes[s_idx]+1):-1])<=silence_thresh:
                        # if not a single unit spiked in this ms and preceding silence_size steps
                        #if len(np.argwhere(trial_spikes[j]==0))==300:
                        if avalanche_counter>0: # previously, there have been spikes
                            trained_avalanches.append(avalanche_counter)
                            # now we reach the end of the avalanche; append previous total
                        avalanche_counter = 0 # either way, avalanche counter should be zeroed
                    else: # at least one unit spiked in this or any of the preceding silence_size steps
                        # count up spikes in current ms slice (may be 0) and add to current avalanche counter
                        if subsample:
                            avalanche_counter += len(np.argwhere(trial_spikes[j][rand_idx]!=0))
                        else:
                            avalanche_counter += len(np.argwhere(trial_spikes[j]!=0))

            [trained_s,trained_p] = np.unique(trained_avalanches, return_counts=True)
            # plot p(s) vs s on log-log scale
            ax[s_idx,1].scatter(trained_s, trained_p, s=2)
            ax[s_idx,1].set_xscale("log")
            ax[s_idx,1].set_yscale("log")
            ax[s_idx,1].set_xlabel('S (avalanche size)')
            ax[s_idx,1].set_ylabel('P(S)')
            ax[s_idx,1].set_title('Trained; '+str(silence_sizes[s_idx])+'ms quiescence')

        plt.suptitle('E avalanche size dist; input p=0.3; dual trained')
        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.96)
        save_fname = savepath+'/criticality/avalanches_e_dualtrained_epoch50_02.58.19.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

    # do this for several different silence durations and also for training on task and rate alone
    # also ofc for
    # plot log-log distributions for several points along training and note the task and rate losses


def plot_input_channel_rates():
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]

    # for each of 600 trials
    for i in range(0,np.shape(y)[0]):
    # for each of 4080 time steps
    # determine if coherence 1 or 0
        coh0_idx = np.where(y[i]==0)[0]
        coh1_idx = np.where(y[i]==1)[0]
    # take average rates and append
        if len(coh0_idx)>0:
            if not 'coh0_channel_trial_rates' in locals():
                coh0_channel_trial_rates = np.average(x[i][coh0_idx],0)
            else:
                coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(x[i][coh0_idx],0)])

        if len(coh1_idx)>0:
            if not 'coh1_channel_trial_rates' in locals():
                coh1_channel_trial_rates = np.average(x[i][coh1_idx],0)
            else:
                coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(x[i][coh1_idx],0)])

    coh0_rates = np.average(coh0_channel_trial_rates,0)
    coh1_rates = np.average(coh1_channel_trial_rates,0)

    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].hist(coh0_channel_trial_rates,histtype='step')
    ax[0].set_title('responses to coherence 0')
    ax[0].set_xlabel('spike rates')
    ax[1].hist(coh1_channel_trial_rates,bins=50,histtype='step')
    ax[1].set_title('responses to coherence 1')
    ax[1].set_xlabel('spike rates')
    #ax[1,0].hist(late_in[i,:],bins=50,histtype='step')
    #ax[1,1].hist(trained_in[i,:],bins=50,histtype='step')

    plt.suptitle("Input channels' rates")

    # Draw and save
    plt.draw()
    save_fname = savepath+'/specinput0.2/input_rates.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

    return [coh0_rates,coh1_rates]


def output_projection(
    data_dir=LOAD_DIR,
    experiment_string=XSTR,
    recruit_path=RECRUIT_PATH,
    save_name=SAVE_NAME,
    weighted=False,
    coh_lvl="coh1",
    e_end=NUM_EXCI
):
    """TODO: document function

    Looking at only the units that project to the output, find their
    interconnected density and plot their degrees relative to the
    degrees of the rest of the network.
    """

    def _rgraph_avg(rfile):
        """TODO: document function"""
        rdata = np.load(rfile, allow_pickle=True)
        try:
            # Data is not smallified
            rnets = rdata[coh_lvl]
        except KeyError:
            # Data is smallified
            rnets = biggify(rdata)[coh_lvl]

        #return np.mean(rnets, axis=(0, 1))  # can't do axes bc jagged
        m = np.row_stack(rnets)
        return np.mean(m, 0)



    def _multiple_hists(ax, xs, title):
        """TODO: document function"""

        def _hist(ax, x, c, lbl):
            """Preconfigured hist plotter."""
            ax.hist(x=x, color=c, label=lbl, density=True, alpha=0.7)

        # normalized by the total number of units within that
        # population set (1)those that project and 2)those that
        # don't project to output)

        # Generate histograms for each dataset
        cs = ["dodgerblue", "tomato", "mediumseagreen", "darkorange"]
        for ((lbl, x), c) in zip(xs, cs):
            _hist(ax, x, c, lbl)

        # Title and label the plot
        ax.legend()
        ax.set_xlabel("total unweighted degree")
        ax.set_ylabel("density")
        ax.set_title(title)


    # Plot output projection degrees (vs the rest of the network's
    # degrees) for each experiment specified
    data_dirs = get_experiments(data_dir, experiment_string)
    recruit_dirs = [f.path for f in os.scandir(recruit_path)]
    for exp in recruit_dirs:

        # -------------------------------------------------------------
        #  Setup
        # -------------------------------------------------------------

        # Confirm target recruitment graph exists
        final_recruit_file = os.path.join(exp, "991-1000-batch99.npz")
        if not os.path.isfile(final_recruit_file):
            continue

        # Find experiment directory
        exp_string = exp[-8:]  # timestamp (used as unique identifier)
        for dir in data_dirs:
            if (exp_string in dir):
                exp_data_dir = dir
        try:
            npz_data_dir = os.path.join(exp_data_dir, "npz-data")
        except UnboundLocalError:
            continue

        # Load naive and trained data
        naive_data = np.load(os.path.join(npz_data_dir, "1-10.npz"))
        train_data = np.load(os.path.join(npz_data_dir, "991-1000.npz"))

        naive_out = naive_data['tv2.postweights'][0]
        train_out = train_data['tv2.postweights'][99]

        # Instantiate plot
        _, ax = plt.subplots(nrows=1, ncols=2)

        # -------------------------------------------------------------
        #  Calculate Degree Values
        # -------------------------------------------------------------

        # Average all recruitment graphs
        naive_w = _rgraph_avg(os.path.join(exp, "1-10-batch50.npz"))
        train_w = _rgraph_avg(final_recruit_file)

        # Calculate degree values
        naive_degrees = get_degrees(naive_w, weighted)
        train_degrees = get_degrees(train_w, weighted)

        all_naive_degrees = np.add(naive_degrees[1], naive_degrees[0])
        all_train_degrees = np.add(train_degrees[1], train_degrees[0])

        # -------------------------------------------------------------
        #  Find the Degree of Units Which Project to Output
        # -------------------------------------------------------------

        # Find the indices of units which project to output
        naive_e_out_idx = np.argwhere(naive_out[:e_end, :] > 0)[:, 0]
        train_e_out_idx = np.argwhere(train_out[:e_end, :] > 0)[:, 0]
        naive_i_out_idx = np.argwhere(naive_out[e_end:, :] < 0)[:, 0]
        train_i_out_idx = np.argwhere(train_out[e_end:, :] < 0)[:, 0]

        # Find degree values for units which project to output
        naive_e_set_degrees = np.take(
            all_naive_degrees[:e_end], naive_e_out_idx, 0
        )
        naive_i_set_degrees = np.take(
            all_naive_degrees[e_end:], naive_i_out_idx, 0
        )
        train_e_set_degrees = np.take(
            all_train_degrees[:e_end], train_e_out_idx, 0
        )
        train_i_set_degrees = np.take(
            all_train_degrees[e_end:], train_i_out_idx, 0
        )

        # -------------------------------------------------------------
        #  Find the Degree of Units Which Do Not Project to Output
        # -------------------------------------------------------------

        # Find the indices of units which do not project to output
        naive_e_rest_idx = np.argwhere(naive_out[:e_end, :] == 0)[:, 0]
        train_e_rest_idx = np.argwhere(train_out[:e_end, :] == 0)[:, 0]
        naive_i_rest_idx = np.argwhere(naive_out[e_end:, :] == 0)[:, 0]
        train_i_rest_idx = np.argwhere(train_out[e_end:, :] == 0)[:, 0]

        # Find degree values for units which do not project to output
        naive_e_rest_degrees = np.take(
            all_naive_degrees[:e_end], naive_e_rest_idx, 0
        )
        naive_i_rest_degrees = np.take(
            all_naive_degrees[e_end:], naive_i_rest_idx, 0
        )
        train_e_rest_degrees = np.take(
            all_train_degrees[:e_end], train_e_rest_idx, 0
        )
        train_i_rest_degrees = np.take(
            all_train_degrees[e_end:], train_i_rest_idx, 0
        )

        # -------------------------------------------------------------
        #  Generate Plots
        # -------------------------------------------------------------

        # Plot datasets
        _multiple_hists(  # naive
            ax=ax[0],
            xs=[
                ("e projection units", naive_e_set_degrees),
                ("i projection units", naive_i_set_degrees),
                ("e other units", naive_e_rest_degrees),
                ("i other units", naive_i_rest_degrees),
            ],
            title="naive (batch 50)"
        )
        _multiple_hists(  # trained
            ax=ax[1],
            xs=[
                ("e projection units", train_e_set_degrees),
                ("i projection units", train_i_set_degrees),
                ("e other units", train_e_rest_degrees),
                ("i other units", train_i_rest_degrees),
            ],
            title="trained"
        )

        # Label and title overall plot
        plt.suptitle(f"Recruitment graph, {coh_lvl}")

        # Draw and save plot
        plt.draw()
        plt.subplots_adjust(wspace=0.5)
        plt.savefig(os.path.join(
            savepath,
            f"{save_name}_plots",
            f"projectionset",
            f"{exp_string}_ei_recruit_{coh_lvl}_degree.png"
        ))

        # Teardown
        plt.clf()
        plt.close()


def loss_comps_over_all_time(save_name):
    # load in all experiments and plot their losses over time on the same plot
    # plot both rate loss and task loss
    data_dirs = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=5, ncols=1)
    # get your usual experiments first
    for xdir in data_dirs:
        # Get all of a single experiment's losses
        task_losses = []
        rate_losses = []
        density = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            task_losses += data["step_task_loss"].tolist()
            rate_losses += data["step_rate_loss"].tolist()
            for i in range(0,np.shape(data["tv1.postweights"])[0]):
                density.append(calc_density(data["tv1.postweights"][i][0:e_end,0:e_end]))
        # Plot losses for a single experiment
        ax[0].plot(rate_losses)
        ax[1].plot(task_losses)
        ax[2].plot(density)
    # now do for the task only experiments
    task_data_dirs = get_experiments(data_dir, task_experiment_string)
    for xdir in task_data_dirs:
        task_losses = []
        density = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            task_losses += data["step_task_loss"].tolist()
            # for each batch update, calculate e-e graph density
            for i in range(0,np.shape(data["tv1.postweights"])[0]):
                density.append(calc_density(data["tv1.postweights"][i][0:e_end,0:e_end]))
        ax[3].plot(task_losses)
        ax[4].plot(density)

    ax[0].set_title('dual trained network')
    ax[0].set_xlabel('batch')
    ax[0].set_ylabel('rate loss')
    ax[1].set_title('dual trained network')
    ax[1].set_xlabel('batch')
    ax[1].set_ylabel('task loss')
    ax[2].set_title('dual trained network')
    ax[2].set_xlabel('batch')
    ax[2].set_ylabel('e-e density')
    ax[3].set_title('task only trained network')
    ax[3].set_xlabel('batch')
    ax[3].set_ylabel('task loss')
    ax[4].set_title('task only trained network')
    ax[4].set_xlabel('batch')
    ax[4].set_ylabel('e-e density')
    plt.draw()
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(savepath+'/set_plots/losses_dual_vs_task_trained.png')
    plt.clf()
    plt.close()

def pairwise_rate_MI_relation(save_name):
    # for each pair of units, plot their confMI vs sqrt(fr1^2+fr2^2)
    data_dirs = get_experiments(data_dir, experiment_string)
    MI_dirs = [f.path for f in os.scandir(MI_path) if f.is_dir()]

    for exp in MI_dirs:
        # check if final functional graph has been made
        naive_MI_file = exp + '/1-10.npz'
        final_MI_file = exp + '/991-1000.npz'
        if os.path.isfile(final_MI_file) and os.path.isfile(naive_MI_file):
            # plot single experiments
            fig, ax = plt.subplots(nrows=2, ncols=2)
            # get coh0 data and coh1 data
            exp_string = exp[-8:]
            for dir in data_dirs:
                if (exp_string in dir):
                    exp_data_dir = dir
            # load in just the naive batch
            data = np.load(exp_data_dir + '/npz-data/1-10.npz')
            spikes = data['spikes'][50] # designated naive is 50 for now...
            # figure out which spikes correspond to current coherence
            true_y = data['true_y'][50]
            true_y = np.squeeze(true_y)
            rates_0 = []
            rates_1 = []
            for i in range(np.shape(spikes)[0]): # for each trial
                spikes_trial = np.transpose(spikes[i])
                coh_0_idx = np.squeeze(np.where(true_y[i]==0))
                coh_1_idx = np.squeeze(np.where(true_y[i]==1))
                if np.size(coh_0_idx) > 0:
                    coh_0_spikes = spikes_trial[:,coh_0_idx]
                    # calculate rates for each unit
                    rates_0.append(np.mean(coh_0_spikes,1))
                if np.size(coh_1_idx) > 0:
                    coh_1_spikes = spikes_trial[:,coh_1_idx]
                    # calculate rates for each unit
                    rates_1.append(np.mean(coh_1_spikes,1))
            # collapse across trials for a given unit
            unitwise_0_rates = np.mean(rates_0,0)
            unitwise_1_rates = np.mean(rates_1,0)

            # load functional graphs
            MI_data = np.load(naive_MI_file, allow_pickle=True)
            # again, just focusing on single batch as naive rn
            MI_coh0 = MI_data['coh0'][50]
            MI_coh1 = MI_data['coh1'][50]

            # for every pair of e units
            for i in range(0,e_end):
                for j in range(0,e_end):
                    ax[0,0].scatter(np.sqrt(np.add(np.square(unitwise_0_rates[i]),np.square(unitwise_0_rates[j]))), MI_coh0[i,j], s=2, c='blue')
                    ax[0,1].scatter(np.sqrt(np.add(np.square(unitwise_1_rates[i]),np.square(unitwise_1_rates[j]))), MI_coh1[i,j], s=2, c='blue')
            ax[0,0].set_title('naive (batch 50) coh 0')
            ax[0,0].set_xlabel('sqrt(fr1^2 + fr2^2)')
            ax[0,0].set_ylabel('confMI(1,2)')
            ax[0,1].set_title('naive (batch 50) coh 1')
            ax[0,1].set_xlabel('sqrt(fr1^2 + fr2^2)')
            ax[0,1].set_ylabel('confMI(1,2)')

            # and repeat for trained
            data = np.load(exp_data_dir + '/npz-data/991-1000.npz')
            spikes = data['spikes'][99]
            true_y = data['true_y'][99]
            true_y = np.squeeze(true_y)
            rates_0 = []
            rates_1 = []
            for i in range(np.shape(spikes)[0]): # for each trial
                spikes_trial = np.transpose(spikes[i])
                coh_0_idx = np.squeeze(np.where(true_y[i]==0))
                coh_1_idx = np.squeeze(np.where(true_y[i]==1))
                if np.size(coh_0_idx) > 0:
                    coh_0_spikes = spikes_trial[:,coh_0_idx]
                    # calculate rates for each unit
                    rates_0.append(np.mean(coh_0_spikes,1))
                if np.size(coh_1_idx) > 0:
                    coh_1_spikes = spikes_trial[:,coh_1_idx]
                    # calculate rates for each unit
                    rates_1.append(np.mean(coh_1_spikes,1))
            # collapse across trials for a given unit
            unitwise_0_rates = np.mean(rates_0,0)
            unitwise_1_rates = np.mean(rates_1,0)

            # load functional graphs
            MI_data = np.load(final_MI_file, allow_pickle=True)
            # again, just focusing on single batch as naive rn
            MI_coh0 = MI_data['coh0'][99]
            MI_coh1 = MI_data['coh1'][99]

            # for every pair of e units
            for i in range(0,e_end):
                for j in range(0,e_end):
                    ax[1,0].scatter(np.sqrt(np.add(np.square(unitwise_0_rates[i]),np.square(unitwise_0_rates[j]))), MI_coh0[i,j], s=2, c='blue')
                    ax[1,1].scatter(np.sqrt(np.add(np.square(unitwise_1_rates[i]),np.square(unitwise_1_rates[j]))), MI_coh1[i,j], s=2, c='blue')
            ax[1,0].set_title('trained coh 0')
            ax[1,0].set_xlabel('sqrt(fr1^2 + fr2^2)')
            ax[1,0].set_ylabel('confMI(1,2)')
            ax[1,1].set_title('trained coh 1')
            ax[1,1].set_xlabel('sqrt(fr1^2 + fr2^2)')
            ax[1,1].set_ylabel('confMI(1,2)')

            plt.suptitle('excitatory rate MI correspondence')
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            plt.draw()
            save_fname = savepath+'/MI_bin10_plots/'+exp_string+'_e_MI_v_rate.png'
            plt.savefig(save_fname,dpi=300)
            plt.clf()
            plt.close()


def loss_comps_vs_degree(save_name,coh_lvl='coh0',weighted=False):
    # plot loss over time and the average total recruitment degree and synaptic degree
    # for the whole network and then just for the units that project to output
    # so that would be... four plots
    # plot both rate and task losses on same plot
    # plot just the first 100 epochs for now
    data_dirs = get_experiments(data_dir, experiment_string)
    recruit_dirs = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    for exp in recruit_dirs:
        # check if recruitment graph has been made
        final_recruit_file = exp + '/1-10-batch99.npz'
        if os.path.isfile(final_recruit_file):
            fig, ax = plt.subplots(nrows=2, ncols=2) # plot them all together now

            # get original data
            exp_string = exp[-8:]
            for dir in data_dirs:
                if (exp_string in dir):
                    exp_data_dir = dir
            data = np.load(exp_data_dir + '/npz-data/1-10.npz')
            task_loss = data['step_task_loss']
            ax[0,0].plot(task_loss)
            ax[0,0].set_xlabel('batch')
            ax[0,0].set_ylabel('task loss')
            ax[0,0].set_title('Task Loss')
            rate_loss = data['step_rate_loss']
            ax[0,1].plot(rate_loss)
            ax[0,1].set_xlabel('batch')
            ax[0,1].set_ylabel('rate loss')
            ax[0,1].set_title('Rate Loss')
            w = data['tv1.postweights']

            mean_recruit_degrees = []
            # get recruit degrees
            for i in range(100): # over all batches in the first 1-10.npz file
                recruit_file = exp + '/1-10-batch' + str(i) + '.npz'
                recruit_data = np.load(recruit_file, allow_pickle=True)
                recruit = recruit_data[coh_lvl]
                degrees_rec = []
                for i in range(np.shape(recruit)[0]): # for each trial
                    for j in range(np.shape(recruit[i])[0]): # for each timepoint
                        arr = recruit[i][j]
                        # get degrees for each naive unit
                        degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                        # returns [in, out]
                        degrees_rec.append(np.add(degrees[1],degrees[0]))
                mean_recruit_degrees.append(np.mean(degrees_rec))
            ax[1,1].plot(mean_recruit_degrees)
            ax[1,1].set_xlabel('batch')
            ax[1,1].set_ylabel('mean e degree')
            ax[1,1].set_title('Recruitment Graph (Coh 0) Degrees')

            # plot synaptic degrees
            mean_syn_degrees = []
            for i in range(100):
                arr = w[i]
                degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                mean_syn_degrees.append(np.mean(degrees))
            ax[1,0].plot(mean_syn_degrees)
            ax[1,0].set_xlabel('batch')
            ax[1,0].set_ylabel('mean e degree')
            ax[1,0].set_title('Synaptic Graph Degrees')
            plt.subplots_adjust(wspace=0.4, hspace=0.7)

            # Render and save
            plt.draw()
            save_fname = savepath+'/'+save_name+'_plots/lossversus/'+exp_string+'_coh0_e_degree.png'
            plt.savefig(save_fname,dpi=300)

            # Teardown
            plt.clf()
            plt.close()


def synaptic_vs_recruit_weight(save_name, coh_lvl='coh0', e_only=True, weighted=True):
    # for each experiment
    # check recruitment plots generated
    # plot at 0, 10, 100, and 1000 epochs
    data_dirs = get_experiments(data_dir, experiment_string)
    recruit_dirs = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    fig, ax = plt.subplots(nrows=2, ncols=2) # plot them all together now

    for exp in recruit_dirs: # for all experiments
        # check if recruitment graph has been made
        recruit_file_0 = exp + '/1-10-batch1.npz'
        recruit_file_10 = exp + '/1-10-batch10.npz'
        recruit_file_100 = exp + '/1-10-batch99.npz'
        recruit_file_10000 = exp + '/991-1000-batch99.npz'
        if os.path.isfile(recruit_file_10000):

            # get synaptic data
            exp_string = exp[-8:]
            for dir in data_dirs:
                if (exp_string in dir):
                    exp_data_dir = dir
            data = np.load(exp_data_dir + '/npz-data/1-10.npz')
            w_0 = data['tv1.postweights'][0][0:e_end,0:e_end].reshape([e_end*e_end])
            w_10 = data['tv1.postweights'][9][0:e_end,0:e_end].reshape([e_end*e_end])
            w_100 = data['tv1.postweights'][98][0:e_end,0:e_end].reshape([e_end*e_end])
            data = np.load(exp_data_dir + '/npz-data/991-1000.npz')
            w_10000 = data['tv1.postweights'][98][0:e_end,0:e_end].reshape([e_end*e_end])

            """
            # get synaptic degrees
            degrees = get_degrees(w_0[0:e_end,0:e_end],weighted)
            # sum in and out degrees for total degree of each unit
            degrees_w_0 = np.add(degrees[1],degrees[0])
            degrees = get_degrees(w_10[0:e_end,0:e_end],weighted)
            degrees_w_10 = np.add(degrees[1],degrees[0])
            degrees = get_degrees(w_100[0:e_end,0:e_end],weighted)
            degrees_w_100 = np.add(degrees[1],degrees[0])
            degrees = get_degrees(w_10000[0:e_end,0:e_end],weighted)
            degrees_w_10000 = np.add(degrees[1],degrees[0])
            """

            # for recruitment graphs, mean across coherence level
            recruit_data = np.load(recruit_file_0, allow_pickle=True)
            recruit_0 = recruit_data[coh_lvl]
            recruit_data = np.load(recruit_file_10, allow_pickle=True)
            recruit_10 = recruit_data[coh_lvl]
            recruit_data = np.load(recruit_file_100, allow_pickle=True)
            recruit_100 = recruit_data[coh_lvl]
            recruit_data = np.load(recruit_file_10000, allow_pickle=True)
            recruit_10000 = recruit_data[coh_lvl]

            #get mean weights for recruitment graphs
            w_rec_0 = []
            w_rec_10 = []
            w_rec_100 = []
            w_rec_10000 = []
            for i in range(np.shape(recruit_0)[0]): # for each trial
                # mean across time points of the trial for each neuron
                w_rec_0.append(np.mean(recruit_0[i],0)[0:e_end,0:e_end])
            # collapse across trials for each neuron
            w_rec_0 = np.mean(w_rec_0,0).reshape([e_end*e_end])

            for i in range(np.shape(recruit_10)[0]):
                w_rec_10.append(np.mean(recruit_10[i],0)[0:e_end,0:e_end])
            w_rec_10 = np.mean(w_rec_10,0).reshape([e_end*e_end])

            for i in range(np.shape(recruit_100)[0]):
                w_rec_100.append(np.mean(recruit_100[i],0)[0:e_end,0:e_end])
            w_rec_100 = np.mean(w_rec_100,0).reshape([e_end*e_end])

            for i in range(np.shape(recruit_10000)[0]):
                w_rec_10000.append(np.mean(recruit_10000[i],0)[0:e_end,0:e_end])
            w_rec_10000 = np.mean(w_rec_10000,0).reshape([e_end*e_end])

            """
            # get mean degrees for recruitment graphs
            degrees_rec_0 = []
            degrees_rec_10 = []
            degrees_rec_100 = []
            degrees_rec_10000 = []
            for i in range(np.shape(recruit_0)[0]): # for each trial
                for j in range(np.shape(recruit_0[i])[0]): # for each timepoint
                    arr = recruit_0[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    degrees_rec_0.append(np.add(degrees[1],degrees[0]))
            for i in range(np.shape(recruit_10)[0]): # for each trial
                for j in range(np.shape(recruit_10[i])[0]): # for each timepoint
                    arr = recruit_10[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    degrees_rec_10.append(np.add(degrees[1],degrees[0]))
            for i in range(np.shape(recruit_100)[0]): # for each trial
                for j in range(np.shape(recruit_100[i])[0]): # for each timepoint
                    arr = recruit_100[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    degrees_rec_100.append(np.add(degrees[1],degrees[0]))
            for i in range(np.shape(recruit_10000)[0]): # for each trial
                for j in range(np.shape(recruit_10000[i])[0]): # for each timepoint
                    arr = recruit_10000[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    degrees_rec_10000.append(np.add(degrees[1],degrees[0]))
            """

            # now plot correspondingly
            #ax[0,0].scatter(degrees_w_0,np.mean(degrees_rec_0,0), s=2)
            #ax[0,0].set_xlabel('synaptic degree')
            #ax[0,0].set_ylabel('recruitment degree')
            ax[0,0].scatter(w_0,w_rec_0,s=2)
            ax[0,0].set_xlabel('synaptic weight')
            ax[0,0].set_ylabel('recruitment weight')
            ax[0,0].set_title('Epoch 1')
            #ax[0,1].scatter(degrees_w_10,np.mean(degrees_rec_10,0), s=2)
            #ax[0,1].set_xlabel('synaptic degree')
            #ax[0,1].set_ylabel('recruitment degree')
            ax[0,1].scatter(w_10,w_rec_10,s=2)
            ax[0,1].set_xlabel('synaptic weight')
            ax[0,1].set_ylabel('recruitment weight')
            ax[0,1].set_title('Epoch 10')
            #ax[1,0].scatter(degrees_w_100,np.mean(degrees_rec_100,0), s=2)
            #ax[1,0].set_xlabel('synaptic degree')
            #ax[1,0].set_ylabel('recruitment degree')
            ax[1,0].scatter(w_100,w_rec_100,s=2)
            ax[1,0].set_xlabel('synaptic weight')
            ax[1,0].set_ylabel('recruitment weight')
            ax[1,0].set_title('Epoch 100')
            #ax[1,1].scatter(degrees_w_10000,np.mean(degrees_rec_10000,0), s=2)
            #ax[1,1].set_xlabel('synaptic degree')
            #ax[1,1].set_ylabel('recruitment degree')
            ax[1,1].scatter(w_10000,w_rec_10000,s=2)
            ax[1,1].set_xlabel('synaptic weight')
            ax[1,1].set_ylabel('recruitment weight')
            ax[1,1].set_title('Epoch 10000')
    fig.suptitle('Excitatory synaptic vs. recruitment (coh 0) weight')
    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # Draw and save plot
    plt.draw()
    save_fname = savepath+'/'+save_name+'_plots/synvrecruit/weighted_e_only_coh0_quad.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()


# =============================================================================
#  Analyze Vertex Degrees
# =============================================================================

def track_synaptic_high_degree_units_over_time(save_name,weighted=True):
    data_dirs = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # plot all experiments together now
    for exp in data_dirs:
        #data = np.load(exp + '/npz-data/991-1000.npz')
        data = np.load(exp + '/npz-data/1-10.npz')
        naive_w = data['tv1.postweights'][50] # use batch 50 to be consistent with recruitment graphs
        data = np.load(exp + '/npz-data/991-1000.npz')
        trained_w = data['tv1.postweights'][99]

        # get degrees of excitatory units only
        degrees = get_degrees(naive_w[0:e_end,0:e_end],weighted)
        # sum in and out degrees for total degree of each unit
        naive_degrees = np.add(degrees[1],degrees[0])
        degrees = get_degrees(trained_w[0:e_end,0:e_end],weighted)
        trained_degrees = np.add(degrees[1],degrees[0])

        # find the top 15% of each
        threshold_idx = int((1 - 0.15) * np.size(naive_degrees))
        naive_thresh = np.sort(naive_degrees)[threshold_idx]
        naive_max = np.amax(naive_degrees)
        trained_thresh = np.sort(trained_degrees)[threshold_idx]
        trained_max = np.amax(trained_degrees)
        # get the top 15% idx
        naive_top_idx = np.argwhere(naive_degrees>=naive_thresh)
        trained_top_idx = np.argwhere(trained_degrees>=trained_thresh)

        # plot those indices' degrees relative to the top degree
        ax[0].scatter(naive_degrees[naive_top_idx]/naive_max, trained_degrees[naive_top_idx]/trained_max, s=2)
        ax[1].scatter(naive_degrees[trained_top_idx]/naive_max, trained_degrees[trained_top_idx]/trained_max, s=2)

    ax[0].set_title('tracking naive top 15%')
    ax[0].set_xlabel('naive degree / max')
    ax[0].set_ylabel('trained degree / max')
    ax[1].set_title('tracking trained top 15%')
    ax[1].set_xlabel('naive degree / max')
    ax[1].set_ylabel('trained degree / max')

    fig.suptitle('Top 15% of e units with highest synaptic degree')

    # Draw and save plot
    plt.draw()
    save_fname = savepath+'/'+save_name+'_plots/tracking/totaldegree_synaptic_weighted_relative_50.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

def track_high_degree_units_over_time(save_name,weighted=True):
    recruit_dirs = [f.path for f in os.scandir(recruit_path) if f.is_dir()]
    # plot all experiments together
    fig, ax = plt.subplots(nrows=2, ncols=2)
    # for each experiment
    for exp in recruit_dirs:
        # find the units with highest degree in epoch 50
        # find the units with highest degree in epoch 100000
        # do this separately for the two coherence levels

        # check if recruitment graph has been made
        naive_recruit_file = exp + '/1-10-batch50.npz'
        trained_recruit_file = exp + '/991-1000-batch99.npz'
        if os.path.isfile(naive_recruit_file) and os.path.isfile(trained_recruit_file):

            # load recruitment graphs
            naive_recruit_data = np.load(naive_recruit_file, allow_pickle=True)
            naive_recruit_graphs = naive_recruit_data['coh0']

            trained_recruit_data = np.load(trained_recruit_file, allow_pickle=True)
            trained_recruit_graphs = trained_recruit_data['coh0']

            naive_degrees = []
            trained_degrees = []

            for i in range(np.shape(naive_recruit_graphs)[0]): # for each trial
                for j in range(np.shape(naive_recruit_graphs[i])[0]): # for each timepoint
                    arr = naive_recruit_graphs[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    naive_degrees.append(np.add(degrees[1],degrees[0]))

            for i in range(np.shape(trained_recruit_graphs)[0]): # for each trial
                for j in range(np.shape(trained_recruit_graphs[i])[0]): # for each timepoint
                    arr = trained_recruit_graphs[i][j]
                    # get degrees for each trained unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    trained_degrees.append(np.add(degrees[1],degrees[0]))

            # unitwise means
            naive_means = np.mean(naive_degrees,0)
            trained_means = np.mean(trained_degrees,0)

            # find the top 15% of each
            threshold_idx = int((1 - 0.15) * np.size(naive_means))
            naive_thresh = np.sort(naive_means)[threshold_idx]
            naive_max = np.amax(naive_means)
            trained_thresh = np.sort(trained_means)[threshold_idx]
            trained_max = np.amax(trained_means)
            # get the top 15% idx
            naive_top_idx = np.argwhere(naive_means>=naive_thresh)
            trained_top_idx = np.argwhere(trained_means>=trained_thresh)

            # plot those indices' degrees relative to the top degree
            ax[0,0].scatter(naive_means[naive_top_idx]/naive_max, trained_means[naive_top_idx]/trained_max, s=2)
            ax[0,1].scatter(naive_means[trained_top_idx]/naive_max, trained_means[trained_top_idx]/trained_max, s=2)

            # now do for coherence 1
            naive_recruit_graphs = naive_recruit_data['coh1']
            trained_recruit_graphs = trained_recruit_data['coh1']

            naive_degrees = []
            trained_degrees = []

            for i in range(np.shape(naive_recruit_graphs)[0]): # for each trial
                for j in range(np.shape(naive_recruit_graphs[i])[0]): # for each timepoint
                    arr = naive_recruit_graphs[i][j]
                    # get degrees for each naive unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    naive_degrees.append(np.add(degrees[1],degrees[0]))

            for i in range(np.shape(trained_recruit_graphs)[0]): # for each trial
                for j in range(np.shape(trained_recruit_graphs[i])[0]): # for each timepoint
                    arr = trained_recruit_graphs[i][j]
                    # get degrees for each trained unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    trained_degrees.append(np.add(degrees[1],degrees[0]))

            # unitwise means
            naive_means = np.mean(naive_degrees,0)
            trained_means = np.mean(trained_degrees,0)

            # find the top 15% of each
            threshold_idx = int((1 - 0.15) * np.size(naive_means))
            naive_thresh = np.sort(naive_means)[threshold_idx]
            naive_max = np.amax(naive_means)
            trained_thresh = np.sort(trained_means)[threshold_idx]
            trained_max = np.amax(trained_means)
            # get the top 15% idx
            naive_top_idx = np.argwhere(naive_means>=naive_thresh)
            trained_top_idx = np.argwhere(trained_means>=trained_thresh)

            # plot those indices' degrees
            ax[1,0].scatter(naive_means[naive_top_idx]/naive_max, trained_means[naive_top_idx]/trained_max, s=2)
            ax[1,1].scatter(naive_means[trained_top_idx]/naive_max, trained_means[trained_top_idx]/trained_max, s=2)

    ax[0,0].set_title('tracking naive top 15%, coh 0')
    ax[0,0].set_xlabel('naive degree / max')
    ax[0,0].set_ylabel('trained degree / max')
    ax[0,1].set_title('tracking trained top 15%, coh 0')
    ax[0,1].set_xlabel('naive degree / max')
    ax[0,1].set_ylabel('trained degree / max')
    ax[1,0].set_title('tracking naive top 15%, coh 1')
    ax[1,0].set_xlabel('naive degree / max')
    ax[1,0].set_ylabel('trained degree / max')
    ax[1,1].set_title('tracking trained top 15%, coh 1')
    ax[1,1].set_xlabel('naive degree / max')
    ax[1,1].set_ylabel('trained degree / max')

    fig.suptitle('Top 15% of e units with highest weighted degree')
    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # Draw and save plot
    plt.draw()
    save_fname = savepath+'/'+save_name+'_plots/tracking/totaldegree_weighted_relative_50.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

# might as well plot correspondence of synaptic and recruitment degree soon?

def synaptic_degree_rate_correspondence(save_name,weighted=False):
    data_dirs = get_experiments(data_dir, experiment_string)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # plot all experiments together now
    for exp in data_dirs:
        #data = np.load(exp + '/npz-data/991-1000.npz')
        data = np.load(exp + '/npz-data/1-10.npz')
        spikes = data['spikes'][10]
        w = data['tv1.postweights'][9] # single synaptic graph for this batch

        # for each unit, find its in/out degree and its avg rate over all trials
        degrees_e = get_degrees(w[0:e_end,0:e_end],weighted)
        degrees_i = get_degrees(w[e_end:i_end,e_end:i_end],weighted)

        rates = []
        for i in range(np.shape(spikes)[0]): # for each trial
            spikes_trial = np.transpose(spikes[i])
            rates.append(np.mean(spikes_trial,1))
        # collapse across trials for a given unit
        unitwise_rates = np.mean(rates,0)

        ax[0].scatter(unitwise_rates[0:e_end], np.mean(degrees_e,0), s=2)
        ax[1].scatter(unitwise_rates[e_end:i_end], np.mean(degrees_i,0), s=2)

    ax[0].set_title('e total degree')
    ax[0].set_xlabel('avg rate')
    ax[0].set_ylabel('avg total degree')
    ax[1].set_title('i total degree')
    ax[1].set_xlabel('avg rate')
    ax[1].set_ylabel('avg total degree')

    fig.suptitle('Total synaptic degree vs. rate, final batch')

    # Draw and save plot
    plt.draw()
    save_fname = savepath+'/'+save_name+'_plots/ratevdegree/synaptic_10.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

def degree_rate_correspondence(recruit_path,save_name,weighted=False):
# plot the relationship between firing rates and in degree, out degree, and out/in-degree ratio
# for trained recruitment graphs

# get all the experiments
# load in spike data for the final batch
# load in recruitment graph data
# calculate firing rates for whole trial
# match with the degree average for the recruitment graphs

    data_dirs = get_experiments(data_dir, experiment_string)
    recruit_dirs = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    fig, ax = plt.subplots(nrows=2, ncols=2) # plot them all together now

    for exp in recruit_dirs: # for all experiments
        # check if recruitment graph has been made
        recruit_file = exp + '/1-10-batch10.npz'
        if os.path.isfile(recruit_file):

            exp_string = exp[-8:]
            for dir in data_dirs:
                if (exp_string in dir):
                    exp_data_dir = dir

            # load in just the last batch
            data = np.load(exp_data_dir + '/npz-data/1-10.npz')
            spikes = data['spikes'][10] # only taking the final batch
            # figure out which spikes correspond to current coherence
            true_y = data['true_y'][10]
            true_y = np.squeeze(true_y)
            rates_0 = []
            rates_1 = []
            for i in range(np.shape(spikes)[0]): # for each trial
                spikes_trial = np.transpose(spikes[i])
                coh_0_idx = np.squeeze(np.where(true_y[i]==0))
                coh_1_idx = np.squeeze(np.where(true_y[i]==1))
                if np.size(coh_0_idx) > 0:
                    coh_0_spikes = spikes_trial[:,coh_0_idx]
                    # calculate rates for each unit
                    rates_0.append(np.mean(coh_0_spikes,1))
                if np.size(coh_1_idx) > 0:
                    coh_1_spikes = spikes_trial[:,coh_1_idx]
                    # calculate rates for each unit
                    rates_1.append(np.mean(coh_1_spikes,1))
            # collapse across trials for a given unit
            unitwise_0_rates = np.mean(rates_0,0)
            unitwise_1_rates = np.mean(rates_1,0)

            # load recruitment graphs
            recruit_data = np.load(recruit_file, allow_pickle=True)
            recruit_graphs = recruit_data['coh0']

            e_degrees = []
            i_degrees = []

            for i in range(np.shape(recruit_graphs)[0]): # for each trial
                for j in range(np.shape(recruit_graphs[i])[0]): # for each timepoint
                    arr = recruit_graphs[i][j]

                    # get degrees for each e unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    e_degrees.append(np.add(degrees[1],degrees[0]))
                    #e_d_ratio.append(np.nan_to_num(np.divide(degrees[1],degrees[0])))

                    # now do the same for i units
                    degrees = get_degrees(arr[e_end:i_end,e_end:i_end],weighted)
                    i_degrees.append(np.add(degrees[1],degrees[0]))
                    #i_d_ratio.append(np.nan_to_num(np.divide(degrees[1],degrees[0])))
                    # appends over both timepoints and trials, but it doesn't matter bc we are averaging over both for single units anyway

            ax[0,0].scatter(unitwise_0_rates[0:e_end], np.mean(e_degrees,0), s=2)
            ax[0,1].scatter(unitwise_0_rates[e_end:i_end], np.mean(i_degrees,0), s=2)

            # now do for coherence 1
            recruit_graphs = recruit_data['coh1']

            e_degrees = []
            i_degrees = []

            for i in range(np.shape(recruit_graphs)[0]): # for each trial
                for j in range(np.shape(recruit_graphs[i])[0]): # for each timepoint
                    arr = recruit_graphs[i][j]

                    # get degrees for each e unit
                    degrees = get_degrees(arr[0:e_end,0:e_end],weighted)
                    # returns [in, out]
                    e_degrees.append(np.add(degrees[1],degrees[0]))
                    #e_d_ratio.append(np.nan_to_num(np.divide(degrees[1],degrees[0])))

                    # now do the same for i units
                    degrees = get_degrees(arr[e_end:i_end,e_end:i_end],weighted)
                    i_degrees.append(np.add(degrees[1],degrees[0]))
                    #i_d_ratio.append(np.nan_to_num(np.divide(degrees[1],degrees[0])))
                    # appends over both timepoints and trials, but it doesn't matter bc we are averaging over both for single units anyway

            ax[1,0].scatter(unitwise_1_rates[0:e_end], np.mean(e_degrees,0), s=2)
            ax[1,1].scatter(unitwise_1_rates[e_end:i_end], np.mean(i_degrees,0), s=2)

    ax[0,0].set_title('e total degree, coh 0')
    ax[0,0].set_xlabel('average rate')
    ax[0,0].set_ylabel('avg total degree')
    ax[0,1].set_title('i total degree, coh 0')
    ax[0,1].set_xlabel('average rate')
    ax[0,1].set_ylabel('avg total degree')
    ax[1,0].set_title('e total degree, coh 1')
    ax[1,0].set_xlabel('average rate')
    ax[1,0].set_ylabel('avg total degree')
    ax[1,1].set_title('i total degree, coh 1')
    ax[1,1].set_xlabel('average rate')
    ax[1,1].set_ylabel('avg total degree')

    fig.suptitle('Weighted total degree vs. rate, final batch')
    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # Draw and save plot
    plt.draw()
    save_fname = savepath+'/'+save_name+'_plots/ratevdegree/recruit_10.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()


# =============================================================================
#  Analyze Recruitment Networks
# =============================================================================

def plot_recruit_metrics_tribatch(recruit_path, coh_lvl, save_name):
    """TODO: document function"""

    if coh_lvl == "coh0":
        coh_str = "15% coherence"
    elif coh_lvl == "coh1":
        coh_str = "100% coherence"

    # get recruitment graph experiment files
    experiment_paths = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    for exp in experiment_paths:
        exp_string = exp[-8:]

        batch_strings = [
            exp+'/1-10-batch1.npz',
            exp+'/1-10-batch10.npz',
            exp+'/1-10-batch99.npz',
            exp+'/991-1000-batch99.npz'
        ]
        batch_names = [
            'batch 1',
            'batch 10',
            'batch100',
            'batch 10000'
        ]
        batch_colors = [
            'yellowgreen',
            'mediumseagreen',
            'darkturquoise',
            'dodgerblue'
        ]

        # Setup
        plt.figure()

        # Plot data
        for i in range(4): # for each of the three batches
            data = np.load(batch_strings[i], allow_pickle=True)
            coh = data[coh_lvl]
            cc_e = []

            for trial in range(np.shape(coh)[0]):
                # for the timesteps in this trial
                for time in range(np.shape(coh[trial])[0]):

                    #w_e.append(np.mean(coh[trial][time][0:e_end,0:e_end]))
                    #dens_e.append(calc_density(coh[trial][time][0:e_end,0:e_end]))
                    #recip_e = reciprocity(coh[trial][time][0:e_end,0:e_end])

                    # still does not support negative weights, so take abs
                    # convert from object to float array

                    arr = np.abs(coh[trial][time])
                    float_arr = np.vstack(arr[:, :]).astype(np.float)
                    Ge = nx.from_numpy_array(float_arr[0:e_end,0:e_end],create_using=nx.DiGraph)
                    #Gi = nx.from_numpy_array(float_arr[e_end:i_end,e_end:i_end],create_using=nx.DiGraph)
                    cc_e.append(nx.average_clustering(Ge,nodes=Ge.nodes,weight='weight'))
                    #cc_i.append(nx.average_clustering(Gi,nodes=Gi.nodes,weight='weight'))
            # PLOT
            cc_arr = np.array(cc_e)
            if len(cc_arr[cc_arr>0])>0:
                sns.histplot(
                    data=cc_arr[cc_arr>0],
                    color=batch_colors[i],
                    label=batch_names[i],
                    stat="density",
                    alpha=0.5,
                    kde=True,
                    edgecolor="white",
                    linewidth=0.5,
                    line_kws=dict(color="black", alpha=0.5, linewidth=1.5),
                )

        # Title and label
        plt.xlabel("absolute weighted clustering coefficient")
        plt.ylabel("density")
        plt.title("Clustering of e units in recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_e_tribatch_clustering.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

def plot_recruit_metrics_single_update(recruit_path,coh_lvl,save_name):
    # plot weights, density, and weighted clustering of just epoch 0 and epoch 1000
    # plot as distributions for comparison
    # do first column for naive
    # 1: e and i distribution for weights
    # 2: e and i distribution for density
    # 3: e and i distribution for clustering
    # repeat in second column for trained

    if coh_lvl == 'coh0':
        coh_str = '15% coherence'
    elif coh_lvl == 'coh1':
        coh_str = '100% coherence'

    # get recruitment graph experiment files
    experiment_paths = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    for exp in experiment_paths:
        exp_string = exp[-8:]

        w_e = []
        w_i = []

        dens_e = []
        dens_i = []

        cc_e = []
        cc_i = []

        naive_batch_str = exp+'/1-10-batch1.npz'
        data = np.load(naive_batch_str, allow_pickle=True)
        coh = data[coh_lvl]

        for trial in coh:
            # for the timesteps in this trial
            for timestep in trial:

                w_e.append(np.mean(timestep[0:e_end,0:e_end]))
                dens_e.append(calc_density(timestep[0:e_end,0:e_end]))
                #recip_e = reciprocity(timestep[0:e_end,0:e_end])

                w_i.append(np.mean(timestep[e_end:i_end,e_end:i_end]))
                dens_i.append(calc_density(timestep[e_end:i_end,e_end:i_end]))
                #recip_i = reciprocity(timestep[e_end:i_end,e_end:i_end])

                # still does not support negative weights, so take abs
                # convert from object to float array
                arr = np.abs(timestep)
                float_arr = np.vstack(arr[:, :]).astype(np.float)
                Ge = nx.from_numpy_array(
                    float_arr[0:e_end,0:e_end],
                    create_using=nx.DiGraph
                )
                Gi = nx.from_numpy_array(
                    float_arr[e_end:i_end,e_end:i_end],
                    create_using=nx.DiGraph
                )
                cc_e.append(nx.average_clustering(
                    Ge, nodes=Ge.nodes, weight='weight'
                ))
                cc_i.append(nx.average_clustering(
                    Gi, nodes=Gi.nodes, weight='weight'
                ))

            """
            # collect and average over timesteps
            # across all timesteps of a trial, get the mean metric value
            w_e.append(np.mean(time_w_e))
            w_i.append(np.mean(time_w_i))

            dens_e.append(np.mean(time_dens_e))
            dens_i.append(np.mean(time_dens_i))

            cc_e.append(np.mean(time_cc_e))
            cc_i.append(np.mean(time_cc_i))
            """

        # plot as 3 separate histplots

        # PLOT WEIGHTS
        plt.figure()
        def _histplot(x, c, lbl):
            """Preconfigured histplot plotter."""
            sns.histplot(
                # stuff what varies
                data=x,
                color=c,
                label=lbl,

                # stuff what stays the same
                stat="density",
                bins=30,
                alpha=0.5,
                kde=True,
                edgecolor="white",
                linewidth=0.5,
                line_kws={
                    "color": "black",
                    "alpha": 0.5,
                    "linewidth": 1.5
                }
            )
        _histplot(w_e, "blue", "within e units")
        _histplot(w_i, "red", "within i units")

        plt.xlabel("weights")
        plt.ylabel("density")
        plt.title("Weight dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1_weights.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT DENSITY
        plt.figure()

        _histplot(dens_e, "blue", "within e units")
        _histplot(dens_i, "red", "within i units")

        plt.xlabel("connection density")
        plt.ylabel("density")
        plt.title("Density dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1_density.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT CLUSTERING
        # PLOT DENSITY
        plt.figure()
        _histplot(cc_e, "blue", "within e units")
        _histplot(cc_i, "red", "within i units")

        plt.xlabel("absolute weighted clustering")
        plt.ylabel("density")
        plt.title("Clustering dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1_clustering.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

def plot_recruit_metrics_naive_trained(recruit_path,coh_lvl,save_name):
    # plot weights, density, and weighted clustering of just epoch 0 and epoch 1000
    # plot as distributions for comparison
    # do first column for naive
    # 1: e and i distribution for weights
    # 2: e and i distribution for density
    # 3: e and i distribution for clustering
    # repeat in second column for trained

    if coh_lvl == 'coh0':
        coh_str = '15% coherence'
    elif coh_lvl == 'coh1':
        coh_str = '100% coherence'

    # get recruitment graph experiment files
    experiment_paths = [f.path for f in os.scandir(recruit_path) if f.is_dir()]

    for exp in experiment_paths:
        exp_string = exp[-8:]

        w_e = []
        w_i = []

        dens_e = []
        dens_i = []

        cc_e = []
        cc_i = []

        naive_batch_str = exp+'/1-10-batch0.npz'
        data = np.load(naive_batch_str, allow_pickle=True)
        coh = data[coh_lvl]

        for trial in coh:
            # for the timesteps in this trial
            for timestep in trial:

                w_e.append(np.mean(timestep[0:e_end,0:e_end]))
                dens_e.append(calc_density(timestep[0:e_end,0:e_end]))
                #recip_e = reciprocity(timestep[0:e_end,0:e_end])

                w_i.append(np.mean(timestep[e_end:i_end,e_end:i_end]))
                dens_i.append(calc_density(timestep[e_end:i_end,e_end:i_end]))
                #recip_i = reciprocity(timestep[e_end:i_end,e_end:i_end])

                # still does not support negative weights, so take abs
                # convert from object to float array
                arr = np.abs(timestep)
                float_arr = np.vstack(arr[:, :]).astype(np.float)

                Ge, Gi = _nets_from_weights(float_arr, e_end)
                cc_e.append(nx.average_clustering(
                    Ge, nodes=Ge.nodes, weight='weight'
                ))
                cc_i.append(nx.average_clustering(
                    Gi, nodes=Gi.nodes, weight='weight'
                ))

            """
            # collect and average over timesteps
            # across all timesteps of a trial, get the mean metric value
            w_e.append(np.mean(time_w_e))
            w_i.append(np.mean(time_w_i))

            dens_e.append(np.mean(time_dens_e))
            dens_i.append(np.mean(time_dens_i))

            cc_e.append(np.mean(time_cc_e))
            cc_i.append(np.mean(time_cc_i))
            """

        # plot as 3 separate histplots

        # PLOT WEIGHTS
        plt.figure()
        def _histplot(x, c, lbl):
            """Preconfigured histplot plotter"""

            sns.histplot(
                # stuff what stays the same
                data=x,
                color=c,
                label=lbl,

                # stuff what changes
                stat="density",
                bins=30,
                alpha=0.5,
                kde=True,
                edgecolor="white",
                linewidth=0.5,
                line_kws={
                    "color": "black",
                    "alpha": 0.5,
                    "linewidth": 1.5
                }
            )
        _histplot(w_e, "blue", "within e units")
        _histplot(w_i, "red", "within i units")

        plt.xlabel("weights")
        plt.ylabel("density")
        plt.title("Weight dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch0_weights.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT DENSITY
        plt.figure()
        _histplot(dens_e, "blue", "within e units")
        _histplot(dens_i, "red", "within i units")

        plt.xlabel("connection density")
        plt.ylabel("density")
        plt.title("Density dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch0_density.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT CLUSTERING
        # PLOT DENSITY
        plt.figure()
        _histplot(cc_e, "blue", "within e units")
        _histplot(cc_i, "red", "within i units")

        plt.xlabel("absolute weighted clustering")
        plt.ylabel("density")
        plt.title("Clustering dist of naive recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch0_clustering.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # NOW TRAINED DATA (EPOCH 1000)
        trained_batch_str = exp+'/991-1000-batch99.npz'
        data = np.load(trained_batch_str, allow_pickle=True)
        coh = data[coh_lvl]

        for trial in range(np.shape(coh)[0]):
            # for the timesteps in this trial
            for time in range(np.shape(coh[trial])[0]):

                w_e.append(np.mean(coh[trial][time][0:e_end,0:e_end]))
                dens_e.append(calc_density(coh[trial][time][0:e_end,0:e_end]))
                #recip_e = reciprocity(coh[trial][time][0:e_end,0:e_end])

                w_i.append(np.mean(coh[trial][time][e_end:i_end,e_end:i_end]))
                dens_i.append(calc_density(coh[trial][time][e_end:i_end,e_end:i_end]))
                #recip_i = reciprocity(coh[trial][time][e_end:i_end,e_end:i_end])

                # still does not support negative weights, so take abs
                # convert from object to float array
                arr = np.abs(coh[trial][time])
                float_arr = np.vstack(arr[:, :]).astype(np.float)

                Ge, Gi = _nets_from_weights(float_arr, e_end)

                cc_e.append(nx.average_clustering(
                    Ge, nodes=Ge.nodes, weight='weight'
                ))
                cc_i.append(nx.average_clustering(
                    Gi, nodes=Gi.nodes, weight='weight'
                ))

        # plot as 3 separate histplots

        # PLOT WEIGHTS
        plt.figure()
        _histplot(w_e, "blue", "within e units")
        _histplot(w_i, "red", "within i units")

        plt.xlabel("weights")
        plt.ylabel("density")
        plt.title("Weight dist of trained recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1000_weights.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT DENSITY
        plt.figure()
        _histplot(dens_e, "blue", "within e units")
        _histplot(dens_i, "red", "within i units")

        plt.xlabel("connection density")
        plt.ylabel("density")
        plt.title("Density dist of trained recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1000_density.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        # PLOT CLUSTERING
        # PLOT DENSITY
        plt.figure()
        _histplot(cc_e, "blue", "within e units")
        _histplot(cc_i, "red", "within i units")

        plt.xlabel("absolute weighted clustering")
        plt.ylabel("density")
        plt.title("Clustering dist of trained recruitment graph, "+coh_str)
        plt.legend()

        # Draw and save plot
        plt.draw()
        plt_name = savepath+save_name+'_'+exp_string+'_'+coh_lvl+'_epoch1000_clustering.png'
        plt.savefig(plt_name, dpi=300)

        # Teardown
        plt.clf()
        plt.close()

def plot_recruit_metrics(recruit_path, epoch_id, coh_lvl, save_name):
    """TODO: document function"""

    # -----------------------------------------------------------------
    #  Utility Functions
    # -----------------------------------------------------------------

    def _trialwise_weights(w):
        """Compute the trialwise mean of a dataset's weights.

        Arguments:
            w <NumPy Array> : Matrix whose axes are, in order: trial,
                timestep, projecting unit, target unit. The last two
                dimensions are equivalent (indicating a weighted
                adjacency matrix for synaptic connections).

        Returns:
            weights <NumPy Array> : Matrix of mean weights of w with
                shape (num_trials,)
        """
        non_trial_axes = tuple(range(1, w.ndims))
        return np.mean(w, axis=non_trial_axes)


    def _trialwise_density(w):
        """TODO: document function

        Arguments:
            w <NumPy Array> : Matrix whose axes are, in order: trial,
                timestep, projecting unit, target unit. The last two
                dimensions are equivalent (indicating a weighted
                adjacency matrix for synaptic connections).

        Returns:
            densities <NumPy Array> : Matrix of densities of w with
                shape (num_trials,)
        """
        # The number of units is also the length of the diagonals of
        # the adjacency matrices. Because self-connections (the values
        # along the diagonal) are discounted, we subtract this value
        # from the number of connections possible in a synaptic graph.
        num_units = w.shape[-1]
        possible_nonzeros = num_units * num_units - num_units

        # Reduce each synaptic graph at each timestep to the number of
        # nonzero values
        #
        # Axes (-2, -1) indicate the synaptic graphs.
        #
        # Shape: (num_trials, num_timesteps)
        actual_nonzeros = np.count_nonzero(w, axis=(-2, -1))

        # Mean density of the synaptic graph over the course of the
        # trial
        #
        # Shape: (num_trials,)
        return np.mean(actual_nonzeros / possible_nonzeros, axis=1)


    def _trialwise_clustering(coh):
        """TODO: document function"""

        # Compute mean and stddev for clustering
        trial_cc_e = []
        trial_cc_i = []
        for trial in coh:
            # for the timesteps in this trial
            time_cc_e = []
            time_cc_i = []

            for timestep in trial:

                #recip_e = reciprocity(timestep[0:e_end,0:e_end])
                #recip_i = reciprocity(timestep[e_end:i_end,e_end:i_end])

                # still does not support negative weights, so take abs
                # convert from object to float array
                arr = np.abs(timestep)
                float_arr = np.vstack(arr[:, :]).astype(np.float)

                Ge, Gi = _nets_from_weights(float_arr, e_end)

                time_cc_e.append(nx.average_clustering(
                    Ge, nodes=Ge.nodes, weight='weight'
                ))
                time_cc_i.append(nx.average_clustering(
                    Gi, nodes=Gi.nodes, weight='weight'
                ))

            # collect and average over timesteps
            # across all timesteps of a trial, get the mean metric value
            trial_cc_e.append(np.mean(time_cc_e))
            trial_cc_i.append(np.mean(time_cc_i))

        return trial_cc_e, trial_cc_i


    def _plot_mean_and_stddev(ax, mean, stddev):
        """TODO: document function"""
        ax.plot(mean)
        ax.fill_between(
            mean - stddev,
            mean + stddev,
            alpha=0.5
        )

    # -----------------------------------------------------------------
    #  Setup
    # -----------------------------------------------------------------

    # Get data file names
    experiment_paths = [
        f.path for f in os.scandir(recruit_path) if f.is_dir()
    ]
    data_files = filenames(num_epochs, epochs_per_file)

    # Get strings for naming
    epoch_string = data_files[epoch_id][:-4]

    if coh_lvl == "coh0":
        coh_str = " 15% coherence"
    elif coh_lvl == "coh1":
        coh_str = " 100% coherence"

    # Instantiate plot
    fig, ax = plt.subplots(nrows=3, ncols=2)

    # -----------------------------------------------------------------
    #  Calculate Various Network Statistics
    # -----------------------------------------------------------------

    for exp in experiment_paths:
        exp_string = exp[-8:]

        # each of these will be plotted per experiment
        stats = {
            "w_ee_mean": [],     # weights, excitatory mean
            "w_ee_std": [],      # weights, excitatory stddev
            "w_ii_mean": [],     # weights, inhibitory mean
            "w_ii_std": [],      # weights, inhibitory stddev
            "dens_ee_mean": [],  # density, excitatory mean
            "dens_ee_std": [],   # density, excitatory stddev
            "dens_ii_mean": [],  # density, inhibitory mean
            "dens_ii_std": [],   # density, inhibitory stddev
            "cc_ee_mean": [],    # clustering, excitatory mean
            "cc_ee_std": [],     # clustering, excitatory stddev
            "cc_ii_mean": [],    # clustering, inhibitory mean
            "cc_ii_std": [],     # clustering, inhibitory stddev
        }
        for batch in range(100):  # for each batch update..

            # Read data from disk
            batch_string = os.path.join(
                exp, f"{epoch_string}-batch{batch}.npz"
            )
            data = np.load(batch_string, allow_pickle=True)
            coh = data[coh_lvl]

            # Separate out excitatory and inhibitory connections
            exci_data = coh[..., :e_end, :e_end]
            inhi_data = coh[..., e_end:, e_end:]

            # ---------------------------------------------------------
            #  Calculate Weight Statistics
            # ---------------------------------------------------------

            # Get mean weight of every trial
            exci_weights = _trialwise_weights(exci_data)
            inhi_weights = _trialwise_weights(inhi_data)

            # Compute mean value of trials' mean weights
            stats["w_ee_mean"].append(np.mean(exci_weights))
            stats["w_ii_mean"].append(np.mean(inhi_weights))

            # Compute standard deviation of trials' mean weights
            stats["w_ee_std"].append(np.std(exci_weights))
            stats["w_ii_std"].append(np.std(inhi_weights))

            # ---------------------------------------------------------
            #  Calculate Density Statistics
            # ---------------------------------------------------------

            # Get mean density of every trial
            exci_density = _trialwise_density(exci_data)
            inhi_density = _trialwise_density(inhi_data)

            # Compute mean value of trials' mean densities
            stats["dens_ee_mean"].append(np.mean(exci_density))
            stats["dens_ii_mean"].append(np.mean(inhi_density))

            # Compute standard deviation of trials' mean densities
            stats["dens_ee_std"].append(np.std(exci_density))
            stats["dens_ii_std"].append(np.std(inhi_density))

            # ---------------------------------------------------------
            #  Calculate Clustering Statistics
            # ---------------------------------------------------------

            # Get mean clustering coefficient of every trial
            exci_clustering, inhi_clustering = _trialwise_clustering(
                exci_data
            )

            # Compute mean value of trials' mean CCs
            stats["cc_ee_mean"].append(np.mean(exci_clustering))
            stats["cc_ii_mean"].append(np.mean(inhi_clustering))

            # Compute standard deviation of trial's mean CCs
            stats["cc_ee_std"].append(np.std(exci_clustering))
            stats["cc_ee_std"].append(np.std(inhi_clustering))

        # -------------------------------------------------------------
        #  Save Calculated Statistics
        # -------------------------------------------------------------

        savefile = os.path.join(
            savepath,
            f"{save_name}_{exp_string}_{coh_lvl}_epoch{epoch_string}"
        )
        np.savez_compressed(savefile, **stats)

        # -------------------------------------------------------------
        #  Plot Calculated Statistics (per file)
        # -------------------------------------------------------------

        _plot_mean_and_stddev(      # plot e. weight statistics
            ax[0, 0],
            stats["w_ee_mean"],
            stats["w_ee_std"]
        )
        _plot_mean_and_stddev(      # plot i. weight statistics
            ax[0, 1],
            stats["w_ii_mean"],
            stats["w_ii_std"]
        )
        _plot_mean_and_stddev(      # plot e. density statistics
            ax[1, 0],
            stats["dens_ee_mean"],
            stats["dens_ee_std"]
        )
        _plot_mean_and_stddev(      # plot i. density statistics
            ax[1, 1],
            stats["dens_ii_mean"],
            stats["dens_ii_std"]
        )
        _plot_mean_and_stddev(      # plot e. clustering statistics
            ax[2, 0],
            stats["cc_ee_mean"],
            stats["cc_ee_std"]
        )
        _plot_mean_and_stddev(      # plot i. clustering statistics
            ax[2, 1],
            stats["cc_ii_mean"],
            stats["cc_ii_std"]
        )

    # -----------------------------------------------------------------
    #  Plot Calculated Statistics (per experiment)
    # -----------------------------------------------------------------

    ax[0, 0].set_title("e-to-e weights")
    ax[0, 0].set_xlabel("batch")
    ax[0, 0].set_ylabel("weight")
    ax[1, 0].set_title("e-to-e density")
    ax[1, 0].set_xlabel("batch")
    ax[1, 0].set_ylabel("density")
    ax[2, 0].set_title("e-to-e clustering")
    ax[2, 0].set_xlabel("batch")
    ax[2, 0].set_ylabel("abs weighted clustering")

    ax[0, 1].set_title("i-to-i weights")
    ax[0, 1].set_xlabel("batch")
    ax[0, 1].set_ylabel("weight")
    ax[1, 1].set_title("i-to-i density")
    ax[1, 1].set_xlabel("batch")
    ax[1, 1].set_ylabel("density")
    ax[2, 1].set_title("i-to-i clustering")
    ax[2, 1].set_xlabel("batch")
    ax[2, 1].set_ylabel("abs weighted clustering")

    if epoch_id == 0:
        title_str = "Naive (first 10 epochs) recruitment graphs,"
    elif epoch_id == 99:
        title_str = "Trained (last 10 epochs) recruitment graphs,"

    fig.suptitle(title_str + coh_str)

    # Draw and save plot
    plt.draw()
    savefile = os.path.join(
        savepath,
        f"{save_name}_{coh_lvl}_epoch{epoch_string}.png"
    )
    plt.savefig(savefile, dpi=300)

    # Teardown
    plt.clf()
    plt.close()


# =============================================================================
# ...
# =============================================================================

def plot_fn_quad_metrics(load_saved=True):
    # even though it's so simple (just 4 values per xdir),
    # it'll be useful to compare across xdirs and training
    metric_data_file = os.path.join(
        savepath, "set_fn_quad_metrics_withnegative.npy"
    )
    if not load_saved:
        metric_mat = calculate_fn_quad_metrics()
        np.save(metric_data_file, metric_mat)
    else:
        metric_mat = np.load(metric_data_file)
    # shaped [4-metrics,number-of-experiments,2-coherence-levels,4-epochs]
    labels = ["average weight", "density", "reciprocity", "clustering"]
    epochs = [0, 10, 100, 1000]
    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax = ax.flatten()
    ax_idx = [0, 2, 4, 6]
    for i in range(4):  # for each of the four metrics
        for j in range(np.shape(metric_mat)[0]):  # for each experiment
            # plot for both coherence levels
            ax[ax_idx[i]].plot(epochs, metric_mat[i][j][0])
            ax[ax_idx[i] + 1].plot(epochs, metric_mat[i][j][1])
        ax[ax_idx[i]].set_title("coherence level 0")
        ax[ax_idx[i] + 1].set_title("coherence level 1")
        ax[ax_idx[i]].set_ylabel(labels[i])
        ax[ax_idx[i] + 1].set_ylabel(labels[i])
    for i in range(8):
        ax[i].set_xlabel("epoch")
    fig.suptitle("functional graph metrics for just 4 epochs")

    # Draw and save plot
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=1.5)
    plt.savefig(
        os.path.join(savepath, "set_fn_quad_metrics_withnegative.png"),
        dpi=300,
    )

    # Teardown
    plt.clf()
    plt.close()


def calculate_fn_quad_metrics(e_only, positive_only=False):
    # for now, metrics are mean weight, density, reciprocity, clustering
    # each metric is sized [number-of-experiments,2-coherence-levels,4-epochs,]
    w_mat = []
    dens_mat = []
    recips_mat = []
    ccs_mat = []
    experiments = get_experiments(data_dir, experiment_string)
    for xdir in experiments:
        xdir_quad_fn = generate_quad_fn(xdir, e_only, positive_only)
        n_epochs = np.shape(xdir_quad_fn)[0]
        ws = np.zeros([2, n_epochs])
        dens = np.zeros([2, n_epochs])
        recips = np.zeros([2, n_epochs])
        ccs = np.zeros([2, n_epochs])
        for i in range(
            np.shape(xdir_quad_fn)[1]
        ):  # for each of the 2 coherence levels:
            for j in range(n_epochs):  # for each of the four epochs
                # flipping indices from epoch,coherence to coherence,epoch
                # calculate mean weight
                ws[i][j] = np.average(xdir_quad_fn[j][i])
                # calculate density
                dens[i][j] = calc_density(xdir_quad_fn[j][i])
                # calculate reciprocity
                recips[i][j] = reciprocity(xdir_quad_fn[j][i])
                # calculate weighted clustering coefficient
                # nx clustering is still not supported for negative values (it will create complex cc values)
                # so, use absolute values for now
                G = nx.from_numpy_array(
                    np.abs(xdir_quad_fn[j][i]), create_using=nx.DiGraph
                )
                ccs[i][j] = nx.average_clustering(
                    G, nodes=G.nodes, weight="weight"
                )
        w_mat.append(ws)
        dens_mat.append(dens)
        recips_mat.append(recips)
        ccs_mat.append(ccs)
    return [w_mat, dens_mat, recips_mat, ccs_mat]


def plot_fn_w_dist_experiments():
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

        xdir_quad_fn = generate_quad_fn(xdir, e_only, positive_only)
        # sized [4,2,240,240]

        for i in range(np.shape(xdir_quad_fn)[0]):
            plt.figure()
            # plot coherence level 0 fn weights
            sns.histplot(
                data=np.ravel(xdir_quad_fn[i][0]),
                bins=30,
                color="blue",
                label="coherence 0",
                stat="density",
                alpha=0.5,
                kde=True,
                edgecolor="white",
                linewidth=0.5,
                line_kws=dict(color="black", alpha=0.5, linewidth=1.5),
            )
            # plot coherence level 1 fn weights
            sns.histplot(
                data=np.ravel(xdir_quad_fn[i][1]),
                bins=30,
                color="red",
                label="coherence 1",
                stat="density",
                alpha=0.5,
                kde=True,
                edgecolor="white",
                linewidth=0.5,
                line_kws=dict(color="black", alpha=0.5, linewidth=1.5),
            )
            plt.xlabel("functional weight distribution")
            plt.ylabel("density")
            plt.title(plt_string[i])
            plt.legend()

            # Draw and save plot
            plt.draw()
            plt_name = plt_string[i] + "_fn_wdist.png"
            plt.savefig(os.path.join(savepath, exp_path, plt_name), dpi=300)

            # Teardown
            plt.clf()
            plt.close()


"""
def generate_quad_fn(xdir, e_only, positive_only):
    # simply generate four functional networks for a given xdir within experiments
    # can then be used to make quick plots
    # epochs = ['epoch0','epoch10','epoch100','epoch1000']
    xdir_quad_fn = []
    data_files = []
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/1-10.npz'))
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/91-100.npz'))
    data_files.append(os.path.join(data_dir, xdir, 'npz-data/991-1000.npz'))
    spikes = []
    true_y = []
    data = np.load(data_files[0])
    # batch 0 (completely naive) spikes
    spikes.append(data['spikes'][0])
    true_y.append(data['true_y'][0])
    for i in range(3): # load other spikes
        data = np.load(data_files[i])
        spikes.append(data['spikes'][99])
        true_y.append(data['true_y'][99])
    for i in range(4):
        [fn_coh0, fn_coh1] = generate_batch_ccd_functional_graphs(spikes[i],true_y[i],e_only,positive_only)
        xdir_quad_fn.append([fn_coh0, fn_coh1])
    # returns [epoch,coherence,FN]
    # sized [4,2,240,240]
    return xdir_quad_fn
"""


# =============================================================================
# Graph Generation
# =============================================================================

def batch_recruitment_graphs(w, fn, spikes, trialends, threshold):
    """TODO: document function

    Arguments:
        w: synaptic graph (300x300, usually)
        fn: functional network for a particular batch and coherence
            level
        spikes: binned spikes for 30 trials of a particular batch and
            coherence level
        trialends = indices for which spikes reach a discontinuity. It
            follows the returned recruitment graphs will be have shape
            (trial(segment), timestep, pre units, post units)
        threshold: 0.25 meaning we only use the top quartile of
            functional weights; otherwise we have a fully dense graph
    """

    def threshold_fnet(fnet, thr, copy=True):
        """Mask lower-magnitude values in a functional network.

        Masks all entries in functional network `fnet` whose absolute
        value falls outside specificed quantile `thr` of absolute
        values in said network.

        Arguments:
            fnet: functional network (NumPy array)
            thr: floating point value in [0, 1] indicating percentile
                cutoff (e.g. thr=0.25 keeps only the top quartile)

        Returns:
            Masked NumPy array masking all values in `fnet` outside the
            specified quantile. If `copy` is `True`, this array will be
            a deep copy. Otherwise, it will maintain a reference to
            `fnet`
        """

        # Determine which entries to mask
        if thr == 0:
            mask = True  # mask all values (0th percentile)
        elif thr == 1:
            mask = False  # mask no values (100th percentile)
        else:
            magnitude_fnet = np.abs(fnet)
            mask = magnitude_fnet < np.quantile(magnitude_fnet, thr)

        # Mask the array
        return ma.masked_array(fnet, mask, copy=copy, fill_value=0)


    def firing_buddy_mask(z, m):
        """TODO: document function"""

        # Binary matrix where the value at i,j indicates if neurons
        # i and j did (1) or did not (0) *both* spike at time t
        #
        # NOTE: I assumed w_bool was one-dimensional (unit axis)
        #       and that spikes was two-dimensionsional (unit axis,
        #       time axis)
        b1 = np.tile(z, (*z.shape, 1))
        b1 = b1.T * z

        # Binary matrix where the value at i,j indicates if neurons
        # i and j simultaneously spiked at time t *and* that a
        # synapse exists between them at time t. The value at i,j
        # will be 0 if any of the three conditions (i spiked, j
        # spiked, i/j synapse together) are false, otherwise it
        # will be 1. This effectively masks the "firing buddies."
        return b1 * m


    # NOTE: `copy` is currently `False` because the `.filled()` method
    #       creates a deep copy implicitly, thus doing so again in
    #       `threshold_fnet` would be wasteful. If you rewrite this
    #       code to use the masked array directly (instead of the
    #       array output by `.filled()`), be deliberate about how you
    #       set the `copy` flag to avoid issues with propagating
    #       changes back up the stack via side effects. If you're at
    #       all unsure, leave `copy=True`.
    fn = threshold_fnet(fn, threshold, copy=False).filled()

    # mask of 0's and 1's for whether actual synaptic connections exist
    w_bool = np.where(w != 0, 1, 0)

    trialstarts = np.concatenate(([0], trialends[:-1]))
    recruit_graphs = []

    # for each trial segment (determined by trialends):
    for (t0, t1) in zip(trialstarts, trialends):

        # aggregate recruitment graphs for this segment
        segment_dur = t1 - t0
        recruit_segment = np.zeros(  # return variable
            (segment_dur, fn.shape[0], fn.shape[1]),
            dtype=object
        )
        for t in range(t0, t1):
            # Adjusted time index to be relative to the start of the
            # recruitment graph segment, rather than the start of the
            # entire timeseries
            offset_t = t - t0

            # Indicies of all neurons with synaptic connections that
            # are firing together at this timestep
            fb_mask = firing_buddy_mask(spikes[:, t], w_bool)

            # if we found at least one nonzero synaptic connection
            # between the active units fill in recruitment graph at
            # those existing active indices using values from
            # thresholded functional graph
            recruit_segment[offset_t, ...] = fn * fb_mask

        # aggregate for the whole batch, though the dimensions (i.e.
        # duration of each trial segment) will be ragged
        recruit_graphs.append(recruit_segment)


    return recruit_graphs


def get_binned_spikes_trialends(e_only, true_y, spikes, bin):
    """TODO: document function"""

    if e_only:
        n_units = e_end
    else:
        n_units = i_end

    binned_spikes_coh0 = np.empty([n_units, 0])
    trialends_coh0 = []
    binned_spikes_coh1 = np.empty([n_units, 0])
    trialends_coh1 = []


    def _fastbin(z, bin_sz, num_units):
        """MUCH faster version of the following code:

        ```
        def slowbin(z, bin_sz, num_units):
            num_timesteps = z.shape[1]

            num_bins = np.math.floor(num_timesteps / bin_sz)
            r = np.zeros([num_units, num_bins])

            for t in range(num_bins):
                t0 = t * bin_sz
                t1 = (t + 1) * bin_sz
                spikes_in_bin = z[:, t0:t1]

                for j in range(num_units):
                    if 1 in spikes_in_bin[j, :]:
                        r[j, t] = 1
            return r
        ```

        The following code provides a more intuitive example of
        what this code does, but is not generalized for more
        than one dimension:

        ```
        def spike_bin_1d(z, binsize):
            # This code can be broken down into two steps:
            #
            # (1) reshape the spike train so that each row is
            #     a bin
            #
            # (2) compute the maximum value for each row. This
            #     will be either 1 or 0 (as it's a binary
            #     matrix), indicating whether or not a spike
            #     occurred in that bin
            #
            return z.reshape(-1, binsize).max(axis=1)

        z = np.array([0,0,0,  1,1,1,  1,0,0,  0,0,1])
        assert(spike_bin_1d(z, binsize=3) == [0,1,1,1]

        z = [0,0,0,1,  1,1,1,0,  0,0,0,1]
        assert(spike_bin_1d(z, binsize=4) == [0,1,1]

        z = [0,0,  0,1,  1,1,  1,0,  0,0,  0,1]
        assert(spike_bin_1d(z, binsize=2) == [0,1,1,1,0,1]
        ```

        To reiterate, the above code is for a single neuron's
        spiking activity. The operation performed on that
        single neuron in this simplified example is performed
        simultaneously on all neurons in the code generalized
        for higher dimensions.

        Assumes z[1] is the timestep axis. Assumes z is a
        binary matrix. Assumes z[0] is the neuron-index axis.
        Code should work generalize to any circumstance where
        these assumptions are met and the arguments are
        appropriate (z is a non-jagged NumPy array, bin_sz is
        a positive integer factor,
        num_units is a non-negative integer).

        Binning behavior note: truncates timesteps at the end if there
        are not enough for a complete bin

        TODO: confirm with YQ this is desired timestep/bin-size
            handling
        """
        # Correct for timestep dimensions misaligned with the bin size
        leftover_timesteps = z.shape[1] % bin_sz
        if leftover_timesteps > 0:
            z = z[:, :-leftover_timesteps, ...]

        # Perform actual binning
        if z.ndim <= 2:
            d2 = bin_sz
        else:
            d2 = bin_sz * np.prod(z.shape[2:])
        return z.reshape(num_units, -1, d2).max(axis=2)


    for trial in range(true_y.shape[0]):
        # each of 30 trials per batch update
        spikes_trial = np.transpose(spikes[trial])
        trial_y = np.squeeze(true_y[trial])
        # separate spikes according to coherence level
        coh_0_idx = np.squeeze(np.where(trial_y == 0))
        coh_1_idx = np.squeeze(np.where(trial_y == 1))

        if coh_0_idx.size > 0:
            # get spike data for relevant time range

            # if the start of a new coherence level happens in the
            # middle of the trial
            if not (0 in coh_0_idx):
                # ^ if time indices for "coh0 active" do not include
                #   the first timestep...
                #
                # v ...remove the first 50 ms
                coh_0_idx = coh_0_idx[50:]
            z_coh0 = spikes_trial[:, coh_0_idx]

            # bin spikes into 10 ms, discarding trailing ms
            trial_binned_z = _fastbin(z_coh0, bin, n_units)

            binned_spikes_coh0 = np.hstack(
                [binned_spikes_coh0, trial_binned_z]
            )

            # get all the spikes for each coherence level strung together
            trialends_coh0.append(binned_spikes_coh0.shape[1] - 1)
            # keep sight of new trial_end_indices relative to newly binned spikes

        else:

            if not (0 in coh_1_idx):
                coh_1_idx = coh_1_idx[50:]

            z_coh1 = spikes_trial[:, coh_1_idx]

            trial_binned_z = _fastbin(z_coh1, bin, n_units)

            binned_spikes_coh1 = np.hstack([binned_spikes_coh1, trial_binned_z])
            trialends_coh1.append(binned_spikes_coh1.shape[1] - 1)

    return [
        [binned_spikes_coh0, binned_spikes_coh1],
        [trialends_coh0, trialends_coh1],
    ]


def bin_batch_MI_graphs(
    w,
    spikes,
    true_y,
    bin,
    sliding_window_bins,
    threshold,
    e_only,
    positive_only,
    recruit_batch_savepath,
):
    """Calculate functional and binned recruitment graphs.

    Calculates functional graphs (using mutual information) and
    associated binned recruitment graphs for a single batch update.

    Assumes 30 trials per batch.

    """

    [
        [binned_spikes_coh0, binned_spikes_coh1],
        [trialends_coh0, trialends_coh1],
    ] = get_binned_spikes_trialends(e_only, true_y, spikes, bin)

    # pipe into confMI calculation
    fn_coh0 = simple_confMI(binned_spikes_coh0, trialends_coh0, positive_only)
    fn_coh1 = simple_confMI(binned_spikes_coh1, trialends_coh1, positive_only)

    # make and save recruitment graphs
    rn_coh0 = batch_recruitment_graphs(
        w, fn_coh0, binned_spikes_coh0, trialends_coh0, threshold
    )
    rn_coh1 = batch_recruitment_graphs(
        w, fn_coh1, binned_spikes_coh1, trialends_coh1, threshold
    )
    rn_coh0 = np.array(rn_coh0, dtype=object)
    rn_coh1 = np.array(rn_coh1, dtype=object)
    np.savez_compressed(
        recruit_batch_savepath,
        **{
            "coh0": rn_coh0,
            "coh1": rn_coh1
        }
    )
    return [fn_coh0, fn_coh1]






#######################################################################


# THIS IS THE SCRIPT WE CURRENTLY USE TO GENERATE BOTH FUNCTIONAL (MI) AND RECRUITMENT GRAPHS
# FOR THE FIRST TEN NAIVE (1-10.npz) AND LAST TEN TRAINED (991-1000.npz) EPOCHS ONLY
def generate_naive_trained_recruitment_graphs(
    experiment_string,
    overwrite=False,
    bin=10,
    sliding_window_bins=False,
    threshold=1,
    e_only=False,
    positive_only=False,
):
    """TODO: document function"""

    def _get_batch_w(_batch, _file_idx, _w, _data_dir, _exp_dir, _datafiles):
        """TODO: document function"""

        # In the case we are in the middle of a file's data, we have no
        # need to update the post-training weight values from the last
        # batch, so we use just the values from that last batch
        if _batch != 0:
            return _w[_batch - 1]

        # The two remaining cases both require reading from disk, so
        # we instantiate the path up to the deepest common directory
        _path = os.path.join(_data_dir, _exp_dir, "npz-data")

        # In the case we are at the start of the first file in the
        # experiment, we simply load the naive weights saved to disk
        # after network initialization
        if file_idx == 0:
            return np.load(os.path.join(_path, "main_preweights.npy"))

        # In the case we are at the start of a file which is not the
        # first of the experiment, we use the last post-training
        # weights from the previous file
        else:
            _prev_data = np.load(
                os.path.join(_path, _datafiles[_file_idx - 1])
            )
            return _prev_data["tv1.postweights"][99]


    if e_only:
        raise NotImplementedError("this flag is not currently supported")

    # TO REDUCE DISK SPACE (which is 35T per experiment for recruitment graphs),
    # we are first saving just fns for 1-10.npz and 991-1000.npz,
    # and the individual batch recruitment graphs corresponding.

    # VARIABLES:
    # experiment_string: the data we want to turn into recruitment graphs (for you, the string that is common to all rate-only experiments saved in /data/experiments/ (or wherever you saved them))

    # overwrite=False: do not overwrite already-saved files that contain generated networks

    # bin=10: use 10-ms bins of spikes to construct functional graphs

    # sliding_window_bins=False: for the sake of efficiency, these are discrete bins rather than sliding window through each ms

    # threshold=0.25: we take just the top quartile of functional weights to calculate recruitment graphs
    # threshold=1: we take all functional weights to calculate recruitment graphs; due to sparsity of spikes, this is the best option.

    # e_only=False: generate for e-e, e-i, i-e, and i-i connections (if e_only=True, only do e-e)

    # positive_only=False: we DO include negative confMI values (negative correlations); decide what to do with them later
    # previously we had always removed those, but now we'll try to make sense of negative correlations with Gabriella's help.

    # may need to modify the following to get the rate-only experiments
    experiments = get_experiments(LOAD_DIR, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    # networks will be saved as npz files (each containing multiple arrays), using the same names as data_files

    # may need to modify the following to save rate-only functional networks in a new folder
    # I've saved task-and-rate-trained functional and recruitment graphs on ava in /data/results/experiment1/MI_graphs_bin10/ and /data/results/experiment1/recruitment_graphs_bin10_full respectively


    # NOTE: YQ - may need to modify the following to save rate-only
    #       functional networks in a new folder. I've saved
    #       task-and-rate-trained functional and recruitment graphs on
    #       ava in /data/results/experiment1/MI_graphs_bin10/ and
    #       /data/results/experiment1/recruitment_graphs_bin10_full,
    #       respectively
    #
    # NOTE: YQ - may need to modify the recruit savepath to save
    #       rate-only recruitment networks in a new folder
    recruit_savepath = safely_make_joint_dirpath(
        SAVE_DIR, "recruitment_graphs_bin10_full"
    )
    MI_savepath = safely_make_joint_dirpath(SAVE_DIR, "MI_graphs_bin10")

    # for each rate-only experiment that you've run:
    for xdir in experiments:

        # I am using the time of each experiment as its identifying name for exp_path (like '05.21.35')
        exp_path = xdir[-9:-1]

        # check if MI and recruitment graph folders have already been generated
        # again you may need to modify this according to where you want to save your rate-only graphs
        safely_make_joint_dirpath(recruit_savepath, exp_path)
        safely_make_joint_dirpath(MI_savepath, exp_path)

        # for each batch update, there should be 2 functional networks (1 for each coherence level)
        # for file_idx in range(np.size(data_files)):
        for file_idx in [0, 99]: # just doing the first and last (naive and trained) for now to save space
            # the experimental npz data file (containing 10 epochs x 10 batches)
            filepath = os.path.join(
                data_dir, xdir, "npz-data", data_files[file_idx]
            )

            # case of we HAVE generated FNs for this npz file already (not applicable to you, unless script or screen crashes in the middle (this is why I have this))
            if os.path.isfile(
                os.path.join(MI_savepath, exp_path, data_files[file_idx])
            ):
                # load in pre-generated functional graphs
                data = np.load(
                    os.path.join(
                        MI_savepath, exp_path, data_files[file_idx]
                    )
                )
                fns_coh0 = data[
                    "coh0"
                ]  # each shaped 100 batch updates x 300 units x 300 units
                fns_coh1 = data["coh1"]
                # reduce decimal precision of fns for disk space
                fns_coh0 = np.around(fns_coh0, 4)
                fns_coh1 = np.around(fns_coh1, 4)
                # load in experiment data
                data = np.load(filepath)
                spikes = data["spikes"]
                true_y = data["true_y"]
                w = data["tv1.postweights"]

                # generate recruitment graphs
                for batch in range(true_y.shape[0]):
                    # paths for saving recruitment graphs
                    epoch_string = data_files[file_idx][:-4]
                    batch_string = (
                        epoch_string + "-batch" + str(batch) + ".npz"
                    )
                    recruit_batch_savepath = os.path.join(
                        os.path.join(
                            recruit_savepath, exp_path, batch_string
                        )
                    )

                    # continue if we have NOT generated this batch
                    # update's recruitment graph yet
                    if os.path.isfile(recruit_batch_savepath):
                        continue

                    batch_w = _get_batch_w(
                        batch, file_idx, w, data_dir, xdir, data_files
                    )

                    # bin spikes and find trialends again
                    [
                        [binned_spikes_coh0, binned_spikes_coh1],
                        [trialends_coh0, trialends_coh1],
                    ] = get_binned_spikes_trialends(
                        e_only, true_y[batch], spikes[batch], bin
                    )
                    # use to generate just recruitment graphs
                    rn_coh0 = batch_recruitment_graphs(
                        batch_w,
                        fns_coh0[batch],
                        binned_spikes_coh0,
                        trialends_coh0,
                        threshold,
                    )
                    rn_coh1 = batch_recruitment_graphs(
                        batch_w,
                        fns_coh1[batch],
                        binned_spikes_coh1,
                        trialends_coh1,
                        threshold,
                    )
                    # save batchwise recruitment graphs
                    rn_coh0 = np.array(rn_coh0, dtype=object)
                    rn_coh1 = np.array(rn_coh1, dtype=object)
                    np.savez_compressed(
                        recruit_batch_savepath,
                        coh0=rn_coh0,
                        coh1=rn_coh1,
                    )

            # case of FNs have NOT yet been generated (applicable to
            # you and your rate-only experiments) you might still need
            # to modify some folder / file names
            if os.path.isfile(
                os.path.join(MI_savepath, exp_path, data_files[file_idx])
            ):
                continue
            # load in data from experimental npz files
            data = np.load(filepath)
            spikes = data["spikes"]
            true_y = data["true_y"]
            w = data["tv1.postweights"]
            # generate MI and recruitment graphs from spikes for
            # each coherence level
            fns_coh0 = []
            fns_coh1 = []
            for batch in range(true_y.shape[0]):
                # each file contains 100 batch updates
                # each batch update has 30 trials
                # those spikes and labels are passed to generate
                # graphs batch-wise; each w is actually a
                # postweight, so corresponds to the next batch

                batch_w = _get_batch_w(
                    batch, file_idx, w, data_dir, xdir, data_files
                )

                # generate batch-wise MI and recruitment graphs
                # (batchwise recruitment graphs are saved within
                # this function call)
                epoch_string = data_files[file_idx][:-4]
                batch_string = (
                    epoch_string + "-batch" + str(batch) + ".npz"
                )
                recruit_batch_savepath = os.path.join(
                    os.path.join(
                        recruit_savepath, exp_path, batch_string
                    )
                )

                # recruitment graphs for each batch are saved
                # within this function call; saved as
                # "1-10-batch49.npz" for example
                batch_fns = bin_batch_MI_graphs(
                    batch_w,
                    spikes[batch],
                    true_y[batch],
                    bin,
                    sliding_window_bins,
                    threshold,
                    e_only,
                    positive_only,
                    recruit_batch_savepath,
                )

                # batch_fns is sized [2, 300, 300]
                # batch_rns is sized [2, 408, 300, 300]
                # aggregate functional networks to save
                fns_coh0.append(batch_fns[0])
                fns_coh1.append(batch_fns[1])

            # saving convention is same as npz data files (save as
            # 1-10.npz for example); fns_coh0 are fns_coh1 are each
            # sized [100 batch updates, 300 pre units, 300 post
            # units]; rns_coh0 (recruitment graph for coherence 0,
            # which is saved in the above function call to
            # bin_batch_MI_graphs) and rns_coh1 are each sized
            # [100 batch updates, variable # trial segments,
            # variable timesteps, 300 pre units, 300 post units];
            # we will separate by connection type (ee, ei, ie, ee)
            # in further analyses reduce decimal precision of fns
            # for disk space
            fns_coh0 = np.around(fns_coh0, 4)
            fns_coh1 = np.around(fns_coh1, 4)
            np.savez_compressed(
                os.path.join(
                    MI_savepath, exp_path, data_files[file_idx]
                ),
                **{
                    "coh0": fns_coh0,
                    "coh1": fns_coh1
                }
            )


#######################################################################






"""
# this function generates functional graphs (using confMI) to save
# so that we'll have them on hand for use in all the analyses we need
def generate_all_functional_graphs(experiment_string, overwrite=False, e_only=True, positive_only=False):
    # do not overwrite already-saved files that contain generated functional networks
    # currently working only with e units
    # positive_only=False means we DO include negative confMI values (negative correlations)
    # previously we had always removed those, but now we'll try to make sense of negative correlations as we go
    experiments = get_experiments(data_dir, experiment_string)
    data_files = filenames(num_epochs, epochs_per_file)
    save_files = generic_filenames(num_epochs, epochs_per_file)
    MI_savepath = os.path.join(savepath,"MI_graphs")
    for xdir in experiments:
        exp_path = xdir[-9:-1]
        if not os.path.isdir(os.path.join(MI_savepath,exp_path)):
            os.makedirs(os.path.join(MI_savepath,exp_path))
        # check if FN folders have already been generated
        # for every batch update, there should be 2 FNs for the 2 coherence levels
        # currently we are only generating FNs for e units, not yet supporting i units
        if e_only:
            subdir_names = ['e_coh0','e_coh1']
            for subdir in subdir_names:
                if not os.path.isdir(os.path.join(MI_savepath,exp_path,subdir)):
                    os.makedirs(os.path.join(MI_savepath,exp_path,subdir))
            for file_idx in range(np.size(data_files)):
                filepath = os.path.join(data_dir, xdir, 'npz-data', data_files[file_idx])
                # check if we haven't generated FNs already
                if not os.path.isfile(os.path.join(MI_savepath,exp_path,subdir_names[0],save_files[file_idx])) or overwrite:
                    data = np.load(filepath)
                    spikes = data['spikes']
                    true_y = data['true_y']
                    # generate MI graph from spikes for each coherence level
                    fns_coh0 = []
                    fns_coh1 = []
                    for batch in range(np.shape(true_y)[0]): # each file contains 100 batch updates
                    # each batch update has 30 trials
                    # those spikes and labels are passed to generate FNs batch-wise
                        [batch_fn_coh0, batch_fn_coh1] = generate_batch_ccd_functional_graphs(spikes[batch],true_y[batch],e_only,positive_only)
                        fns_coh0.append(batch_fn_coh0)
                        fns_coh1.append(batch_fn_coh1)
                    # saving convention is same as npz data files
                    # within each subdir (e_coh0 or e_coh1), save as 1-10.npy for example
                    # the data is sized [100 batch updates, 240 pre e units, 240 post e units]
                    np.save(os.path.join(MI_savepath,exp_path,subdir_names[0],save_files[file_idx]), fns_coh0)
                    np.save(os.path.join(MI_savepath,exp_path,subdir_names[1],save_files[file_idx]), fns_coh1)
"""


def plot_rates_over_time(output_only=True):
    """TODO: document function"""

    # separate into coherence level 1 and coherence level 0
    experiments = get_experiments(data_dir, experiment_string)
    # plot for each experiment, one rate value per coherence level per batch update
    # this means rates are averaged over entire runs (or section of a run by coherence level) and 30 trials for each update
    # do rates of e units only
    # do rates of i units only
    data_files = filenames(num_epochs, epochs_per_file)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    # subplot 0: coherence level 0, e units, avg rate (for batch of 30 trials) over training time
    # subplot 1: coherence level 1, e units, avg rate (for batch of 30 trials) over training time
    # subplot 2: coherence level 0, i units, avg rate (for batch of 30 trials) over training time
    # subplot 3: coherence level 1, i units, avg rate (for batch of 30 trials) over training time
    for xdir in experiments:
        e_0_rate = []
        e_1_rate = []
        i_0_rate = []
        i_1_rate = []

        # TODO: add inline documentation
        for filename in data_files:

            # Read data from disk
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)

            # Unpack data
            spikes = data["spikes"]
            y = data["true_y"]
            output_w = data["tv2.postweights"]
            y.resize([y.shape[0], y.shape[1], y.shape[2]])

            # TODO: add inline documentation
            for i in range(y.shape[0]):
                # each file contains 100 batch updates
                # find indices for coherence level 0 and for 1
                # do this for each of 30 trials bc memory can't accommodate the whole batch
                # this also circumvents continuity problems for calculating branching etc
                # then calculate rate of spikes for each trial according to coherence level idx
                batch_e_0_rate = []
                batch_e_1_rate = []
                batch_i_0_rate = []
                batch_i_1_rate = []
                # find the units that have nonzero projections to the output
                if output_only:
                    batch_w = np.reshape(
                        output_w[i], np.shape(output_w[i])[0]
                    )
                    e_out_idx = np.squeeze(np.where(batch_w > 0))
                    i_out_idx = np.squeeze(np.where(batch_w < 0))

                # TODO: add inline documentation
                for j in range(y.shape[1]):
                    # each of 30 trials per batch update
                    batch_y = np.squeeze(y[i][j])
                    coh_0_idx = np.squeeze(np.where(batch_y == 0))
                    coh_1_idx = np.squeeze(np.where(batch_y == 1))
                    spikes_trial = np.transpose(spikes[i][j])
                    if np.size(coh_0_idx) != 0:
                        if output_only:
                            batch_e_0_rate.append(
                                np.mean(
                                    spikes_trial[
                                        e_out_idx[:, None], coh_0_idx[None, :]
                                    ]
                                )
                            )
                            batch_i_0_rate.append(
                                np.mean(
                                    spikes_trial[
                                        i_out_idx[:, None], coh_0_idx[None, :]
                                    ]
                                )
                            )
                        else:
                            batch_e_0_rate.append(
                                np.mean(spikes_trial[0:e_end, coh_0_idx])
                            )
                            batch_i_0_rate.append(
                                np.mean(spikes_trial[e_end:i_end, coh_0_idx])
                            )
                    if np.size(coh_1_idx) != 0:
                        if output_only:
                            batch_e_1_rate.append(
                                np.mean(
                                    spikes_trial[
                                        e_out_idx[:, None], coh_1_idx[None, :]
                                    ]
                                )
                            )
                            batch_i_1_rate.append(
                                np.mean(
                                    spikes_trial[
                                        i_out_idx[:, None], coh_1_idx[None, :]
                                    ]
                                )
                            )
                        else:
                            batch_e_1_rate.append(
                                np.mean(spikes_trial[0:e_end, coh_1_idx])
                            )
                            batch_i_1_rate.append(
                                np.mean(spikes_trial[e_end:i_end, coh_1_idx])
                            )
                e_0_rate.append(np.mean(batch_e_0_rate))
                e_1_rate.append(np.mean(batch_e_1_rate))
                i_0_rate.append(np.mean(batch_i_0_rate))
                i_1_rate.append(np.mean(batch_i_1_rate))
        ax[0].plot(e_0_rate)
        ax[1].plot(e_1_rate)
        ax[2].plot(i_0_rate)
        ax[3].plot(i_1_rate)

    # TODO: add inline documentation
    for i in range(4):
        ax[i].set_xlabel("batch")
        ax[i].set_ylabel("rate")

    ax[0].set_title("e-to-output units, coherence 0")
    ax[1].set_title("e-to-output units, coherence 1")
    ax[2].set_title("i-to-output units, coherence 0")
    ax[3].set_title("i-to-output units, coherence 1")
    # Create and save the final figure
    fig.suptitle("experiment set 1.5 rates according to coherence level")

    # Draw and save plot
    plt.draw()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(savepath, "set_output_rates.png"), dpi=300)

    # Teardown
    plt.clf()
    plt.close()


# def plot_synch_over_time():
# use the fast measure

"""
def branching_param(bin_size,spikes): # spikes in shape of [units, time]
    run_time = np.shape(spikes)[1]
    nbins = np.round(run_time/bin_size)
    branch_over_time = np.zeros([nbins])

    # determine number of neurons which spiked at least once
    Nactive = 0
    for i in range(np.shape(spikes)[0]):
        if np.size(np.argwhere(spikes[i,:]==1))!=0:
            Nactive+=1

    # for every pair of timesteps, determine the number of ancestors and the number of descendants
    numA = np.zeros([nbins-1]); # number of ancestors for each bin
    numD = np.zeros([nbins-1]); # number of descendants for each ancestral bin
    d = np.zeros([nbins-1]); # the ratio of electrode descendants per ancestor

    for i in range(nbins-1):
        numA[i] = np.size(spikes[:,i]==1)
        numD[i] = np.size(spikes[:,i+bin]==1)
        d[i] = np.round(numD[i]/numA[i])

    # filter out infs and nans
    d = d[~numpy.isnan(d)]
    d = d[~numpy.isinf(d)]
    dratio = np.unique(d)
    pd = np.zeros([np.size(dratio)])
    Na = np.sum(numA)
    norm = (Nactive-1)/(Nactive-Na) # correction for refractoriness
    # and then what do I do with this?

    i = 1
    for ii in range(np.size(dratio)):
        idx = np.argwhere(d==dratio[ii])
        if np.size(idx)!=0:
            nad = np.sum(numA[idx])
            pd[i] = (nad/Na)
        i+=1

    net_bscore = np.sum(dratio*pd)
    return net_bscore"""

def get_binned_spikes(z, bin_sz):
    # Perform actual binning
    if z.ndim <= 2:
        d2 = bin_sz
    else:
        d2 = bin_sz * np.prod(z.shape[2:])
    return z.reshape(np.shape(z)[0], -1, d2).max(axis=2)

def simple_branching_param(
    bin_size, spikes
):  # spikes in shape of [units, time]

    # Correct for timestep dimensions misaligned with the bin size
    leftover_timesteps = spikes.shape[1] % bin_size
    if leftover_timesteps > 0:
        spikes = spikes[:, :-leftover_timesteps, ...]

    # get binned spikes
    spikes = get_binned_spikes(spikes, bin_size)

    nbins = np.shape(spikes)[1]

    # for every pair of timesteps, determine the number of ancestors
    # and the number of descendants
    numA = np.zeros([nbins - 1])
    # number of ancestors for each bin
    numD = np.zeros([nbins - 1])
    # number of descendants for each ancestral bin

    for i in range(nbins - 1): # stepping through each adjacent bin of 10ms
        numA[i] = np.size(np.argwhere(spikes[:, i] == 1))
        numD[i] = np.size(np.argwhere(spikes[:, i + 1] == 1))

    # the ratio of descendants per ancestor
    d = weird_division(numD,numA)
    # if we get a nan, that means there were no ancestors in the
    # previous time point;
    # in that case it probably means our choice of bin size is wrong
    # but to handle it for now we should probably just disregard
    # if we get a 0, that means there were no descendants in the next
    # time point;
    # 0 in that case is correct, because branching 'dies'
    # however, that's also incorrect because it means we are choosing
    # our bin size wrong for actual synaptic effects!
    # will revisit this according to time constants
    bscore = np.nanmean(d[d!=0])

    return bscore

def weird_division(n, d):
    divided = np.copy(n)
    for i in range(len(divided)):
        if d[i]!=0:
            divided[i] = n[i] / d[i]
        else:
            divided[i] = 0
    return divided

def plot_branching_over_time(experiment_string=rate_experiment_string):
    # count spikes in adjacent time bins
    # or should they be not adjacent?
    bin_size = 10  # for now adjacent pre-post bins are just adjacent ms
    # separate into coherence level 1 and coherence level 0
    experiments = get_experiments(data_dir, experiment_string)
    # plot for each experiment, one branching value per coherence level
    # per batch update
    #
    # this means branching params are averaged over entire runs (or
    # section of a run by coherence level) and 30 trials for each update
    data_files = filenames(num_epochs, epochs_per_file)

    # subplot 0: coherence level 0, e units, avg branching (for batch
    # of 30 trials) over training time

    # subplot 1: coherence level 1, e units, avg branching (for batch
    # of 30 trials) over training time

    # subplot 2: coherence level 0, i units, avg branching (for batch
    # of 30 trials) over training time

    # subplot 3: coherence level 1, i units, avg branching (for batch
    # of 30 trials) over training time

    for xdir in experiments:
        exp_path = xdir[-9:-1]
        #task_exp_path = 'taskloss_'+exp_path
        rate_exp_path = 'rateloss_'+exp_path
        if not os.path.isdir(os.path.join(savepath, rate_exp_path)):
            os.makedirs(os.path.join(savepath, rate_exp_path))
        #if not os.path.isdir(os.path.join(savepath, exp_path)):
        #    os.makedirs(os.path.join(savepath, exp_path))
        # plot and save separately for each experiment (lest we take forever)
        #fig, ax = plt.subplots(nrows=2, ncols=1)
        e_0_branch = []
        e_1_branch = []
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            spikes = data["spikes"]
            y = data["true_y"]
            y.resize(y.shape[:3])  # remove trailing dimension
            for i in range(y.shape[0]): # for each batch
                # each file contains 100 batch updates
                # find indices for coherence level 0 and for 1
                # do this for each of 30 trials bc memory can't
                # accommodate the whole batch
                # this also circumvents continuity problems for
                # calculating branching etc
                # then calculate rate of spikes for each trial
                # according to coherence level idx

                # average according to batch
                # (since it is a single update)
                batch_e_0_branch = []
                batch_e_1_branch = []
                for j in range(y.shape[1]): # for each trial
                    coh_0_idx = np.argwhere(y[i][j] == 0)
                    coh_1_idx = np.argwhere(y[i][j] == 1)
                    spikes_trial = np.transpose(spikes[i][j])
                    if np.size(coh_0_idx) > 0:
                        # remove trailing dim
                        coh_0_idx.resize(coh_0_idx.shape[:1])
                        batch_e_0_branch.append(
                            simple_branching_param(
                                bin_size, spikes_trial[0:e_end, coh_0_idx]
                            )
                        )
                    if np.size(coh_1_idx) > 0:
                        # remove trailing dim
                        coh_1_idx.resize(coh_1_idx.shape[:1])
                        batch_e_1_branch.append(
                            simple_branching_param(
                                bin_size, spikes_trial[0:e_end, coh_1_idx]
                            )
                        )
                e_0_branch.append(np.mean(batch_e_0_branch))
                e_1_branch.append(np.mean(batch_e_1_branch))
        plt.figure()
        plt.plot(e_0_branch)
        plt.plot(e_1_branch)
        plt.legend(["coherence 0","coherence 1"])
        plt.xlabel('batch')
        plt.ylabel('branching param')
        plt.title('excitatory branching over rate-only training')
        plt.draw()
        save_fname = savepath+rate_exp_path+'/'+exp_path+'_e_nonzero_branching.png'
        plt.savefig(save_fname, dpi=300)

        # Teardown
        plt.clf()
        plt.close()
        # now on to the next experiment
