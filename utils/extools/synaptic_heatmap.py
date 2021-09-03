"""Generate heatmaps of synapse weights."""

# [!] Considering adding a matplotlib slider option for an
#     interactive way to see how the data evolves over time

# [!] Considering adding a "make movie" option

import argparse
import glob
import logging
import numpy as np
import os
import pickle
import seaborn as sns

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Core Functionality                                                        │
#┴───────────────────────────────────────────────────────────────────────────╯

def plot_heatmap(weights, filename, show_value_bounds=True):
    """Generate a heatmap of synaptic weights.

    Setting show_value_bounds to false will replace the min/max values
    displayed on the colorbar with ('−') and ('+'), indicating which
    color is inhibitory and which is excitatory. This can be useful if
    you're making a movie from a series of heatmaps.
    """
    weights = weights.copy()  # we don't want to change internal values

    min_wgt = np.min(weights)
    max_wgt = np.max(weights)

    ticklabels = [' (−)', ' 0', ' (+)']

    # Normalize the colorbar's gradient
    if min_wgt < 0:
        weights[weights < 0] /= abs(min_wgt)
        if show_value_bounds:
            ticklabels[0] = f' {min_wgt:.4f}'
    if max_wgt > 0:
        weights[weights > 0] /= max_wgt
        if show_value_bounds:
            ticklabels[2] = f' {max_wgt:.4f}'

    # Generate heatmap
    heatmap = sns.heatmap(
        weights,
        cmap='seismic', vmin=-1, vmax=1,
        cbar_kws={
            'ticks': [-1, 0, 1],
            'label': 'Synapse Strength'
        },
        xticklabels=[], yticklabels=[]
    )
    heatmap.collections[0].colorbar.set_ticklabels(ticklabels)

    heatmap.set_title('Synaptic Weights')
    heatmap.set_xlabel('Target Neuron')
    heatmap.set_ylabel('Projecting Neuron')

    heatmap.get_figure().savefig(filename)
    heatmap.get_figure().clf()

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Script-Based Execution                                                    │
#┴───────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    """Generate plots using npz files.

    When invoked directly, the script takes the following commandline
    arguments:

        Mandatory:
        > input: file or directory to read synapse weights from
        > output: name of output directory (can create new directories)

        Optional:
        > var (-v, --var): dictionary key indexing synapse weights in
          the target file(s)
        > granularity (-g, --granularity): granularity of plot
          generation (STEP, EPOCH)
        > epoch size (-e, --epoch-size): number of steps per epoch in
          the specified data
        > show-bounds (-b, --show-bounds): whether or not to show
          numeric values for min/max weights on the colorbar
    """

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Parse Commandline Arguments                                           │
    #┴───────────────────────────────────────────────────────────────────────╯

    parser = argparse.ArgumentParser(description="generate synaptic heatmaps")

    parser.add_argument(
        'input',
        metavar='I',
        help='file or directory to read synapse weights from'
    )
    parser.add_argument(
        'output',
        metavar='O',
        help='name of output directory (can create new directories)'
    )
    parser.add_argument(
        '-v',
        '--var',
        metavar='V',
        default='tv1.postweights',
        help='dictionary key indexing synapse weights in the target file(s)'
    )
    parser.add_argument(
        '-g',
        '--granularity',
        metavar='G',
        default='EPOCH',
        choices=['STEP', 'EPOCH'],
        help='granularity of plot generation (STEP, EPOCH)'
    )
    parser.add_argument(
        '-e',
        '--epoch-size',
        metavar='E',
        default=None, type=int,
        help='number of steps per epoch in the specified data'
    )
    parser.add_argument(
        '-b',
        '--show-bounds',
        metavar='B',
        choices=[True, False], type=bool,
        help='whether or not to show numeric values for min/max weights'
    )

    args = parser.parse_args()

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Setup                                                                 │
    #┴───────────────────────────────────────────────────────────────────────╯

    # Check if script input is a file or a directory and get datafiles
    if os.path.isdir(args.input):
        os.chdir(args.input)
        npz_files = glob.glob('*.npz')
    elif os.path.isfile(args.input):
        npz_files = [args.input]
    else:
        raise ValueError('input is neither file nor directory')
    logging.info(f'found {len(npz_files)} file(s)')

    # Determine epoch size if necessary
    pickle_path = os.path.abspath('config.pickle')  # note chdir above
    if args.granularity == 'EPOCH':
        if args.epoch_size != None:
            epoch_size = args.epoch_size
        elif os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                cfg = pickle.load(f)
            epoch_size = cfg['train'].n_batch  # batches per epoch
        else:
            raise Exception('epoch size unspecified or missing config.pickle')

    # Create the output directory if necessary
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Generate Plots                                                        │
    #┴───────────────────────────────────────────────────────────────────────╯

    cur_step = 1
    cur_epoch = 1

    for npz_file in npz_files:
        npz_data = np.load(npz_file)[args.var]
        num_steps = npz_data.shape[0]

        if args.granularity == 'EPOCH':
            for i in range(0, num_steps, epoch_size):
                plot_heatmap(
                    npz_data[i + epoch_size - 1],
                    os.path.join(args.output, f'e{cur_epoch:05}.png'),
                    args.show_bounds
                )
                cur_epoch += 1
        else:
            for wmatrix in npz_data:
                plot_heatmap(
                    wmatrix,
                    os.path.join(args.output, f's{cur_step:05}.png'),
                    args.show_bounds
                )
                cur_step += 1
