import numpy as np
import seaborn as sns

def plot_heatmap(weights, filename, show_value_bounds=True):
    """Generate a heatmap of synaptic weights.

    Setting show_value_bounds to false will replace the min/max values
    displayed on the colorbar with ('−') and ('+'), indicating which
    color is inhibitory and which is excitatory. This can be useful if
    you're making a movie from a series of heatmaps.

    [!] TODO: add behaviors for when no excitatory or no inhbitory
    synapses are present (i.e. either start or end the color bar at
    zero, with it being all red or all blue)
    """
    weights = weights.copy()

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
            "ticks":[-1, 0, 1],
            "label":"Synapse Strength"
        },
        xticklabels=[], yticklabels=[]
    )
    heatmap.collections[0].colorbar.set_ticklabels(ticklabels)
    heatmap.set_title(f'Synaptic Weights')
    heatmap.set_xlabel("Target Neuron")
    heatmap.set_ylabel("Projecting Neuron")
    heatmap.get_figure().savefig(filename)
    heatmap.get_figure().clf()


if __name__ == "__main__":
    """Generate plots using npz files.

    When invoked directly, the script requires a file name and a key
    specifying the values to plot. An optional argument may also be
    passed specifying which steps should be plotted. By default, only
    the first and last synaptic states will generate plots.
    """
    # [!] TODO:
    #     - parse commandline arguments
    #     - get data
    #     - generate plot(s)
    #
    # [!] Also considering adding a matplotlib slider option for an
    #     interactive way to see how the data evolves over time
    pass
