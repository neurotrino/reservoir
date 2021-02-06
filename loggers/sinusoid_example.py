"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`logging.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

from loggers.base import BaseLogger

import matplotlib.pyplot as plt
import os
import tensorflow as tf

class Logger(BaseLogger):
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)

        # Lists updated at the end of every epoch
        #
        # [?] Some room for improvement:
        #
        # The way this is setup is that we have numpy arrays which fill
        # with values we want to log until some point we specify in the
        # training loop which dumps their contents to a file. I wasn't
        # sure where the overhead would be the worst: keeping large
        # numpy arrays in memory, or saving to disk, so this seemed
        # like the logical decision. If there's a better/idiomatic way
        # of doing this, please let me know or just implement it.
        #
        # Additionally, because python's append is in O(1) time and the
        # conversion to np array is in O(n) time, we'lll have a linear
        # time operation every time we reset the buffer. If there's a
        # way to initialize the numpy arrays with both dimensions at
        # once, that would be constant time. I feel like we should know
        # all the dimensions based on our model, but I'm not sure, and
        # wanted to move on to more pressing matters, but if this
        # becomes a bottleneck we should be able to use values from
        # cfg['data'] and cfg['train'] to initialize 2D numpy arrays.
        #
        # The buffer length would be number of batches times however
        # many epochs we want to keep the data in them for, then the
        # numpy dimensions would be ... something
        self.voltages = list()
        self.spikes = list()
        self.pred_ys = list()

        self.inputs = list()
        self.true_ys = list()

    def plot_sinusoid(
        self,
        epoch_idx=None,  # Epoch to plot the data of (epoch_idx >= 1)
        filename=None,   # Name of the saved plot (not the full path)
        show=False,      # When true, pause execution and show the plot
        save=True        # When false, the plot won't be saved
    ):

        # If no epoch is specified, plot the most recent
        if epoch_idx is None:
            epoch_idx = len(self.true_ys)

        # If no filename is specified, save as "sine_{epoch_idx}.png"
        if filename is None:
            filename = f"sinusoid_{epoch_idx}.png"

        # Create the plot
        true_y = self.true_ys[epoch_idx - 1]
        pred_y = self.pred_ys[epoch_idx - 1]

        plt.plot(true_y[0, :, :])
        plt.plot(pred_y[0, :, :])
        plt.draw()

        # Show the plot immediately, if requested (halts execution)
        if show:
            plt.show()

        # Save the plot, unless requested not to
        if save:
            plt.savefig(os.path.join(self.cfg['save'].plot_dir, filename))

        plt.clf()

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Pseudo Callbacks                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def on_epoch_end(self, epoch_idx, model):
        pass
