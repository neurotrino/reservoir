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

import logging

class Logger(BaseLogger):
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)

        # Lists updated at the end of every step
        self.step_gradients = list()
        self.step_losses = list()

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

        self.epoch_losses = list()

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Standard Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def post(self, req="full"):
        """Operations you want to do with/on the data post-training.

        request can either be a list of spefici items or "full" to run
        all operations  TODO
        """
        pass

        # Convert checkpoints to numpy arrays of weights
        #...

        # We already have voltages in a numpy array
        # We already have spikes in a numpy array
        # We already have outputs (predictions) in a numpy array
        # We already have inputs in a numpy array
        # We already have the correcy values in a numpy array

        # If there are any values from the cfg we want to save in the
        # pickle, we can do that here
        #...

        # We can get *some* gradients, not sure if these are all the
        # gradients we want (talk to YQ)
        #...

        # This example does most of its plotting as the training loop
        # advances, but we could put more plot creation here if we
        # wanted to
        #...

        # Should we also be tracking loss in our logger? Yeah, but it's
        # also in the event file(s). Need to talk to YQ about task loss
        # versus spike regularization loss
        #...

        # Adaptation?
        #...


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Pseudo Callbacks                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def on_epoch_end(self, epoch_idx, model):
        # If there are a series of logging operations you want to do at
        # a certain junction in your training loop, it might be more
        # legible to bundle them togther in a method like this.
        pass

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Other Methods                                                         │
    #┴───────────────────────────────────────────────────────────────────────╯

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


    def plot_spikes(self, filename):
        plt.ion()
        fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)

#        plt.plot(self.spikes[-1][0, :, :])
        im = axes[2].pcolormesh(self.spikes[-1][0, :, :].T, cmap='Greys', vmin=0, vmax=1)
        cb2 = fig.colorbar(im, ax=axes[1])
        plt.draw()
        plt.savefig(os.path.join(self.cfg['save'].plot_dir, filename))
        plt.clf()


    def plot_voltages(self, filename):
        plt.plot(self.voltages[-1][0, :, :])
        plt.savefig(os.path.join(self.cfg['save'].plot_dir, filename))
        plt.clf()


    def plot_everything(self, filename):
        # [?] should loggers have their model as an attribute?

        # Input
        x = self.inputs[-1][0]  # shape = (seqlen, n_inputs)

        # Outputs
        pred_y = self.pred_ys[-1][0]
        true_y = self.true_ys[-1][0]
        voltage = self.voltages[-1][0]
        spikes = self.spikes[-1][0]

        # Plot
        fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)

        [ax.clear() for ax in axes]

        im = axes[0].pcolormesh(x.T, cmap='cividis')
        cb1 = fig.colorbar(im, ax=axes[0])
        axes[0].set_ylabel('input')

        im = axes[1].pcolormesh(
            voltage.T,
            cmap='seismic',
            vmin=self.cfg['model'].cell.EL - 15,
            vmax=self.cfg['model'].cell.thr + 15
        )
        cb2 = fig.colorbar(im, ax=axes[1])
        axes[1].set_ylabel('voltage')

        # plot transpose of spike matrix
        im = axes[2].pcolormesh(spikes.T, cmap='Greys', vmin=0, vmax=1)
        cb3 = fig.colorbar(im, ax=axes[2])
        axes[2].set_ylabel('spike')

        axes[3].plot(true_y, 'k--', lw=2, alpha=.7, label='target')
        axes[3].plot(pred_y, 'b', lw=2, alpha=.7, label='prediction')
        axes[3].set_ylabel('output')
        axes[3].legend(frameon=False)

        # plot weight distribution after this epoch
        #self.axes[4].hist(weights)
        #self.axes[4].set_ylabel('count')
        #self.axes[4].set_xlabel('recurrent weights')

        [ax.yaxis.set_label_coords(-0.05, 0.5) for ax in axes]

        plt.draw()

        plt.savefig(os.path.join(self.cfg['save'].plot_dir, filename))

        cb1.remove()
        cb2.remove()
        cb3.remove()
