"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`loggers.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

# external ----
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

# local -------
from loggers.base import BaseLogger

class Logger(BaseLogger):
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)

        self.logvars = {
            # Lists updated at the end of every step
            "step_gradients": list(),
            "step_losses": list(),
            "sr_out": list(),
            "sr_in": list(),
            "sr_wgt": list(),
            "sr_losses": list(),

            "voltages": list(),
            "spikes": list(),
            "pred_ys": list(),
            "inputs": list(),
            "true_ys": list(),

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
            "epoch_losses": list(),
        }

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Standard Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def post(self, epoch_idx):
        """Save stuff to disk."""
        t0 = time.time()

        cfg = self.cfg

        lo_epoch = epoch_idx - cfg['log'].post_every + 1
        hi_epoch = epoch_idx

        fp = os.path.join(
            cfg['save'].pickle_dir,
            f"{lo_epoch}-{hi_epoch}.pickle"
        )

        # Save the data to disk (pickle, npy, hdf5, etc.)
        with open(fp, "wb") as file:
            pickle.dump(self.logvars, file)

        # Free up RAM
        for k in self.logvars.keys():
            if type(self.logvars[k]) == list:
                self.logvars[k] = []

        # Report how long the saving operation(s) took
        logging.info(
            f"posted data for epochs {lo_epoch}-{hi_epoch}"
            + f" ({time.time() - t0:.2f} seconds)"
        )

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

    def plot_everything(self, filename):
        # [?] should loggers have their model as an attribute?

        # Input
        x = self.logvars['inputs'][-1][0]  # shape = (seqlen, n_inputs)

        # Outputs
        pred_y = self.logvars['pred_ys'][-1][0]
        true_y = self.logvars['true_ys'][-1][0]
        voltage = self.logvars['voltages'][-1][0]
        spikes = self.logvars['spikes'][-1][0]

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

        plt.clf()
        plt.close()
