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

        # Lists updated at the end of every step
        self.step_gradients = list()
        self.step_losses = list()
        self.sr_out = list()
        self.sr_in = list()
        self.sr_wgt = list()
        self.sr_losses = list()

        self.voltages = list()
        self.spikes = list()
        self.pred_ys = list()
        self.inputs = list()
        self.true_ys = list()

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
        self.epoch_losses = list()

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
        #
        # For now these have to be attributes, not one dictionary which
        # is itself an attribute (because of an issue with thread
        # locking and pickling). This unfortunately means you need to
        # manually add the items to this list.
        with open(fp, "wb") as file:
            pickle.dump(
                {
                    "step_gradients": self.step_gradients,
                    "step_losses": self.step_losses,
                    "sr_out": self.sr_out,
                    "sr_in": self.sr_in,
                    "sr_wgt": self.sr_wgt,
                    "sr_losses": self.sr_losses,

                    "voltages": self.voltages,
                    "spikes": self.spikes,
                    "pred_ys": self.pred_ys,
                    "inputs": self.inputs,
                    "true_ys": self.true_ys,

                    "epoch_losses": self.epoch_losses,
                },
                file
            )

        # Free up RAM
        for k in pickle_values.keys():
            if type(pickle_values[k]) == list:
                try:
                    exec(f"self.{k}")
                except:
                    logging.warning(f"pickle/class key mismatch: {k}")

        # Report how long the saving operation(s) took
        logging.info(
            f"posted data for epochs {lo_epoch}-{hi_epoch}"
            + f" ({time.time() - t0:.2f} seconds)"
        )

        # Convert checkpoints to numpy arrays of weights
        #...

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

        plt.clf()
        plt.close()
