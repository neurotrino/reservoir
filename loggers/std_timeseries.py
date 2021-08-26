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
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

# local -------
from loggers.base import BaseLogger

class Logger(BaseLogger):
    """Standard timeseries logger.

    Logger equipped with functionality core to most experiments using
    timeseries data.

    [?] Is there a way we can add abstraction in here so we can pass
        layers, so that the logger isn't tied to any specific model
        architecture? (I mean there is a way, this is basically just me
        writing a "TODO" to find the cleanest way)
    """

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)  # cb == callback(s)

        # TODO: support NoneType in these positions
        self.logvar_whitelist = cfg['log'].logvar_whitelist
        self.logvar_blacklist = cfg['log'].logvar_blacklist

        self.todisk_whitelist = cfg['log'].todisk_whitelist
        self.todisk_blacklist = cfg['log'].todisk_blacklist


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Standard Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def post(self):
        """Flush logvars to disk and generate plots.

        Buffered logvars are saved to the output directory specified by
        the HJSON configuration file, named `n-m.npz`, where n is the
        lowest-numbered epoch associated with buffered logvars and m
        the highest.

        Additionally, for each epoch j, a plot `j.png` showing voltage,
        spiking, and prediction-versus-truth data is created in an
        adjacent directory.

        The logvars buffer is emptied after this operation.
        """
        t0 = time.time()  # used to track `.post()` overhead

        # Plot data from the end of each epoch
        # [!] prefer not to rely on post_every?
        lo_epoch = 1 if self.last_post['epoch'] is None else self.last_post['epoch'] + 1
        hi_epoch = self.cur_epoch
        for epoch_idx in range(self.cfg['log'].post_every):
            step_idx = epoch_idx * self.cfg['train'].n_batch
            self.plot_everything(f"{lo_epoch + epoch_idx}.png", step_idx)

        # Write to disk, free RAM, and perform bookkeeping
        super().post()

        # Report how long the saving operation(s) took
        # [?] have two timers for plots and disk writing so we can have
        #     a more self-contained parent class?
        logging.info(
            f"posted data for epochs {lo_epoch}-{hi_epoch}"
            + f" ({time.time() - t0:.2f} seconds)"
        )

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ (Pseudo) Callbacks                                                    │
    #┴───────────────────────────────────────────────────────────────────────╯

    def on_train_begin(self):
        """Save some static data about the training."""

        # [*] right now I'm just pickling the whole config file
        fp = os.path.join(
            self.cfg['save'].main_output_dir,
            "config.pickle"
        )
        with open(fp, 'wb') as file:
            pickle.dump(self.cfg, file)


    def on_train_end(self):
        # Post any unposted data
        if self.logvars != {}:
            self.post()

        # Save accrued metadata
        fp = os.path.join(
            self.cfg['save'].main_output_dir,
            "meta.pickle"
        )
        with open(fp, 'wb') as file:
            pickle.dump(self.meta, file)


    def on_step_end(self):
        """
        Any logic to be performed in the logger whenever a step
        completes in the training.

        Returns a list of actions to be performed in the trainer (only
        for use when you can't do things on the logger side alone, such
        as saving model weights, where `.save_weights()` must be called
        from the trainer, at least for now).
        """
        action_list = super().on_step_end()

        # Maintain, for convenience, a list of epoch and step numbers
        # to align stepwise data to in the npz file
        self.log('step', self.cur_step, meta={'stride': 'step'})
        self.log('sw_epoch', self.cur_epoch, meta={'stride': 'step'})

        return action_list


    def on_epoch_end(self):
        """
        Any logic to be performed in the logger whenever an epoch
        completes in the training.

        Returns a list of actions to be performed in the trainer (only
        for use when you can't do things on the logger side alone, such
        as saving model weights, where `.save_weights()` must be called
        from the trainer, at least for now).
        """
        action_list = super().on_epoch_end()

        # Maintain, for convenience, a list of epoch numbers to align
        # epochwise data to in the npz file
        self.log('ew_epoch', self.cur_epoch, meta={'stride': 'epoch'})

        if self.cur_epoch % self.cfg['log'].post_every == 0:
            self.post()

            # [?] Originally used a CheckpointManager in the logger
            action_list['save_weights'] = True

        return action_list


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Other Logging Methods                                                 │
    #┴───────────────────────────────────────────────────────────────────────╯

    def plot_everything(self, filename, idx=-1):
        return#[!] tmp
        # Because matplotlib infringes on our logger, we quiet it here,
        # then unquiet it when we actually need to use it
        logger = logging.getLogger()
        true_level = logger.level
        logger.setLevel(max(true_level, logging.WARN))

        # [?] should loggers have their model as an attribute?

        # last_trial_idx = self.cfg['train'].batch_size - 1
        last_trial_idx = 0

        # Input
        x = self.logvars['inputs'][idx][last_trial_idx]

        # Outputs
        pred_y = self.logvars['pred_y'][idx][last_trial_idx]
        true_y = self.logvars['true_y'][idx][last_trial_idx]
        voltage = self.logvars['voltage'][idx][last_trial_idx]
        spikes = self.logvars['spikes'][idx][last_trial_idx]

        # Initialize plot
        fig, axes = plt.subplots(5, figsize=(6, 8), sharex=True)

        [ax.clear() for ax in axes]

        # Plot input
        im = axes[0].pcolormesh(x.T, cmap='cividis')
        cb1 = fig.colorbar(im, ax=axes[0])
        axes[0].set_ylabel('input')

        # Plot voltage
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

        # Plot prediction-vs-actual
        axes[3].plot(true_y, 'k--', lw=2, alpha=0.7, label='target')
        axes[3].plot(pred_y, 'b', lw=2, alpha=0.7, label='prediction')
        axes[3].set_ylabel('output')
        axes[3].legend(frameon=False)

        # plot weight distribution after this epoch
        # (n_input x n_recurrent) [?] do we need to flatten?
        axes[4].hist(self.logvars['tv0.postweights'][idx])
        axes[4].set_ylabel('count')
        axes[4].set_xlabel('recurrent weights')

        [ax.yaxis.set_label_coords(-0.05, 0.5) for ax in axes]

        # Export
        plt.draw()
        plt.savefig(os.path.join(self.cfg['save'].plot_dir, filename))

        # Teardown
        cb1.remove()
        cb2.remove()
        cb3.remove()

        plt.clf()
        plt.close()

        # Restore logger status
        logger.setLevel(true_level)
