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
    """Logging interface used while training."""

    def __init__(self, cfg, cb=None):
        super().__init__(cfg, cb)


    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Standard Methods                                                      │
    #┴───────────────────────────────────────────────────────────────────────╯

    def log(self, data_label, data, meta={}):
        """Main logging interface.

        All data must be reduceable to a numpy array. If you have data
        you want nested, you'll have to name it in a way that that info
        can be recovered.

        Any metadata you want about the main data will be included in
        `meta.pickle`. This should be about the nature of the variable,
        not the individually logged values, as it will only be added
        the first time the data is observed. Examples of good metadata
        include whether the data is stepwise or epochwise, a text
        description, etc. `stride` is the expected keyword as either
        'step' or 'epoch' or 'static' (never changes)
        """

        # Reduce float precision if specified in the HJSON
        try:
            if data.dtype == np.float64:
                print(f">> np.{self.cfg['log'].dtype}")
                data = eval(f"data.astype(np.{self.cfg['log'].dtype})")
                logging.debug('reduced {data_label} precision')
        except:
            pass

        # Primary data
        if data_label not in self.logvars:
            self.logvars[data_label] = [data]
        else:
            self.logvars[data_label].append(data)

        # Metadata
        if data_label not in self.meta:
            meta['dtype'] = type(data)
            self.meta[data_label] = meta

            if 'stride' not in meta:
                logging.warning('stride unspecified for ' + data_label)


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

        t0 = time.time()

        cfg = self.cfg

        lo_epoch = 1 if self.last_post[0] is None else self.last_post[0] + 1
        hi_epoch = self.cur_epoch

        fp = os.path.join(
            cfg['save'].main_output_dir,
            f"{lo_epoch}-{hi_epoch}.npz"
        )

        # Plot data from the end of each epoch
        for epoch_idx in range(cfg['log'].post_every):
            step_idx = epoch_idx * cfg['train'].n_batch
            self.plot_everything(f"{lo_epoch + epoch_idx}.png", step_idx)

        # Save the data to disk
        for k in self.logvars.keys():
            self.logvars[k] = np.array(self.logvars[k])

        np.savez_compressed(fp, **self.logvars)

        # Free up RAM
        self.logvars = {}

        # Bookkeeping
        self.last_post = (self.cur_epoch, self.cur_step)

        # Report how long the saving operation(s) took
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
        action_list = {}

        # Bookkeeping
        self.cur_step += 1

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
        action_list = {}

        # Bookkeeping
        self.cur_epoch += 1
        self.cur_step = 0

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
        logging_level = logging

        # [?] should loggers have their model as an attribute?

        last_trial_idx = self.cfg['train'].batch_size - 1

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
