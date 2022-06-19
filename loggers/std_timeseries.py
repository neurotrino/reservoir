"""Logger class(es) for monitoring network training and testing.

The `Logger` class is fairly minimal, serving only to interface between
a `Trainer` and TensorFlow's logging mechanisms. Formatting of data to
be logged is left up to the `Trainer` and `CallBacks` in
`loggers.callbacks`.

Resources:
  - "Complete TensorBoard Guide" : youtube.com/watch?v=k7KfYXXrOj0
"""

# external ----
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import time

# internal ----
from loggers.base import BaseLogger

WHITE_COLORMAP = LinearSegmentedColormap(
    "white_cmap",
    {
        "red": (
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        "green": (
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        "blue": (
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        )
    }
)

# ┬──────────────────────────────────────────────────────────────────────────╮
# │ Plotting Functions                                                       │
# ┴──────────────────────────────────────────────────────────────────────────╯

def save_weight_hist(filename, weights, title="Recurrent Layer Weights"):
    """Plot a histogram of recurrent weights."""

    # Plot
    plt.hist(weights)
    plt.ylabel("count")
    plt.xlabel("weight strength (recurrent layer)")
    plt.title(title)
    plt.savefig(filename)

    # Teardown
    plt.clf()
    plt.close()


def save_io_plot(
    filename,
    inps,
    voltages,
    spikes,
    pred_y,
    true_y,
    title="Model I/O",
    input_cmap_kwargs={},
    voltage_cmap_kwargs={},
):
    """Plot model input and output (voltage, spiking, performance)."""


    def plot_input(inp, fig, axes, ax_idx=0):
        ax = axes[ax_idx]

        im = ax.pcolormesh(inp.T, **input_cmap_kwargs)
        ax.set_ylabel("input")

        ax.set_title("Model I/O\n(epoch=17)")

        return fig.colorbar(im, ax=axes[0])


    def plot_voltage(voltage, fig, axes, ax_idx=1):
        ax = axes[ax_idx]

        im = ax.pcolormesh(voltage.T, **voltage_cmap_kwargs)
        ax.set_ylabel("voltage [mV]")

        return fig.colorbar(im, ax=ax)


    def plot_spikes(spikes, fig, axes, ax_idx=2):
        ax = axes[ax_idx]

        im = ax.pcolormesh(spikes.T, vmin=0, vmax=1, cmap="binary")
        axes[2].set_ylabel("spike")

        return fig.colorbar(im, ax=ax)


    def plot_performance(true_y, pred_y, axes, ax_idx=3):
        ax = axes[ax_idx]

        ax.plot(true_y, "k--", lw=2, alpha=0.7, label="target")
        ax.plot(pred_y, "b", lw=2, alpha=0.7, label="prediction")
        ax.set_ylabel("output")
        ax.legend(frameon=False)

        # All these extra steps (creating an invisible colorbar) are to
        # align the plots
        max_val = np.max(true_y)
        im = ax.pcolormesh(
            true_y,
            vmin=(max_val + 1),
            vmax=(max_val + 2),
            cmap=WHITE_COLORMAP
        )
        cb = fig.colorbar(im)
        cb.set_ticks([])
        cb.outline.set_visible(False)
        return cb


    # Because matplotlib infringes on our logger, we quiet it here,
    # then unquiet it when we actually need to use it
    logger = logging.getLogger()
    true_level = logger.level
    logger.setLevel(max(true_level, logging.WARN))

    # last_trial_idx = self.cfg['train'].batch_size - 1
    last_trial_idx = 0

    # Input
    inp = inps[last_trial_idx]

    # Outputs
    pred_y = pred_y[last_trial_idx]
    true_y = true_y[last_trial_idx]
    voltage = voltages[last_trial_idx]
    spikes = spikes[last_trial_idx]

    # Initialize plot
    fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)

    [ax.clear() for ax in axes]

    # Create subplots
    colorbar_objects = [
        plot_input(inp, fig, axes),
        plot_voltage(voltage, fig, axes),
        plot_spikes(spikes, fig, axes),
    ]
    plot_performance(true_y, pred_y, axes)

    # Label, title
    plt.xlabel("timestep")
    plt.title(title)

    # Bodge
    r1 = Rectangle(
        xy=(0.75, 0.25),
        width=0.15,
        height=0.25,
        fc='white',
        zorder=2
    )
    r2 = Rectangle(
        xy=(0.75, 0.7),
        width=0.15,
        height=0.25,
        fc='white',
        zorder=2
    )
    fig.add_artist(r1)
    fig.add_artist(r2)

    # Label, title
    plt.xlabel("timestep")

    # Export
    plt.draw()
    plt.savefig(filename)

    # Teardown
    [cb.remove() for cb in colorbar_objects]
    plt.clf()
    plt.close()

    # Restore logger status
    logger.setLevel(true_level)


# ┬──────────────────────────────────────────────────────────────────────────╮
# │ Logger                                                                   │
# ┴──────────────────────────────────────────────────────────────────────────╯

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
            self.plot_everything(f"{lo_epoch + epoch_idx}", step_idx)

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

    def plot_everything(self, unique_id, idx=-1):
        """Save all plots the logger is requested to create."""

        # Voltage, spike, and performance data for the model
        model_output_filename = os.path.join(
            self.cfg["save"].plot_dir, f"{unique_id}-model-output.png"
        )
        save_io_plot(
            model_output_filename,
            self.logvars["inputs"][idx],
            self.logvars["voltage"][idx],
            self.logvars["spikes"][idx],
            self.logvars["true_y"][idx],
            self.logvars["pred_y"][idx],
            title=f"Model I/O\n(epoch {self.cur_epoch})",
            input_cmap_kwargs={
                "cmap": "cividis"
            },
            voltage_cmap_kwargs={
                "cmap": "seismic",
                "vmin": self.cfg["model"].cell.EL - 15,
                "vmax": self.cfg["model"].cell.thr + 15
            },
        )

        # Weight distribution in the recurrent layer
        weight_distr_filename = os.path.join(
            self.cfg["save"].plot_dir, f"{unique_id}-weight-distr.png"
        )
        save_weight_hist(
            weight_distr_filename,
            self.logvars["tv0.postweights"][idx],
            title=(
                "Distribution of Weights in the Recurrent Layer\n" +
                f"(epoch={self.cur_epoch})"
            )
        )
