"""TODO: docs"""

import pickle
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ ...                                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯

class Generic(tf.keras.callbacks.Callback):
    """TODO: docs"""

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Special Methods                                                       │
    #┴───────────────────────────────────────────────────────────────────────╯

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    #┬───────────────────────────────────────────────────────────────────────╮
    #┤ Callbacks                                                             │
    #┴───────────────────────────────────────────────────────────────────────╯

    def on_epoch_begin(self, epoch_id, logs=None):
        """Logging actions performed at the start of every epoch."""

        # Save checkpoint
        self.model.save_weights(
            f"{self.cfg['save'].checkpoint_dir}/begin_epoch_{epoch_id}.h5"
        )

    def on_epoch_end(self, epoch_id, logs=None):
        """Logging actions performed at the end of every epoch."""

        save_cfg = self.cfg['save']
        ckpt_fp = f"{save_cfg.checkpoint_dir}/end_epoch_{epoch_id}.h5"

        # Save checkpoint
        self.model.save_weights(ckpt_fp)

        # Save summary data
        summary = {  # TODO: logs dict in HJSON?
            "loss": logs['loss'],

            #"v": logs['outputs'][0].numpy()[0],
            #"z": logs['outputs'][1].numpy()[0],
            #"out": logs['outputs'][2].numpy()[0, :, 0],

            "ckpt_path": ckpt_fp
        }
        with open(
            f"{save_cfg.summary_dir}/epoch_{epoch_id}_model_output.pkl",
            'wb'
        ) as file:
            pickle.dump(summary, file)

        # Dump
        """FROM....

        # in addition to weights, save spikes, target, prediction, loss, rate, and rate loss for trials in this epoch.
        # so not for a test example as in PlotCallback, but truly what was happening in the training trials.
        # okay but for now, save output for a test example
        output = self.model(self.test_example[0])
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        data = [v,z,out]
        """


        # TODO: Loss

        # TODO: gradients
        # TODO: target
        # TODO: input
        # TODO: output
        # TODO:
        # TODO:

    # TODO:
    # - loss
    # - gradients
    # - target
    # - input
    # - output
    # - weight matrices (init & on update)
    # - neuron spike times
    # - input
    # - neuron state (model specific)


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ ...                                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯


