"""TODO: module docs"""

# TODO: if we end up with too many callbacks, probably create a `plots`
#       directory here, with `Generic` in `base.py`, and so on

import tensorflow as tf
import os
import matplotlib.pyplot as plt

import models.neurons.adex as adex
import models.neurons.lif as lif

import matplotlib.pyplot as plt

import logging
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ ...                                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯

class Generic(tf.keras.callbacks.Callback):
    def on_epoch_end():
        if isinstance(self.model, lif.LIF):
            pass
        elif isinstance(self.model, lif.ALIF):
            pass
        else:
            pass  # and so on...

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ ...                                                                       │
#┴───────────────────────────────────────────────────────────────────────────╯

class LIF(tf.keras.callbacks.Callback):
    def __init__(self, cfg, test_example, fig, axes):
        super().__init__()
        self.cfg = cfg
        self.test_example = test_example
        self.fig = fig
        self.axes = axes

    def on_epoch_end(self, epoch, logs=None):

        # plot every n epochs
        if (epoch + 1) % self.model.cfg['log'].plot_every == 0:
            last_pred = self.model.prediction_buffer[-1]
            last_true = self.model.true_y_buffer[-1]

            # [?] save plots w/out showing? faster?
            plt.plot(last_true[0, :, :])
            plt.plot(last_pred[0, :, :])
            plt.show()

        """
        output = self.model(self.test_example[0])
        #weights = self.model.layers[0].get_weights()[0]
        [ax.clear() for ax in self.axes]
        im = self.axes[0].pcolormesh(self.test_example[0].numpy()[0].T, cmap='cividis')
        self.axes[0].set_ylabel('input')
        cb1 = self.fig.colorbar(im, ax = self.axes[0])
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        # abs_max = np.abs(v).max()

        # plot transpose of voltage matrix as colormap
        print()
        print(self.model)
        print()

# function...
        # self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        im = self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=EL-15, vmax=thr+15)
        cb2 = self.fig.colorbar(im, ax = self.axes[1])

# aesthetic...
        self.axes[1].set_ylabel('voltage')

# function...
        # plot transpose of spike matrix
        im = self.axes[2].pcolormesh(z.T, cmap='Greys', vmin=0, vmax=1)
        cb3 = self.fig.colorbar(im, ax = self.axes[2])

# aesthetic...
        self.axes[2].set_ylabel('spike')
        self.axes[3].plot(self.test_example[1]['tf_op_layer_output'][0, :, 0], 'k--', lw=2, alpha=.7, label='target')
        self.axes[3].plot(out, 'b', lw=2, alpha=.7, label='prediction')
        self.axes[3].set_ylabel('output')
        self.axes[3].legend(frameon=False)

# function...
        # plot weight distribution after this epoch
        #self.axes[4].hist(weights)

# aesthetic...
        #self.axes[4].set_ylabel('count')
        #self.axes[4].set_xlabel('recurrent weights')
        [ax.yaxis.set_label_coords(-.05, .5) for ax in self.axes]

# overhead...
        plt.draw()
        plt.savefig(os.path.expanduser('test_epoch_{}.png'.format(epoch)), dpi=300)
        #plt.savefig(os.path.expanduser(os.path.join(root_path, 'tf2_testing/LIF/p{}/test_epoch_{}.png'.format(int(p*100), epoch))), dpi=300)
        cb1.remove()
        cb2.remove()
        cb3.remove()
        """
