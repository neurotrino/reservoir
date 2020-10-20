import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import models
import os
import pickle
# import tfrecord_dataset

from keras.callbacks import ModelCheckpoint

root_path = '../data'
# root_path = '../tarek2'

# neuron model param flags
# some of these are not currently used, but will be needed for adapting units, adex, conductance-based synapses, etc.
Volt = 1e3
Siemens = 1e3
Second = 1e3
Ampere = Volt * Siemens
mAmpere = Ampere / 1e3
nAmpere = Ampere / 1e9
pAmpere = Ampere / 1e12
Farad = Ampere * Second / Volt
Ohm = 1 / Siemens
MOhm = Ohm * 1e6
uFarad = Farad / 1e6
mSecond = Second / 1e3
mVolt = Volt / 1e3
mSiemens = Siemens / 1e3
nSiemens = Siemens / 1e9
Hertz = 1 / Second

# Training parameters needed for all models
seq_len = 1000
learning_rate = 1e-2
n_epochs = 20
target_rate = 0.02
rate_cost = 0.1
do_plot = True
do_save = True
rewiring = False
n_input = 20
n_recurrent = 100
frac_e = 0.8  # in LIF_EI and Adex_EI, the fraction of total recurrent units that are excitatory
mu = -0.64  # mu for normal dist which we exponentiate for lognormal weights
sigma = 0.51  # sigma for normal dist which we exponentiate for lognormal weights
p = 0.2  # connectivity probability when using LIF or Adex model
p_ee = 0.160  # connectivity probability from E to E when using Adex_EI or LIF_EI
p_ei = 0.244  # connectivity probability from E to I when using Adex_EI or LIF_EI
p_ie = 0.318  # connectivity probability from I to E when using Adex_EI or LIF_EI
p_ii = 0.343  # connectivity probability from I to I when using Adex_EI or LIF_EI

# Parameters values for LIF cells
thr = -50.4 * mVolt
EL = -70.6 * mVolt
n_refrac = 4
tau = 20.
dt = 1.
dampening_factor = 0.3
"""
# Parameters values for Adex cells (currently used by Tarek)

EL = -70.6 * mVolt
gL = 30 * nSiemens
C = 281 * uFarad
deltaT = 2 * mVolt
thr = -40.4 * mVolt
tauw = 144 * mSecond
a = 4 * nSiemens
b = 80.5 * pAmpere
V_reset = -70.6 * mVolt
n_refrac = 2
dt = 1. * mSecond
dampening_factor = 0.30

# Parameters values for Adex cells with conductance-based synapses (model not working yet)

EL = -70.6 * mVolt
gL = 30 * nSiemens
C = 281 * uFarad
deltaT = 2 * mVolt
thr = -40.4 * mVolt
tauw = 144 * mSecond
a = 4 * nSiemens
b = 80.5 * pAmpere
V_reset = -70.6 * mVolt
n_refrac = 2
tauS = 10 * mSecond
VS = 0 * mVolt
dt = 1. * mSecond
dampening_factor = 0.30
"""
# Old parameters kept in case we need them
"""
# these ones are for later neuron models
flags.DEFINE_float('gL', 0.00003 * mSiemens, 'leak conductance')
flags.DEFINE_float('a', .000004 * mSiemens, 'adaptative scalar / coupling param')
flags.DEFINE_float('deltaT', 2 * mVolt, 'slope factor; quantifies sharpness of spikes')
flags.DEFINE_float('tauw', 144 * mSecond, 'tau for adaptation term w in adex')
flags.DEFINE_float('C', .000281 * uFarad, 'membrane capacitance')
flags.DEFINE_float('b', 0.0805 * nAmpere, 'amount by which adaptation value w is changed per spike in adex')
flags.DEFINE_float('V_reset', -70.6 * mVolt, 'reset voltage after spike (equal to EL)')

# flags for task / training set-up
flags.DEFINE_integer('seq_len', 1000, '')
flags.DEFINE_float('learning_rate', 1e-3, '')
flags.DEFINE_integer('n_epochs', 10, '')

flags.DEFINE_float('target_rate', 0.02, 'spikes/ms; for rate regularization') # right now separate layer in models.py
flags.DEFINE_float('rate_cost', 0.1, '')

flags.DEFINE_bool('do_plot', True, 'plotting')
flags.DEFINE_bool('do_save', True, 'saving data and plots')

# you may need to define these flags based on task set-up; some of these are currently hard-baked into this toy script
flags.DEFINE_integer('n_iter', 10000, 'total number of iterations')
flags.DEFINE_integer('batch_size', 32, 'trials in each training batch')
flags.DEFINE_integer('voltage_cost', 0.01, 'for voltage regularization')

# flags to be used in future versions
flags.DEFINE_bool('random_eprop', False, 'random or symmetric eprop feedback weights')
flags.DEFINE_float('reg_f', 1., 'regularization coefficient for firing rate')
flags.DEFINE_integer('print_every', 50, 'print out metrics to terminal after this many iterations to assess how training is going')

# SNN model architecture flags
flags.DEFINE_integer('n_input', 20, '') # 20 input channels
flags.DEFINE_integer('n_recurrent', 100, '') # recurrent network of 100 spiking units
"""

def create_model(seq_len, n_input, n_recurrent):
    inputs = tf.keras.layers.Input(shape=(seq_len, n_input))

    # cell = models.LIFCell(n_recurrent, thr, EL, tau, dt, n_refrac, dampening_factor, p, mu, sigma, rewiring)
    cell = models.LIF_EI(n_recurrent, frac_e, thr, EL, tau, dt, n_refrac, dampening_factor, p_ee, p_ei, p_ie, p_ii, mu, sigma, rewiring)
    # cell = models.Adex(n_recurrent, n_input, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p, mu, sigma, rewiring)
    # cell = models.AdexEI(n_recurrent, frac_e, n_input, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p_ee, p_ei, p_ie, p_ii, mu, sigma, rewiring)
    # cell = models.AdexCS(n_recurrent, n_input, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p, tauS, VS, mu, sigma, rewiring)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    batch_size = tf.shape(inputs)[0]
    initial_state = cell.zero_state(batch_size)
    rnn_output = rnn(inputs, initial_state=initial_state)
    regularization_layer = models.SpikeRegularization(cell, target_rate, rate_cost)
    voltages, spikes = regularization_layer(rnn_output)
    voltages = tf.identity(voltages, name='voltages')
    spikes = tf.identity(spikes, name='spikes')

    weighted_out_projection = tf.keras.layers.Dense(1)
    weighted_out = weighted_out_projection(spikes)

    prediction = models.exp_convolve(weighted_out, axis=1)
    prediction = tf.identity(prediction, name='output')

    return tf.keras.Model(inputs=inputs, outputs=[voltages, spikes, prediction])

# generate placeholder input data
def create_data_set(seq_len, n_input, n_batch=1):
    x = tf.random.uniform(shape=(seq_len, n_input))[None] * .5
    y = tf.sin(tf.linspace(0., 4 * np.pi, seq_len))[None, :, None]

    return tf.data.Dataset.from_tensor_slices((x, dict(tf_op_layer_output=y))).repeat(count=20).batch(n_batch)
    # 20 repetitions of truly the exact same input sinusoid.
    # should create a set with random small displacements along the x and y axes.

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_example):
        super().__init__()
        self.test_example = test_example

    def on_epoch_begin(self, epoch, logs=None):
        filepath = str(root_path) + "/tf2_testing/LIF_EI/sparse/set4/begin_epoch_" + str(epoch) + ".hdf5"
        #filepath = str(root_path) + "/tf2_testing/LIF/p" + str(int(p*100)) + "/begin_epoch_" + str(epoch) + ".hdf5"
        self.model.save_weights(filepath)

    def on_epoch_end(self, epoch, logs=None):
        filepath = str(root_path) + "/tf2_testing/LIF_EI/sparse/set4/end_epoch_" + str(epoch) + ".hdf5"
        # filepath = str(root_path) + "/tf2_testing/LIF/p" + str(int(p*100)) + "/end_epoch_" + str(epoch) + ".hdf5"
        self.model.save_weights(filepath)

        # in addition to weights, save spikes, target, prediction, loss, rate, and rate loss for trials in this epoch.
        # so not for a test example as in PlotCallback, but truly what was happening in the training trials.
        # okay but for now, save output for a test example
        output = self.model(self.test_example[0])
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        # OKAY, look into tensorboard and tf.summary.scalar


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_example, fig, axes):
        super().__init__()
        self.test_example = test_example
        self.fig = fig
        self.axes = axes

    def on_epoch_end(self, epoch, logs=None):

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
        # self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        im = self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=EL-15, vmax=thr+15)
        cb2 = self.fig.colorbar(im, ax = self.axes[1])
        self.axes[1].set_ylabel('voltage')
        # plot transpose of spike matrix
        im = self.axes[2].pcolormesh(z.T, cmap='Greys', vmin=0, vmax=1)
        cb3 = self.fig.colorbar(im, ax = self.axes[2])
        self.axes[2].set_ylabel('spike')
        self.axes[3].plot(self.test_example[1]['tf_op_layer_output'][0, :, 0], 'k--', lw=2, alpha=.7, label='target')
        self.axes[3].plot(out, 'b', lw=2, alpha=.7, label='prediction')
        self.axes[3].set_ylabel('output')
        self.axes[3].legend(frameon=False)
        # plot weight distribution after this epoch
        #self.axes[4].hist(weights)
        #self.axes[4].set_ylabel('count')
        #self.axes[4].set_xlabel('recurrent weights')
        [ax.yaxis.set_label_coords(-.05, .5) for ax in self.axes]
        plt.draw()
        plt.savefig(os.path.expanduser(os.path.join(root_path, 'tf2_testing/LIF_EI/sparse/set4/test_epoch_{}.png'.format(epoch))), dpi=300)
        #plt.savefig(os.path.expanduser(os.path.join(root_path, 'tf2_testing/LIF/p{}/test_epoch_{}.png'.format(int(p*100), epoch))), dpi=300)
        cb1.remove()
        cb2.remove()
        cb3.remove()

def main():
    model = create_model(seq_len, n_input, n_recurrent)
    data_set = create_data_set(seq_len, n_input, n_batch=1)
    it = iter(data_set)
    test_example = next(it)

    if do_plot:
        plt.ion()
        fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)
        plot_callback = PlotCallback(test_example, fig, axes)

    if do_save:
        save_callback = SaveCallback(test_example) # eventually args will include what vars to save; currently just weights

    # train the model
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=dict(tf_op_layer_output=mse))
    if do_plot and do_save:
        model.fit(data_set, epochs=n_epochs, callbacks=[plot_callback, save_callback])
    elif do_plot:
        model.fit(data_set, epochs=n_epochs, callbacks=[plot_callback])
    elif do_save:
        model.fit(data_set, epochs=n_epochs, callbacks=[save_callback])
    else:
        model.fit(data_set, epochs = n_epochs)

    # analyse the model
    inputs = test_example[0] # sample
    targets = test_example[1]['tf_op_layer_output'].numpy()
    voltage, spikes, prediction = model(inputs)

    voltage = voltage.numpy()
    spikes = spikes.numpy()
    prediction = prediction.numpy()

    print(f'inputs:            array with shape {inputs.shape}')
    print(f'membrane voltages: array with shape {voltage.shape}')
    print(f'spikes:            array with shape {spikes.shape}')
    print(f'prediction:        array with shape {prediction.shape}')
    print(f'targets:           array with shape {targets.shape}')


if __name__ == '__main__':
    main()
