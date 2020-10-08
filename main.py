import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app
from absl import flags
FLAGS = flags.FLAGS

import models
import tfrecord_dataset

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

# these ones are for LIF
flags.DEFINE_float('thr', -50.4 * mVolt, 'threshold at which neuron spikes')
flags.DEFINE_float('EL', -70.6 * mVolt, 'equilibrium potential for leak (all) channels')
flags.DEFINE_integer('n_refrac', 4, 'Number of refractory steps after each spike [ms]')
flags.DEFINE_float('tau', 20., 'membrane time constant')
flags.DEFINE_float('dt', 1. * mSecond, 'simulation time step')
flags.DEFINE_float('dampening_factor', 0.3, 'factor that controls amplitude of pseudoderivative')

# Parameters values for Adex cells (currently used by Tarek)
"""
flags.DEFINE_float('EL', -70.6 * mVolt, 'Equilibrium potential for leak (all) channels')
flags.DEFINE_float('gL', 30 * nSiemens, 'Leak conductance')
flags.DEFINE_float('C', 281 * uFarad, 'Membrane capacitance')
flags.DEFINE_float('deltaT', 2 * mVolt, 'Slope factor')
flags.DEFINE_float('thr', -40.4 * mVolt, 'Threshold at which neuron spikes')
flags.DEFINE_float('tauw', 144 * mSecond, 'Time constant for adaptation')
flags.DEFINE_float('a', 4 * nSiemens, 'Subthreshhold adaptation')
flags.DEFINE_float('b', 80.5 * pAmpere, 'Spike-triggered adaptation')
flags.DEFINE_float('V_reset', -70.6 * mVolt, 'After spike voltage')
flags.DEFINE_integer('n_refrac', 2, 'Number of refractory steps after each spike [ms]')
flags.DEFINE_float('p', 0.2, 'Connectivity probability')
flags.DEFINE_float('dt', 1. * mSecond, 'Simulation time step')
flags.DEFINE_float('dampening_factor', 0.3, 'Factor that controls amplitude of pseudoderivative')
"""

# Parameters values for Adex_EI cells (currently used by Tarek)
"""
flags.DEFINE_float('frac_e', 0.8, 'Proportion of excitatory cells')
flags.DEFINE_float('EL', -70.6 * mVolt, 'Equilibrium potential for leak (all) channels')
flags.DEFINE_float('gL', 30 * nSiemens, 'Leak conductance')
flags.DEFINE_float('C', 281 * uFarad, 'Membrane capacitance')
flags.DEFINE_float('deltaT', 2 * mVolt, 'Slope factor')
flags.DEFINE_float('thr', -40.4 * mVolt, 'Threshold at which neuron spikes')
flags.DEFINE_float('tauw', 144 * mSecond, 'Time constant for adaptation')
flags.DEFINE_float('a', 4 * nSiemens, 'Subthreshhold adaptation')
flags.DEFINE_float('b', 80.5 * pAmpere, 'Spike-triggered adaptation')
flags.DEFINE_float('V_reset', -70.6 * mVolt, 'After spike voltage')
flags.DEFINE_integer('n_refrac', 2, 'Number of refractory steps after each spike [ms]')
flags.DEFINE_float('p_ee', 0.160, 'Connectivity probability from excitatory to excitaotry neurons')
flags.DEFINE_float('p_ei', 0.244, 'Connectivity probability from excitatory to inhibitory neurons')
flags.DEFINE_float('p_ie', 0.318, 'Connectivity probability from inhibitory to excitaotry neurons')
flags.DEFINE_float('p_ii', 0.343, 'Connectivity probability from inhibitory to inhibitory neurons')
flags.DEFINE_float('dt', 1. * mSecond, 'Simulation time step')
flags.DEFINE_float('dampening_factor', 0.3, 'Factor that controls amplitude of pseudoderivative')
"""

"""
# these ones are for later neuron models
flags.DEFINE_float('gL', 0.00003 * mSiemens, 'leak conductance')
flags.DEFINE_float('a', .000004 * mSiemens, 'adaptative scalar / coupling param')
flags.DEFINE_float('deltaT', 2 * mVolt, 'slope factor; quantifies sharpness of spikes')
flags.DEFINE_float('tauw', 144 * mSecond, 'tau for adaptation term w in adex')
flags.DEFINE_float('C', .000281 * uFarad, 'membrane capacitance')
flags.DEFINE_float('b', 0.0805 * nAmpere, 'amount by which adaptation value w is changed per spike in adex')
flags.DEFINE_float('V_reset', -70.6 * mVolt, 'reset voltage after spike (equal to EL)')
"""

# flags for task / training set-up
flags.DEFINE_integer('seq_len', 1000, '')
flags.DEFINE_float('learning_rate', 1e-3, '')
flags.DEFINE_integer('n_epochs', 10, '')

"""
# you may need to define these flags based on task set-up; some of these are currently hard-baked into this toy script
flags.DEFINE_integer('n_iter', 10000, 'total number of iterations')
flags.DEFINE_integer('batch_size', 32, 'trials in each training batch')
"""

"""
# flags to be used in future versions
flags.DEFINE_bool('random_eprop', False, 'random or symmetric eprop feedback weights')
flags.DEFINE_integer('target_rate', 20, 'spikes/s; for rate regularization') # right now hard-coded in SpikeVoltageRegularization() in models.py
flags.DEFINE_float('reg_f', 1., 'regularization coefficient for firing rate')
flags.DEFINE_integer('print_every', 50, 'print out metrics to terminal after this many iterations to assess how training is going')
"""

# SNN model architecture flags
flags.DEFINE_integer('n_input', 20, '')
flags.DEFINE_integer('n_recurrent', 100, '')

def create_model(seq_len=flags.seq_len, n_input=flags.n_input, n_recurrent=flags.n_recurrent):
    inputs = tf.keras.layers.Input(shape=(seq_len, n_input))

    cell = models.LIFCell(n_recurrent, flags.thr, flags.EL, flags.tau, flags.dt, flags.n_refrac, flags.dampening_factor)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    batch_size = tf.shape(inputs)[0]
    initial_state = cell.zero_state(batch_size)
    rnn_output = rnn(inputs, initial_state=initial_state)
    regularization_layer = models.SpikeVoltageRegularization(cell)
    voltages, spikes = regularization_layer(rnn_output)
    voltages = tf.identity(voltages, name='voltages')
    spikes = tf.identity(spikes, name='spikes')

    weighted_out_projection = tf.keras.layers.Dense(1)
    weighted_out = weighted_out_projection(spikes)

    prediction = models.exp_convolve(weighted_out, axis=1)
    prediction = tf.identity(prediction, name='output')

    return tf.keras.Model(inputs=inputs, outputs=[voltages, spikes, prediction])

# generate placeholder input data
def create_data_set(seq_len=flags.seq_len, n_input=flags.n_input, n_batch=1):
    x = tf.random.uniform(shape=(seq_len, n_input))[None] * .5
    y = tf.sin(tf.linspace(0., 4 * np.pi, seq_len))[None, :, None]

    return tf.data.Dataset.from_tensor_slices((x, dict(tf_op_layer_output=y))).repeat(count=20).batch(n_batch)


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_example, fig, axes):
        super().__init__()
        self.test_example = test_example
        self.fig = fig
        self.axes = axes

    def on_epoch_end(self, epoch, logs=None):
        output = self.model(self.test_example[0])
        [ax.clear() for ax in self.axes]
        self.axes[0].pcolormesh(self.test_example[0].numpy()[0].T, cmap='cividis')
        self.axes[0].set_ylabel('input')
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        abs_max = np.abs(v).max()
        self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        self.axes[1].set_ylabel('voltage')
        self.axes[2].pcolormesh(z.T, cmap='Greys')
        self.axes[2].set_ylabel('spike')
        self.axes[3].plot(self.test_example[1]['tf_op_layer_output'][0, :, 0], 'k--', lw=2, alpha=.7, label='target')
        self.axes[3].plot(out, 'b', lw=2, alpha=.7, label='prediction')
        self.axes[3].set_ylabel('output')
        self.axes[3].legend(frameon=False)
        [ax.yaxis.set_label_coords(-.05, .5) for ax in self.axes]
        plt.draw()
        plt.pause(.2)


def main():
    model = create_model()
    data_set = create_data_set()
    it = iter(data_set)
    test_example = next(it)

    if flags.do_plot:
        plt.ion()
        fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)
        plot_callback = PlotCallback(test_example, fig, axes)

    # train the model
    opt = tf.keras.optimizers.Adam(lr=flags.learning_rate)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=dict(tf_op_layer_output=mse))
    model.fit(data_set, epochs=flags.n_epochs, callbacks=[plot_callback])

    # analyse the model
    inputs = test_example[0]
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
