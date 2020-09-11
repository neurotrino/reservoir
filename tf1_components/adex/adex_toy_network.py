import datetime
from time import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from models import AdEx
import os
import pickle
import socket

from dynamic_rnn_with_gradients import dynamic_rnn_with_gradients
from toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()
# other settings
#tf.app.flags.DEFINE_integer('n_desired_spikes', 2, 'Number of spikes desired in final 10 ms, say')
tf.app.flags.DEFINE_bool('do_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Create real-time interactive plot')
tf.app.flags.DEFINE_bool('dump', True, 'Dump results')

# training parameters
tf.app.flags.DEFINE_integer('n_batch', 5, 'batch size')
tf.app.flags.DEFINE_integer('n_iter', 2000, 'total number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'Number of time steps in each trial')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Base learning rate.')
#tf.app.flags.DEFINE_float('stop_crit', 0.07, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_integer('print_every', 50, 'Print every')

tf.app.flags.DEFINE_integer('img_width', 28, 'img width')
tf.app.flags.DEFINE_integer('img_height', 28, 'img height')

# training algorithm with BPTT

#################################################
##################### THESE DEFINE ADEX EQUATIONS
#Fw(vm,w,t) = (a.*(vm - EL) - w)./tauw
#Fvm(vm,w,t,I)= ( -gL.*(vm-EL) + gL.*DeltaT.*exp.((vm-thr)./dt) + I - w )./C
####################
# also see AdEx cell description in models

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

# neuron model and simulation parameters
tf.app.flags.DEFINE_float('thr', -50.4 * mVolt, 'threshold at which ADEX neuron spikes')
tf.app.flags.DEFINE_float('EL', -70.6 * mVolt, 'equilibrium potential for leak (all) channels')
tf.app.flags.DEFINE_float('gL', 0.00003 * mSiemens, 'leak conductance')
tf.app.flags.DEFINE_float('a', .000004 * mSiemens, 'adaptative scalar / coupling param')
tf.app.flags.DEFINE_float('deltaT', 2 * mVolt, 'slope factor; quantifies sharpness of spikes')
tf.app.flags.DEFINE_float('tauw', 144 * mSecond, 'tau for adaptation term w')
tf.app.flags.DEFINE_float('C', .000281 * uFarad, 'membrane capacitance')
tf.app.flags.DEFINE_float('dt', 1. * mSecond, 'simulation time step')

tf.app.flags.DEFINE_float('b', 0.0805 * nAmpere, 'amount by which adaptation value w is changed per spike')
tf.app.flags.DEFINE_float('V_reset', -51 * mVolt, 'reset voltage after spike (equal to EL)')

tf.app.flags.DEFINE_float('dampening_factor', 0.2, 'factor that controls amplitude of pseudoderivative')
tf.app.flags.DEFINE_float('reg_rate', 20, 'target firing rate for regularization [Hz]')
tf.app.flags.DEFINE_float('reg_f', .001, 'regularization coefficient for firing rate')

tf.app.flags.DEFINE_integer('n_refrac', 4, 'Number of refractory steps [ms]') # not currently implemented

# setting up for Gerstner version of AdEx
# using different set of equations for membrane voltage update which includes tau and resistance, no capacitance or leak conductance
# changed values for initial bursting
#tf.app.flags.DEFINE_float('tau', 200.0 * mSecond, 'tau for membrane voltage')
#tf.app.flags.DEFINE_float('a', 0.5 * nSiemens, 'adaptative scalar / coupling param')
#tf.app.flags.DEFINE_float('tauw', 100 * mSecond, 'tau for adaptation term w')
#tf.app.flags.DEFINE_float('b', 7.0 * pAmpere, 'amount by which adaptation value w is changed per spike')
#tf.app.flags.DEFINE_float('V_reset', -55 * mVolt, 'resEt voltage after spike')
#tf.app.flags.DEFINE_float('V_rest', -70 * mVolt, 'rest, equal to EL')
#tf.app.flags.DEFINE_float('R', 500 * MOhm, 'resistance')
#tf.app.flags.DEFINE_float('thr', -50 * mVolt, 'threshold at which ADEX neuron spikes, also rheobase')
#deltaT remains the same
#tf.app.flags.DEFINE_float('I_step', 6.5 * nAmpere, 'current step')


# Network parameters
#tau_v = FLAGS.tau_v
thr = FLAGS.thr
tf.app.flags.DEFINE_float('p', .3, 'Connectivity density in main network')
tf.app.flags.DEFINE_float('l1', 1e-2, 'Regularization coefficient for sparse connectivity')
tf.app.flags.DEFINE_float('rewiring_temperature', .0, 'rewiring temperature')
tf.app.flags.DEFINE_integer('n_neurons', 100, 'Number of toy neurons we are testing')
tf.app.flags.DEFINE_integer('n_in', 200, 'Number of inputs to network')
n_neurons = FLAGS.n_neurons
n_in = FLAGS.n_in

# build computational graph
with tf.variable_scope('CellDefinition'):
    # Generate the cell
    # w_in = np.random.randn(FLAGS.n_in, FLAGS.n_neurons) / np.sqrt(FLAGS.n_in)
    # w_in = np.array([[.1]])  # deterministic initialization for now
    # w_in = 0.4 * np.ones((FLAGS.n_in, FLAGS.n_neurons))

    in_signs = np.ones((FLAGS.n_in))
    rec_signs = np.ones((FLAGS.n_neurons))
    rec_signs[80:] = -1*rec_signs[80:]

    cell = AdEx(
        dtype=tf.float32, n_neurons=FLAGS.n_neurons, n_in=FLAGS.n_in,
        n_batch=FLAGS.n_batch, thr=FLAGS.thr, n_refrac=FLAGS.n_refrac, dt=FLAGS.dt,
        dampening_factor=FLAGS.dampening_factor, tauw=FLAGS.tauw, a=FLAGS.a,
        b=FLAGS.b, stop_gradients=False, gL=FLAGS.gL, EL=FLAGS.EL,
        C=FLAGS.C, deltaT=FLAGS.deltaT, V_reset=FLAGS.V_reset, p=FLAGS.p,
        in_signs=in_signs, rec_signs=rec_signs)

    # w_in = tf.tile(cell.w_in_val[None, ...], (FLAGS.n_batch, 1, 1))
    # w_rec = tf.tile(cell.w_rec_val[None, ...], (FLAGS.n_batch, 1, 1))

    #self, dtype, n_neurons, n_in, n_batch, n_refrac, dt, dampening_factor, stop_gradients, w_in, tau, a, tauw, b, V_reset, V_rest, R, thr, deltaT
    """cell = AdExGerstner(dtype=tf.float32, n_neurons=FLAGS.n_neurons, n_in=FLAGS.n_in,
    n_batch = FLAGS.n_batch, n_refrac = FLAGS.n_refrac, dt = FLAGS.dt, dampening_factor = FLAGS.dampening_factor,
    stop_gradients = False, w_in = w_in, tau = FLAGS.tau, a = FLAGS.a,
    tauw = FLAGS.tauw, b = FLAGS.b, V_reset = FLAGS.V_reset, V_rest = FLAGS.V_rest,
    R = FLAGS.R, thr = FLAGS.thr, deltaT = FLAGS.deltaT)"""


results_tensors = dict()
# results_tensors['w_in'] = cell.w_in

results_tensors['w_in_val'] = cell.w_in_val
results_tensors['w_rec_val'] = cell.w_rec_val

cell_zero_state = cell.zero_state(FLAGS.n_batch, tf.float32)
network_state = cell_zero_state
# inputs = tf.placeholder(tf.float32, shape=(FLAGS.n_batch,FLAGS.seq_len,FLAGS.n_in))

image_inputs = tf.placeholder(tf.float32, shape=(FLAGS.n_batch, FLAGS.seq_len, FLAGS.img_width, FLAGS.img_height, 1))

with tf.name_scope('ConvNet'):
    batched_image_inputs = tf.reshape(
        image_inputs, (FLAGS.n_batch * FLAGS.seq_len, FLAGS.img_width, FLAGS.img_height, 1))
    cnn1_out = tf.keras.layers.Conv2D(
        filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(batched_image_inputs)
    cnn1_out = tf.keras.layers.AveragePooling2D(pool_size=2)(cnn1_out)

    cnn2_out = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(cnn1_out)
    cnn2_out = tf.keras.layers.AveragePooling2D(pool_size=2)(cnn2_out)

    flat_cnn2_out = tf.keras.layers.Flatten()(cnn2_out)
    dense_out = tf.keras.layers.Dense(FLAGS.n_in, activation=tf.nn.relu)(flat_cnn2_out)
    inputs_for_adex = tf.reshape(dense_out, (FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in))

with tf.name_scope('SimulateNetwork'):
    outputs, network_state, grad_fun = dynamic_rnn_with_gradients(cell, inputs_for_adex, initial_state=network_state)
    # z - spikes, v - membrane potentials, w - threshold adaptation variables
    s, z = outputs
    v, w = s[..., 0], s[..., 1]
    results_tensors['z'] = z
    results_tensors['w'] = w
    results_tensors['v'] = v
    # I think these aren't the updated weights, they're just what they are
    # results_tensors['w_in'] = w_in
    # results_tensors['w_rec'] = w_rec

#with tf.name_scope('TaskLoss'):
    # define loss as simply regularization loss
    # spike_ct = tf.reduce_mean(tf.reduce_sum(z[:, 200:300,:], axis=1), axis=0)
    # results_tensors['av_spike_ct'] = spike_ct
    # loss_spike_ct = tf.reduce_sum(tf.square(FLAGS.n_desired_spikes - spike_ct))
    # results_tensors['loss_spike_ct'] = loss_spike_ct

# Target rate regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization throughout trial
    v_scaled = (v - FLAGS.thr) / FLAGS.thr
    av = Second * tf.reduce_mean(z, axis=(0, 1)) / FLAGS.dt # rate in spikes/s
    results_tensors['av'] = av
    regularization_coeff = tf.Variable(np.ones(FLAGS.n_neurons) * FLAGS.reg_f, dtype=tf.float32, trainable=False)
    loss_reg_f = tf.reduce_sum(tf.square(av - FLAGS.reg_rate) * regularization_coeff)
    results_tensors['loss_reg'] = loss_reg_f

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    loss = loss_reg_f #+ loss_spike_ct
    results_tensors['loss'] = loss
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    var_list = tf.trainable_variables()

    # This automatically computes the correct gradients in tensorflow?
    learning_signal = tf.zeros_like(z)
    gradient_list = tf.gradients(loss, var_list)
    grad_error_prints = []
    grad_error_assertions = []
    state_gradients = grad_fun(loss)
    pseudo = tf.maximum(1. - tf.nn.relu((v - cell.thr - 40 * mVolt) / (cell.V_reset - cell.thr - 40 * mVolt)), 0.) * cell.dampening_factor
    results_tensors['pseudo'] = pseudo
    results_tensors['grad_v'] = state_gradients.s[..., 0]
    results_tensors['grad_w'] = state_gradients.s[..., 1]
    results_tensors['grad_z'] = state_gradients.z

grads_and_vars = [(g, v) for g, v in zip(gradient_list, var_list)]

with tf.control_dependencies(grad_error_prints + grad_error_assertions):
    # training just involves changing weights - does not yet involve rewiring
    if FLAGS.p < 0.:
        train_step = opt.apply_gradients(grads_and_vars)
    else:
        not_rewired_grads_and_vars = []
        rewired_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name.count('SynapticSampler') == 0:
                not_rewired_grads_and_vars.append((g, v))
            else:
                rewired_grads_and_vars.append((g, v))
        train_step_not_rewired = opt.apply_gradients(not_rewired_grads_and_vars)
        train_step_rewired = rewiring_optimizer_wrapper(
            opt, loss, FLAGS.learning_rate, FLAGS.l1, FLAGS.rewiring_temperature, FLAGS.p,
            var_list=[v for _, v in rewired_grads_and_vars], grads_and_vars=rewired_grads_and_vars)
        train_step = tf.group(train_step_not_rewired, train_step_rewired)

for v in tf.trainable_variables():
    print(v.name, v.shape)

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# later on, create variables for storing results and plots
#if FLAGS.do_plot and FLAGS.interactive_plot:
    #plt.ion()
#if FLAGS.do_plot:

def raster_plot(ax, spikes, linewidth=0.8, time_offset=0, **kwargs):

    n_t, n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike] + time_offset
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([time_offset, n_t + time_offset])
    ax.set_yticks([0, n_n])

def update_plot(axes, results_values):
    [a.clear() for a in axes]
    axes.plot()
    fig.canvas.set_window_title('spike raster')
    data = results_values['z']
    data = data[1]
    raster_plot(axes, data, linewidth=0.3)
    axes.set_ylabel("neuron index")
    axes.set_xlabel("time (ms)")
    return fig

"""
def update_plot(_axes, _values, bi=0):
    [a.clear() for a in _axes]

    x = np.arange(_values['v'].shape[1])

    ax = _axes[0]
    ax.plot(x, _values['v'][bi, :, 0], lw=1, color='b', alpha=.7)
    ax.plot([200, 200], ax.get_ylim(), 'k--', lw=1)
    ax.plot([300, 300], ax.get_ylim(), 'g--', lw=1)
    ax.set_ylabel('$v$ in $mV$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = _axes[1]
    ax.plot(x, _values['w'][bi, :, 0], lw=1, color='r', alpha=.7)
    ax.plot([200, 200], ax.get_ylim(), 'k--', lw=1)
    ax.plot([300, 300], ax.get_ylim(), 'g--', lw=1)
    ax.set_ylabel('$w$ in $nA$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = _axes[2]
    ax.plot(x, _values['pseudo'][bi], 'k', lw=1)
    ax.set_ylabel('$\\frac{\\partial z}{\\partial v}$')

    ax = _axes[3]
    ax.plot(x, _values['grad_z'][bi], 'k', lw=1)
    ax.set_ylabel('$\\frac{d E}{d z}$')

    ax = _axes[4]
    ax.plot(x, _values['grad_v'][bi], 'k', lw=1)
    ax.set_ylabel('$\\frac{d E}{d v}$')

    ax = _axes[5]
    ax.plot(x, _values['grad_w'][bi], 'k', lw=1)
    ax.set_ylabel('$\\frac{d E}{d w}$')

    ax.set_xlabel('time in $ms$')
    [a.yaxis.set_label_coords(-.1, .5) for a in _axes]

    return fig
    """


if socket.gethostname().count('scherr') > 0:
    root_path = '/data'
else:
    root_path = '../data'
os.makedirs(os.path.join(root_path, 'adex_toy_output/fullnet'), exist_ok=True)

# training loop
for k_iter in range(FLAGS.n_iter + 1):
    # each batch trial
    # constant current step input throughout duration
    batch_inputs = np.zeros((FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in))
    #batch_inputs[:,0:200,:] = FLAGS.I_step * np.ones((FLAGS.n_batch, 200, FLAGS.n_in))
    # batch_inputs[:,0:200,:] = np.random.uniform(low=0.0, high=10.0, size=(FLAGS.n_batch, 200, FLAGS.n_in)) * nAmpere * 0 + 10. * nAmpere
    #batch_inputs[:,200:300,:] = np.random.uniform(low=0.0, high=10.0, size=(FLAGS.n_batch, 100, FLAGS.n_in)) * nAmpere * 0 + 5. * nAmpere
    # batch_inputs += 4 * nAmpere
    # randomize for values in FLAGS.n_batch, FLAGS.seq_len-10, FLAGS.n_in
    # results_tensors['inputs'] = batch_inputs
    np_image_inputs = np.ones((FLAGS.n_batch, FLAGS.seq_len, FLAGS.img_width, FLAGS.img_height, 1))

    feed_dict = {
        image_inputs: np_image_inputs
    }

    if k_iter % FLAGS.print_every == 0:
        results_values = sess.run(results_tensors, feed_dict=feed_dict)
        #update_plot(axes, results_values)
        if FLAGS.do_plot:
            f1 = update_plot(axes, results_values)
            #f1.pause(.1)
            # save figure for voltage trace
            f1.savefig(os.path.expanduser(os.path.join(root_path, 'adex_toy_output/fullnet/figs/raster_{}.png'.format(k_iter))), dpi=300)

        rewired_ref_list = ['w_in_val', 'w_rec_val']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)
        empirical_connectivities = [nz / size for nz, size in zip(non_zeros, sizes)]

        print('|> Iteration {} - {}'.format(k_iter, datetime.datetime.now().strftime('%H:%M:%S %b %d, %Y')))
        #print('|  -- spike count loss {:5.3f}'.format(results_values['loss_spike_ct']))
        print('|  -- regularization loss {:5.3f}'.format(results_values['loss_reg']))
        print('|  -- average firing rate (spikes/s) {:5.3f}'.format(np.mean(results_values['av'])))
        # print('|  -- average spike count in final 10 ms {:5.3f}'.format(results_values['av_spike_ct'][0]))
        print(f'|  -- connectivity total {empirical_connectivity:.3f}')
        print(f'|  -- connectivity w_in {empirical_connectivities[0]:.3f} w_rec {empirical_connectivities[1]:.3f}')
        print(f'|  -- non zero weights w_in {non_zeros[0] / sizes[0]:.3f} w_rec {non_zeros[1] / sizes[1]:.3f}')
        print('|' + '_' * 100)

    if FLAGS.dump and k_iter % 100 == 0:
        # save results data
        with open(os.path.join(root_path, 'adex_toy_output/fullnet/file_{}'.format(k_iter)), 'wb') as f:
            pickle.dump(results_values, f)

    sess.run(train_step, feed_dict)
