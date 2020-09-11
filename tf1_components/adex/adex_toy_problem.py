import datetime
from time import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from models import AdEx_Singular
from models import AdExGerstner
import os
import pickle
import socket

from dynamic_rnn_with_gradients import dynamic_rnn_with_gradients

FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()
# other settings
tf.app.flags.DEFINE_integer('n_desired_spikes', 10, 'Number of spikes desired in final 10 ms, say')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Create real-time interactive plot')
tf.app.flags.DEFINE_bool('dump', True, 'Dump results')

# training parameters
tf.app.flags.DEFINE_integer('n_batch', 5, 'batch size')
tf.app.flags.DEFINE_integer('n_iter', 2000, 'total number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'Number of time steps')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Base learning rate.')
#tf.app.flags.DEFINE_float('stop_crit', 0.07, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_integer('print_every', 50, 'Print every')

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
tf.app.flags.DEFINE_float('V_reset', -70.6 * mVolt, 'reset voltage after spike (equal to EL)')

tf.app.flags.DEFINE_float('dampening_factor', 0.001, 'factor that controls amplitude of pseudoderivative')
tf.app.flags.DEFINE_float('reg_rate', 20, 'target firing rate for regularization [Hz]')
tf.app.flags.DEFINE_float('reg_f', 1., 'regularization coefficient for firing rate')

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
tf.app.flags.DEFINE_integer('n_neurons', 4, 'Number of toy neurons we are testing')
tf.app.flags.DEFINE_integer('n_in', 2, 'Number of input channels to toy neuron')
n_neurons = FLAGS.n_neurons
n_in = FLAGS.n_in

# build computational graph
with tf.variable_scope('CellDefinition'):
    # Generate the cell
    # w_in = np.random.randn(FLAGS.n_in, FLAGS.n_neurons) / np.sqrt(FLAGS.n_in)
    # w_in = np.array([[.1]])  # deterministic initialization for now
    w_in = 0.2 * np.random.uniform(size=(FLAGS.n_in, FLAGS.n_neurons))

    cell = AdEx_Singular(dtype=tf.float32, n_neurons=FLAGS.n_neurons, n_in=FLAGS.n_in,
    n_batch=FLAGS.n_batch, thr=FLAGS.thr, n_refrac=FLAGS.n_refrac, dt=FLAGS.dt,
    dampening_factor=FLAGS.dampening_factor, tauw=FLAGS.tauw, a=FLAGS.a,
    b=FLAGS.b, stop_gradients=False, w_in=w_in, gL=FLAGS.gL, EL=FLAGS.EL,
    C=FLAGS.C, deltaT=FLAGS.deltaT, V_reset=FLAGS.V_reset)

    #self, dtype, n_neurons, n_in, n_batch, n_refrac, dt, dampening_factor, stop_gradients, w_in, tau, a, tauw, b, V_reset, V_rest, R, thr, deltaT
    """cell = AdExGerstner(dtype=tf.float32, n_neurons=FLAGS.n_neurons, n_in=FLAGS.n_in,
    n_batch = FLAGS.n_batch, n_refrac = FLAGS.n_refrac, dt = FLAGS.dt, dampening_factor = FLAGS.dampening_factor,
    stop_gradients = False, w_in = w_in, tau = FLAGS.tau, a = FLAGS.a,
    tauw = FLAGS.tauw, b = FLAGS.b, V_reset = FLAGS.V_reset, V_rest = FLAGS.V_rest,
    R = FLAGS.R, thr = FLAGS.thr, deltaT = FLAGS.deltaT)"""


results_tensors = dict()
results_tensors['w_in'] = cell.w_in

cell_zero_state = cell.zero_state(FLAGS.n_batch, tf.float32)
network_state = cell_zero_state
inputs = tf.placeholder(tf.float32, shape=(FLAGS.n_batch,FLAGS.seq_len,FLAGS.n_in))

with tf.name_scope('SimulateNetwork'):
    outputs, network_state, grad_fun = dynamic_rnn_with_gradients(cell, inputs, initial_state=network_state)
    # z - spikes, v - membrane potentials, w - threshold adaptation variables
    s, z = outputs
    v, w = s[..., 0], s[..., 1]
    results_tensors['z'] = z
    results_tensors['w'] = w
    results_tensors['v'] = v

with tf.name_scope('TaskLoss'):
    # define loss as mean squared difference between desired and actual spike ct in last 10 ms
    spike_ct = tf.reduce_mean(tf.reduce_sum(z[:, 200:300,:], axis=1), axis=0)
    results_tensors['av_spike_ct'] = spike_ct
    loss_spike_ct = tf.reduce_sum(tf.square(FLAGS.n_desired_spikes - spike_ct))
    results_tensors['loss_spike_ct'] = loss_spike_ct

# Target rate regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    v_scaled = (v - FLAGS.thr) / FLAGS.thr
    av = Second * tf.reduce_mean(z, axis=(0, 1)) / FLAGS.dt
    results_tensors['av'] = av
    regularization_coeff = tf.Variable(np.ones(FLAGS.n_neurons) * FLAGS.reg_f, dtype=tf.float32, trainable=False)
    loss_reg_f = tf.reduce_sum(tf.square(av - FLAGS.reg_rate) * regularization_coeff)
    results_tensors['loss_reg'] = loss_reg_f

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    loss = loss_reg_f + loss_spike_ct
    results_tensors['loss'] = loss
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    var_list = [cell.w_in]

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
    train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

for v in tf.trainable_variables():
    print(v.name, v.shape)

# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# later on, create variables for storing results and plots
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, axes = plt.subplots(6, figsize=(8, 6), sharex=True)
    # fig.canvas.set_window_title('adex voltage trace')


def update_plot(_axes, _values, bi=0):
    [a.clear() for a in _axes]

    x = np.arange(_values['v'].shape[1])

    ax = _axes[0]
    ax.plot(x, _values['v'][bi], lw=1, color='b', alpha=.7)
    ax.plot([200, 200], ax.get_ylim(), 'k--', lw=1)
    ax.plot([300, 300], ax.get_ylim(), 'g--', lw=1)
    ax.set_ylabel('$v$ in $mV$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = _axes[1]
    ax.plot(x, _values['w'][bi], lw=1, color='r', alpha=.7)
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


if socket.gethostname().count('scherr') > 0:
    root_path = '/data'
else:
    root_path = '../data'

# training loop
for k_iter in range(FLAGS.n_iter + 1):
    # each batch trial
    # constant current step input throughout duration
    batch_inputs = np.zeros((FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in))
    #batch_inputs[:,0:200,:] = FLAGS.I_step * np.ones((FLAGS.n_batch, 200, FLAGS.n_in))
    batch_inputs[:,0:200,:] = np.random.uniform(low=0.0, high=10.0, size=(FLAGS.n_batch, 200, FLAGS.n_in)) * nAmpere * 0 + 10. * nAmpere
    batch_inputs[:,200:300,:] = np.random.uniform(low=0.0, high=10.0, size=(FLAGS.n_batch, 100, FLAGS.n_in)) * nAmpere * 0 + 5. * nAmpere
    # batch_inputs += 4 * nAmpere
    # randomize for values in FLAGS.n_batch, FLAGS.seq_len-10, FLAGS.n_in
    # results_tensors['inputs'] = batch_inputs

    feed_dict = {
        inputs: batch_inputs
    }

    if k_iter % FLAGS.print_every == 0:
        results_values = sess.run(results_tensors, feed_dict=feed_dict)
        #update_plot(axes, results_values)
        if FLAGS.do_plot:
            f1 = update_plot(axes, results_values)
            plt.draw()
            plt.pause(.1)
            # save figure for voltage trace
            # f1.savefig(os.path.expanduser(os.path.join(root_path, 'adex_toy_output/v6/figs/trace_{}.png'.format(k_iter))), dpi=300)

        print('|> Iteration {} - {}'.format(k_iter, datetime.datetime.now().strftime('%H:%M:%S %b %d, %Y')))
        print('|  -- spike count loss {:5.3f}'.format(results_values['loss_spike_ct']))
        print('|  -- regularization loss {:5.3f}'.format(results_values['loss_reg']))
        print('|  -- average firing rate (spikes/s) {:5.3f}'.format(np.mean(results_values['av'])))
        print('|  -- average spike count in final 10 ms {:5.3f}'.format(results_values['av_spike_ct'][0]))
        print('|' + '_' * 100)

    if FLAGS.dump and k_iter % 100 == 0:
        # save results data
        with open(os.path.join(root_path, 'adex_toy_output/v6/file_{}'.format(k_iter)), 'wb') as f:
            pickle.dump(results_values, f)

    sess.run(train_step, feed_dict)
