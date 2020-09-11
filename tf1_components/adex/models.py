import tensorflow as tf
import numpy as np
from collections import namedtuple
from toolbox.rewiring_tools import weight_sampler

Cell = tf.contrib.rnn.BasicRNNCell
FLAGS = tf.app.flags.FLAGS

def pseudo_derivative(v_scaled, dampening_factor):
    """
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    """
    # return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor
    return tf.maximum(1 - tf.nn.relu(-v_scaled), 0) * dampening_factor


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    """
    The tensorflow function which is defined as a Heaviside function (to compute the spikes),
    but with a gradient defined with the pseudo derivative.
    :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
    :param dampening_factor: parameter to stabilize learning
    :return: the spike tensor
    """
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


class AdExGerstner(Cell):
    # redefine update equations based on Ch. 6 of Gerstner et al. 2014, Neuronal Dynamics
    State = namedtuple('AdExGerstnerStateTuple', ('s','z'))
    def __init__(self, dtype, n_neurons, n_in, n_batch, n_refrac, dt, dampening_factor, stop_gradients, w_in, tau, a, tauw, b, V_reset, V_rest, R, thr, deltaT):
        """
        This AdEx model is based off of the update
            dV/dt = (-(V-V_rest) + deltaT*exp((V-thr)/deltaT) - R*w + R*I)/tau
            dw/dt = (a*(V-V_rest) - w)/tauw
        After each spike, w = w+b and V = V_reset
        For the task envisioned now, there will simply be a current step input
        """

        if tauw is None: raise ValueError("time constant for adaptive bias must be set")
        if a is None: raise ValueError("a parameter for adaptive bias must be set")

        dtype = tf.float32

        self.n_neurons = n_neurons
        self.n_in = n_in              #Input neurons
        self.n_batch = n_batch
        self.n_refrac = n_refrac      #Number of refractory time steps
        self.dt = dt                #Length of time step
        self.dampening_factor = dampening_factor
        self.tau = tau
        self.tauw = tauw # adaptation term w's time constant
        self.a = a              #Adaptation coupling parameter
        self.b = b              #Spike-triggered adaptation parameter added to w
        self.stop_gradients = stop_gradients
        self.thr = thr
        self.reset_thr = self.thr + 100.
        self.w_in = tf.Variable(w_in.astype(np.float32)) # make input a tensorflow variable
        self.deltaT = deltaT
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.R = R

    @property
    def state_size(self):
        return AdExGerstner.State(s=tf.TensorShape((self.n_neurons, 2)), z=self.n_neurons)

    @property
    def output_size(self):
        return [tf.TensorShape((self.n_neurons, 2)), self.n_neurons]

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) + self.V_rest # begin at V_reset mV
        w0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) # begin at 0 nA
        s0 = tf.stack((v0, w0), -1)
        z0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype)

        return AdExGerstner.State(s=s0, z=z0)

    def compute_z(self, v):
        @tf.custom_gradient
        def new_spike_function(_v, _v0, _exp_thr, _thr, _damp):
            _z = _v > _thr
            _z = tf.cast(_z, tf.float32)

            def grad(de_dy):
                de_dv = de_dy * tf.maximum(1. - tf.nn.relu((_v - _thr - 40) / (_v0 - _thr - 40)), 0.) * _damp
                return [de_dv, tf.zeros_like(_v0), tf.zeros_like(_exp_thr), tf.zeros_like(_thr), tf.zeros_like(_damp)]
            return _z, grad

        z = new_spike_function(v, self.V_reset, self.thr, self.reset_thr, self.dampening_factor)
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        s = state.s
        old_z = state.z
        v, w = s[..., 0], s[..., 1] #v is the first, w (adaptation) is the second

        i_in = tf.matmul(inputs, self.w_in)

        #Update information
        i_t             = i_in           #supplied current
        #i_reset         = z * self.thr * self.dt

        # clipping of this term becomes very important
        # VERY IMPORTANT FOR TRAINING: THIS NONLINEARITY WILL MAKE IT ESSENTIAL TO REDESIGN THE PSEUDO-DERIVATIVE
        # IN A SIMILAR NONLINEAR MANNER TOO.
        # dV/dt = (-(V-V_rest) + deltaT*exp((V-thr)/deltaT) - R*w + R*I)/tau
        # dw/dt = (a*(V-V_rest) - w)/tauw

        exp_terms = tf.clip_by_value(self.deltaT * tf.exp((v - self.thr)/self.deltaT), -1e6, 30 / self.tau)
        # exp_terms       = tf.clip_by_value(self.gL * self.deltaT * tf.exp((v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)
        exp_terms = tf.stop_gradient(exp_terms)

        #Update steps
        new_v = v + (-v - self.V_rest)/self.tau + exp_terms + (i_t - w)*self.R/self.tau
        #new_v = v - self.dt_gL__C * (v - self.EL) + self.dt_gL__C * exp_terms + (i_t - w) * self.dt / self.C
        new_w = w + (self.a*(v - self.V_rest) - w)/self.tauw
        #new_w = w - self.dt / self.tauw * w + self.dt_a__tauw * (v - self.EL)

        # If old_z=1, meaning a spike occurred, new_w = new_w + b
        # otherwise it will run (decay) according to dynamics
        # (there is no "reset" for adaptation, just a set change per spike)
        new_w += self.b * old_z

        # If old_z = 1, new_v becomes V_reset. If not, it runs according to dynamics
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)
        # new_v = old_z * self.V_reset + (1 - old_z) * new_v

        new_s = tf.stack((new_v, new_w), axis=-1)  # recompose v and w into state
        new_z = self.compute_z(new_v)

        new_state = AdExGerstner.State(s=new_s, z=new_z)
        return [new_s, new_z], new_state



class AdEx(Cell):
    State = namedtuple('AdExStateTuple', ('s','z'))
    # May 19th to accommodate a network of all these units, including rewiring
    def __init__(self, dtype, n_neurons, n_in, n_batch, thr, n_refrac, dt, dampening_factor, tauw, a, b, stop_gradients, w_in, gL, EL, C, deltaT, V_reset, p, in_signs, rec_signs):

        """
        An AdEx model whose internal state s = [v,w], where v is voltage, w is spike adaptation variable
        Skeleton based off of the CustomALIF model
        Cell output is a tuple: s (states): n_batch x n_neurons x 2 dims,
                                z (spikes): n_batch x n_neurons,
                                diag_j (neuron specific jacobian, partial z / partial s): (n_batch x n_neurons x 2 x 2,
                                                                                           n_batch x n_neurons x 2),
                                partials_wrt_biases (partial s / partial inputs, recurrent spikes):
                                        (n_batch x n_neurons x 2, n_batch x n_neurons x 2)

        diag_j is a tuple of two matrices. diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]
            dnew_s_ds gives the change in s_j^(t) with respect to s_{j-1}^(t), needed for eligibility vectors
            (this is for the voltages and for the adaptation term w)
            dnew_z_dnew_s gives the change in voltage with respect to the new internal state

        partials_wrt_biases: I am not sure yet.

        The AdEx model is based off of the update
            dV/dt = (-gL(V-EL) + gL*deltaT*exp((V-thr)/dt) - w + I)/C
            dw/dt = (a*(V-EL)-w)/tauw

        Furthermore, after each spike, adaptation variable w = w+b and v = V_reset
        For this task the toy neuron will be receiving current input from n_in sources
        Params (w_in) are updated to minimize spike ct mismatch in final 10 ms

        :param dtype: data type of tensors
        :param n_neurons: number of toy AdEx neurons
        :param n_in: number of input neurons
        :param n_batch: number of trials per batch update
        :param thr: spike threshold
        :param n_refrac: number of refractory time steps after spike
        :param dt: length of discrete time steps in ms
        :param dampening_factor: used in pseudo-derivative
        :param tauw: adaptation time constant
        :param a: adaptive scalar / coupling param
        :param b: amount of change to adaptation param w following spike
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in: initial weights for input connections
        :param gL: leak conductance
        :param EL: leak reversal potential
        :param C: membrane capacitance
        :param deltaT: slope factor (sharpness of spikes)
        :param V_reset: rest voltage after spike
        :param p: main network connection probability / density
        :param in_signs: e or i identity of input connections
        :param rec_signs: e or i identity of recurrent main connections
        """

        if tauw is None: raise ValueError("time constant for adaptive bias must be set")
        if a is None: raise ValueError("a parameter for adaptive bias must be set")

        dtype = tf.float32

        #General parameters
        #self.dtype = dtype
        self.n_neurons = n_neurons
        self.n_in = n_in              #Input neurons
        self.n_batch = n_batch
        self.thr = thr
        self.n_refrac = n_refrac      #Number of refractory time steps
        self.dt = dt                #Length of time step
        self.dampening_factor = dampening_factor
        self.tauw = tauw # adaptation term w's time constant
        self.a = a              #Adaptation coupling parameter
        self.b = b              #Spike-triggered adaptation parameter added to w
        self.stop_gradients = stop_gradients
        # self.w_in = tf.convert_to_tensor(w_in.astype(np.float32))
        self.reset_thr = self.thr + 100.
        self.w_in = tf.Variable(w_in.astype(np.float32)) # make input a tensorflow variable
        self.gL = gL #Leak conductance
        self.EL = EL #Leak reversal potential
        self.C = C   #Membrane Capacitance
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.in_signs = in_signs
        self.rec_signs = rec_signs
        self.p = p
        self.dt_gL__C = self.dt * self.gL / self.C
        self.dt_a__tauw = self.dt * self.a / self.tauw

        with tf.variable_scope('input_weights'):
            self.w_in_val, self.w_in_var, th, self.w_in_is_connected = \
                weight_sampler(n_in=self.n_in, n_out=self.n_neurons, p=self.p,
                               dtype=tf.float32, neuron_sign=self.in_signs)

            #self.w_in_val = self.w_in

        with tf.variable_scope('recurrent_weights'):
            self.w_rec_val, self.w_rec_var, th, self.w_rec_is_connected = \
                weight_sampler(n_in=self.n_neurons, n_out=self.n_neurons, p=self.p,
                               dtype=tf.float32, neuron_sign=self.rec_signs)
            # disconnect autapse
            self.recurrent_disconnect_mask = np.diag(np.ones(self.n_neurons, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)

    @property
    def state_size(self):
        return AdEx.State(s=tf.TensorShape((self.n_neurons, 2)), z=self.n_neurons)

    def set_weights(self, w_in, w_rec, signs):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_neurons, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        w_in_is_connected = self.w_in_is_connected
        w_rec_is_connected = self.w_rec_is_connected
        if w_rec_rank == 3: # batch dim
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))
            w_in_is_connected = tf.tile(w_in_is_connected[None, ...], (n_batch, 1, 1))
            w_rec_is_connected = tf.tile(w_rec_is_connected[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(tf.logical_or(tf.logical_not(w_rec_is_connected), recurrent_disconnect_mask),
                                  tf.zeros_like(w_rec), w_rec)
        self.w_in_val = tf.where(tf.logical_not(w_in_is_connected), tf.zeros_like(w_in), w_in)

        # clip weights
        thresh = signs[:, None] * self.w_rec_val
        thresh = tf.nn.relu(thresh)
        self.w_rec_val = signs[:, None] * thresh

    @property
    def output_size(self):
        return [tf.TensorShape((self.n_neurons, 2)), self.n_neurons]

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) + self.V_reset # begin at V_reset mV
        # the w here refers to adaptation variable, NOT weights
        w0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) # begin at 0 nA
        s0 = tf.stack((v0, w0), -1)
        z0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype)

        return AdEx.State(s=s0, z=z0)

    def compute_z(self, v):
        @tf.custom_gradient
        def new_spike_function(_v, _v0, _exp_thr, _thr, _damp):
            _z = _v > _thr
            _z = tf.cast(_z, tf.float32)

            def grad(de_dy):
                de_dv = de_dy * tf.maximum(1. - tf.nn.relu((_v - _thr - 40) / (_v0 - _thr - 40)), 0.) * _damp
                return [de_dv, tf.zeros_like(_v0), tf.zeros_like(_exp_thr), tf.zeros_like(_thr), tf.zeros_like(_damp)]
            return _z, grad

        z = new_spike_function(v, self.V_reset, self.thr, self.reset_thr, self.dampening_factor)
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        s = state.s
        old_z = state.z
        v, w = s[..., 0], s[..., 1] #v is the first, w (adaptation) is the second

        i_in = tf.matmul(inputs, self.w_in_val) + tf.matmul(tf.zeros_like(old_z), self.w_rec_val)

        #Update information
        i_t             = i_in           #supplied current
        #i_reset         = z * self.thr * self.dt

        # clipping of this term becomes very important
        # VERY IMPORTANT FOR TRAINING: THIS NONLINEARITY WILL MAKE IT ESSENTIAL TO REDESIGN THE PSEUDO-DERIVATIVE
        # IN A SIMILAR NONLINEAR MANNER TOO.
        exp_terms       = tf.clip_by_value(self.gL * self.deltaT * tf.exp((v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)
        exp_terms = tf.stop_gradient(exp_terms)

        #Update steps
        new_v = v - self.dt_gL__C * (v - self.EL) + self.dt_gL__C * exp_terms + (i_t - w) * self.dt / self.C
        new_w = w - self.dt / self.tauw * w + self.dt_a__tauw * (v - self.EL)

        # If old_z=1, meaning a spike occurred, new_w = new_w + b
        # otherwise it will run (decay) according to dynamics
        # (there is no "reset" for adaptation, just a set change per spike)
        new_w += self.b * old_z

        # If old_z = 1, new_v becomes V_reset. If not, it runs according to dynamics
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)
        # new_v = old_z * self.V_reset + (1 - old_z) * new_v

        new_s = tf.stack((new_v, new_w), axis=-1)  # recompose v and w into state
        new_z = self.compute_z(new_v)

        new_state = AdEx.State(s=new_s, z=new_z)
        return [new_s, new_z], new_state


class AdEx_Singular(Cell):
    State = namedtuple('AdExStateTuple', ('s','z'))
    # first iteration, April 6th, no hard refractory period (i.e. no original variable 'r')
    def __init__(self, dtype, n_neurons, n_in, n_batch, thr, n_refrac, dt, dampening_factor, tauw, a, b, stop_gradients, w_in, gL, EL, C, deltaT, V_reset):

        """
        An AdEx model whose internal state s = [v,w], where v is voltage, w is spike adaptation variable
        Skeleton based off of the CustomALIF model
        Cell output is a tuple: s (states): n_batch x n_neurons x 2 dims,
                                z (spikes): n_batch x n_neurons,
                                diag_j (neuron specific jacobian, partial z / partial s): (n_batch x n_neurons x 2 x 2,
                                                                                           n_batch x n_neurons x 2),
                                partials_wrt_biases (partial s / partial inputs, recurrent spikes):
                                        (n_batch x n_neurons x 2, n_batch x n_neurons x 2)

        diag_j is a tuple of two matrices. diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]
            dnew_s_ds gives the change in s_j^(t) with respect to s_{j-1}^(t), needed for eligibility vectors
            (this is for the voltages and for the adaptation term w)
            dnew_z_dnew_s gives the change in voltage with respect to the new internal state

        partials_wrt_biases: I am not sure yet.

        The AdEx model is based off of the update
            dV/dt = (-gL(V-EL) + gL*deltaT*exp((V-thr)/dt) - w + I)/C
            dw/dt = (a*(V-EL)-w)/tauw

        Furthermore, after each spike, adaptation variable w = w+b and v = V_reset
        For this task the toy neuron will be receiving current input from n_in sources
        Params (w_in) are updated to minimize spike ct mismatch in final 10 ms

        :param dtype: data type of tensors
        :param n_neurons: number of toy AdEx neurons
        :param n_in: number of input neurons
        :param n_batch: number of trials per batch update
        :param thr: spike threshold
        :param n_refrac: number of refractory time steps after spike
        :param dt: length of discrete time steps in ms
        :param dampening_factor: used in pseudo-derivative
        :param tauw: adaptation time constant
        :param a: adaptive scalar / coupling param
        :param b: amount of change to adaptation param w following spike
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in: initial weights for input connections
        :param gL: leak conductance
        :param EL: leak reversal potential
        :param C: membrane capacitance
        :param deltaT: slope factor (sharpness of spikes)
        :param V_reset: rest voltage after spike
        """

        if tauw is None: raise ValueError("time constant for adaptive bias must be set")
        if a is None: raise ValueError("a parameter for adaptive bias must be set")

        dtype = tf.float32

        #General parameters
        #self.dtype = dtype
        self.n_neurons = n_neurons
        self.n_in = n_in              #Input neurons
        self.n_batch = n_batch
        self.thr = thr
        self.n_refrac = n_refrac      #Number of refractory time steps
        self.dt = dt                #Length of time step
        self.dampening_factor = dampening_factor
        self.tauw = tauw # adaptation term w's time constant
        self.a = a              #Adaptation coupling parameter
        self.b = b              #Spike-triggered adaptation parameter added to w
        self.stop_gradients = stop_gradients
        # self.w_in = tf.convert_to_tensor(w_in.astype(np.float32))
        self.reset_thr = self.thr + 100.
        self.w_in = tf.Variable(w_in.astype(np.float32)) # make input a tensorflow variable
        with tf.variable_scope('recurrent_weights'):
            # self.w_rec_val, self.w_rec_var, th, self.w_rec_is_connected = \
            #     weight_sampler(n_in=self.n_neurons, n_out=self.n_neurons, p=self.p,
            #                    dtype=tf.float32, neuron_sign=self.rec_signs)
            # disconnect autapse
            init_w_rec_var = (np.random.randn(self.n_neurons, self.n_neurons) / np.sqrt(self.n_neurons)).astype(np.float32)
            self.w_rec_var = tf.Variable(init_w_rec_var * .01)
            self.recurrent_disconnect_mask = np.diag(np.ones(self.n_neurons, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var), self.w_rec_var)
        self.gL = gL #Leak conductance
        self.EL = EL #Leak reversal potential
        self.C = C   #Membrane Capacitance
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.dt_gL__C = self.dt * self.gL / self.C
        self.dt_a__tauw = self.dt * self.a / self.tauw

    @property
    def state_size(self):
        return AdEx_Singular.State(s=tf.TensorShape((self.n_neurons, 2)), z=self.n_neurons)

    @property
    def output_size(self):
        return [tf.TensorShape((self.n_neurons, 2)), self.n_neurons]

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) + self.V_reset # begin at V_reset mV
        w0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype) # begin at 0 nA
        s0 = tf.stack((v0, w0), -1)
        z0 = tf.zeros(shape=(batch_size, self.n_neurons), dtype=dtype)

        return AdEx_Singular.State(s=s0, z=z0)

    def compute_z(self, v):
        @tf.custom_gradient
        def new_spike_function(_v, _v0, _exp_thr, _thr, _damp):
            _z = _v > _thr
            _z = tf.cast(_z, tf.float32)

            def grad(de_dy):
                de_dv = de_dy * tf.maximum(1. - tf.nn.relu((_v - _thr - 25) / (_v0 - _thr - 25)), 0.) * _damp
                return [de_dv, tf.zeros_like(_v0), tf.zeros_like(_exp_thr), tf.zeros_like(_thr), tf.zeros_like(_damp)]
            return _z, grad

        z = new_spike_function(v, self.V_reset, self.thr, self.reset_thr, self.dampening_factor)
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        s = state.s
        old_z = state.z
        v, w = s[..., 0], s[..., 1] #v is the first, w (adaptation) is the second

        i_in = tf.matmul(inputs, self.w_in)  # + tf.matmul(tf.zeros_like(old_z), self.w_rec_val)
        i_in_rec = tf.matmul(old_z, self.w_rec_val) * 0.
        # op = tf.print(tf.reduce_max(i_in_rec))
        op = tf.no_op()

        #Update information
        i_t             = i_in + i_in_rec           #supplied current
        #i_reset         = z * self.thr * self.dt

        # clipping of this term becomes very important
        # VERY IMPORTANT FOR TRAINING: THIS NONLINEARITY WILL MAKE IT ESSENTIAL TO REDESIGN THE PSEUDO-DERIVATIVE
        # IN A SIMILAR NONLINEAR MANNER TOO.
        with tf.control_dependencies([op]):
            exp_terms       = tf.clip_by_value(self.gL * self.deltaT * tf.exp((v - self.thr)/self.deltaT), -1e6, 30 / self.dt_gL__C)
        exp_terms = tf.stop_gradient(exp_terms)

        #Update steps
        new_v = v - self.dt_gL__C * (v - self.EL) + self.dt_gL__C * exp_terms + (i_t - w) * self.dt / self.C
        new_w = w - self.dt / self.tauw * w + self.dt_a__tauw * (v - self.EL)

        # If old_z=1, meaning a spike occurred, new_w = new_w + b
        # otherwise it will run (decay) according to dynamics
        # (there is no "reset" for adaptation, just a set change per spike)
        new_w += self.b * old_z

        # If old_z = 1, new_v becomes V_reset. If not, it runs according to dynamics
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)
        # new_v = old_z * self.V_reset + (1 - old_z) * new_v

        new_s = tf.stack((new_v, new_w), axis=-1)  # recompose v and w into state
        new_z = self.compute_z(new_v)

        new_state = AdEx_Singular.State(s=new_s, z=new_z)
        return [new_s, new_z], new_state


class LIF(Cell):
    State = namedtuple('LIFStateTuple', ('v', 'z', 'r'))

    def __init__(self, n_in, n_rec, p, tau=20., thr=0.4, dt=1., dtype=tf.float32, dampening_factor=0.3, n_refractory=5):

        self.dampening_factor = dampening_factor
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self.decay = tf.exp(-dt / tau)
        self.thr = thr
        self.stop_gradient = False
        self.n_refractory = n_refractory

        with tf.variable_scope('input_weights'):
            init_w_in_var = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
            init_w_in_var = tf.cast(init_w_in_var, dtype)
            self.w_in_var = tf.get_variable('w', initializer=init_w_in_var)
            self.w_in_val = self.w_in_var
            #self.w_in_val, self.w_in_var, th, is_connected = weight_sampler(n_in=self.n_in, n_out=self.n_rec, p=p)

        with tf.variable_scope('recurrent_weights'):
            init_w_rec_var = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)
            init_w_rec_var = tf.cast(init_w_rec_var, dtype)
            self.w_rec_var = tf.get_variable('w', initializer=init_w_rec_var)
            self.w_rec_val = self.w_rec_var
            #self.w_rec_val, self.w_rec_var, th, is_connected = weight_sampler(n_in=self.n_rec, n_out=self.n_rec, p=p)

            # disconnect autapse
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)

    @property
    def state_size(self):
        return LIF.State(v=self.n_rec, z=self.n_rec, r=self.n_rec)

    @property
    def output_size(self):
        return self.n_rec, self.n_rec

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)

        return LIF.State(v=v0, z=z0, r=r0)

    def set_weights(self, w_in, w_rec):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        if w_rec_rank == 3:
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(w_rec), w_rec)
        self.w_in_val = w_in

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        w_in_has_batch_dim = len(self.w_in_val.get_shape().as_list()) == 3
        w_rec_has_batch_dim = len(self.w_rec_val.get_shape().as_list()) == 3

        previous_z = state.z

        if w_in_has_batch_dim:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            i_in = tf.matmul(inputs, self.w_in_val)

        if w_rec_has_batch_dim:
            i_rec = tf.einsum('bi,bij->bj', previous_z, self.w_rec_val)
        else:
            i_rec = tf.matmul(previous_z, self.w_rec_val)

        i_reset = previous_z * self.thr * self.dt
        new_v = self.decay * state.v + (i_in + i_rec) - i_reset

        # Spike generation
        v_scaled = (new_v - self.thr) / self.thr
        new_z = spike_function(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt

        # check if refractory
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, new_z)
        new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                 0., float(self.n_refractory))

        new_state = LIF.State(v=new_v, z=new_z, r=new_r)

        return (new_z, new_v), new_state


CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r'))

class CustomALIF(Cell):
    def __init__(self, p, n_in, n_rec_e, n_rec_i, rec_signs, in_signs, tau_e=20., tau_i=20., thr=.4, n_refractory=5, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16,
                 stop_gradients=False, w_in_init=None, w_rec_init=None):
        """
        CustomALIF provides the recurrent tensorflow cell model for implementing LSNNs in combination with
        eligibility propagation (e-prop).

        Cell output is a tuple: z (spikes): n_batch x n_neurons,
                                s (states): n_batch x n_neurons x 2,
                                diag_j (neuron specific jacobian, partial z / partial s): (n_batch x n_neurons x 2 x 2,
                                                                                           n_batch x n_neurons x 2),
                                partials_wrt_biases (partial s / partial input_current for inputs, recurrent spikes):
                                        (n_batch x n_neurons x 2, n_batch x n_neurons x 2)

        UPDATE: This model uses v^{t+1} ~ alpha * v^t + i_t instead of ... + (1 - alpha) * i_t
                it is therefore required to rescale thr, and beta of older version by

                thr = thr_old / (1 - exp(- 1 / tau))
                beta = beta_old * (1 - exp(- 1 / tau_adaptation)) / (1 - exp(- 1 / tau))

        UPDATE: refractory periods are implemented

        :param n_in: number of input neurons
        :param n_rec: number of output neurons
        :param tau: membrane time constant
        :param thr: spike threshold
        :param dt: length of discrete time steps
        :param dtype: data type of tensors
        :param dampening_factor: used in pseudo-derivative
        :param tau_adaptation: time constant of adaptive threshold decay
        :param beta: impact of adapting thresholds
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in_init: initial weights for input connections
        :param w_rec_init: initial weights for recurrent connections
        :param n_refractory: number of refractory time steps
        """

        self.n_refractory = n_refractory

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)
        self.dampening_factor = dampening_factor
        self.stop_gradients = stop_gradients
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec_e + n_rec_i
        self.n_rec_e = n_rec_e
        self.data_type = dtype

        if np.isscalar(tau_e): tau_e = tf.ones(n_rec_e, dtype=dtype) * np.mean(tau_e)
        if np.isscalar(tau_i): tau_i = tf.ones(n_rec_i, dtype=dtype) * np.mean(tau_i)
        if np.isscalar(thr): thr = tf.ones(self.n_rec, dtype=dtype) * np.mean(thr)

        tau_e = tf.cast(tau_e, dtype=dtype)
        tau_i = tf.cast(tau_i, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self._num_units = self.n_rec

        self.tau_e = tau_e
        self.tau_i = tau_i
        self.decay_e = tf.exp(-dt / tau_e)
        self.decay_i = tf.exp(-dt / tau_i)
        self.decay = tf.concat((tf.ones((n_rec_e,)) * self.decay_e, tf.ones((n_rec_i,)) * self.decay_i), 0)
        self.thr = thr

        with tf.variable_scope('InputWeights'):
            # in_signs is vector of appropriate signs (of combined inputs) of length n_in
            #self.w_in_val, self.w_in_var = get_signed_weights_reflect(in_signs, self.n_rec)
            self.w_in_val, self.w_in_var, th, self.w_in_is_connected = \
                weight_sampler(n_in=self.n_in, n_out=self.n_rec, p=p, dtype=tf.float32, neuron_sign=in_signs)

        with tf.variable_scope('RecWeights'):
            #self.w_rec_val, self.w_rec_var = get_signed_weights_reflect(rec_signs,self.n_rec)
            self.w_rec_val, self.w_rec_var, th, self.w_rec_is_connected = \
                weight_sampler(n_in=self.n_rec, n_out=self.n_rec, p=p,
                               dtype=tf.float32, neuron_sign=rec_signs)
            # disconnect autapse
            self.recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)

        #with tf.variable_scope('InputWeights'):
            # Input weights, modified s.t. sparse and supports rewiring
            #init_w_in_var = w_in_init if w_in_init is not None else \
                #(np.random.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
            #self.w_in_var = tf.get_variable("InputWeight", initializer=init_w_in_var, dtype=dtype)
            #self.w_in_val = self.w_in_var

        #with tf.variable_scope('RecWeights'):
            # Recurrent weights, modified s.t. sparse and supports rewiring
            #init_w_rec_var = w_rec_init if w_rec_init is not None else \
                #(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
            #self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
            #self.w_rec_val = self.w_rec_var

        # TODO: Think about how to handle that more cleverly for the case with rewiring with sparse tensors
        dw_val_dw_var_in = np.ones((n_in,self._num_units))
        dw_val_dw_var_rec = np.ones((self._num_units,self._num_units)) - np.diag(np.ones(self._num_units))
        self.dw_val_dw_var = [dw_val_dw_var_in,dw_val_dw_var_rec]

        self.variable_list = [self.w_in_var,self.w_rec_var]
        self.built = True

    @property
    def state_size(self):
        return CustomALIFStateTuple(s=tf.TensorShape((self.n_rec, 2)), z=self.n_rec, r=self.n_rec)

    def set_weights(self, w_in, w_rec, signs):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        w_in_is_connected = self.w_in_is_connected
        w_rec_is_connected = self.w_rec_is_connected
        if w_rec_rank == 3:
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))
            w_in_is_connected = tf.tile(w_in_is_connected[None, ...], (n_batch, 1, 1))
            w_rec_is_connected = tf.tile(w_rec_is_connected[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(tf.logical_or(tf.logical_not(w_rec_is_connected), recurrent_disconnect_mask),
                                  tf.zeros_like(w_rec), w_rec)
        self.w_in_val = tf.where(tf.logical_not(w_in_is_connected), tf.zeros_like(w_in), w_in)

        # clip weights
        thresh = signs[:, None] * self.w_rec_val
        thresh = tf.nn.relu(thresh)
        self.w_rec_val = signs[:, None] * thresh


        #recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        #w_rec_rank = len(w_rec.get_shape().as_list())
        #if w_rec_rank == 3:
            #n_batch = tf.shape(w_rec)[0]
            #recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))

        #self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(w_rec), w_rec)
        #self.w_in_val = w_in

    @property
    def output_size(self):
        return [self.n_rec, tf.TensorShape((self.n_rec, 2)), self.n_rec,
                [tf.TensorShape((self.n_rec, 2, 2)), tf.TensorShape((self.n_rec, 2))],
                [tf.TensorShape((self.n_rec, 2))] * 2]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)  # state contains both v and b for each neuron
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return CustomALIFStateTuple(s=s0, z=z0, r=r0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = spike_function(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        decay = self.decay

        z = state.z
        s = state.s
        v, b = s[..., 0], s[..., 1]

        old_z = self.compute_z(v, b)

        if self.stop_gradients:
            z = tf.stop_gradient(z)

        new_b = self.decay_b * b + old_z
        # TODO: solve the problem that z, s and b don't have the right dependencies

        if len(self.w_in_val.get_shape().as_list()) == 3:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            i_in = tf.matmul(inputs, self.w_in_val)
        if len(self.w_rec_val.get_shape().as_list()) == 3:
            i_rec = tf.einsum('bi,bij->bj', z, self.w_rec_val)
        else:
            i_rec = tf.matmul(z, self.w_rec_val)

        i_t = i_in + i_rec
        I_reset = z * self.thr * self.dt

        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                 0., float(self.n_refractory))
        new_s = tf.stack((new_v, new_b), axis=-1)

        def safe_grad(y, x):
            g = tf.gradients(y, x)[0]
            if g is None:
                g = tf.zeros_like(x)
            return g

        dnew_v_ds = tf.gradients(new_v, s, name='dnew_v_ds')[0]
        dnew_b_ds = tf.gradients(new_b, s, name='dnew_b_ds')[0]
        dnew_s_ds = tf.stack((dnew_v_ds, dnew_b_ds), 2, name='dnew_s_ds')

        dnew_z_dnew_v = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_v))
        dnew_z_dnew_b = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_b))
        dnew_z_dnew_s = tf.stack((dnew_z_dnew_v, dnew_z_dnew_b), axis=-1)

        diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]

        # "in_weights, rec_weights"
        # ds_dW_bias: 2 x n_rec
        dnew_v_di = safe_grad(new_v,i_t)
        dnew_b_di = safe_grad(new_b,i_t)
        dnew_s_di = tf.stack([dnew_v_di,dnew_b_di], axis=-1)

        partials_wrt_biases = [dnew_s_di, dnew_s_di]

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r)
        return [new_z, new_s, new_v, diagonal_jacobian, partials_wrt_biases], new_state


def tile_matrix_to_match(matrix, match_matrix):
    n_batch_dims = len(match_matrix.get_shape()) - len(matrix.get_shape())
    for _ in range(n_batch_dims):
        matrix = matrix[None, ...]

    tile_dims = tf.shape(match_matrix)
    tile_dims = tf.concat((tile_dims[:-2], [1, 1]), -1)
    matrix = tf.tile(matrix, tile_dims, name='tile_matrix_to_match')
    return matrix


def compute_fast_gradients(cell, inputs, outputs, learning_signals, initial_state):
    z, s, v, diag_j, partials_wrt_biases = outputs
    dnew_s_ds, dnew_z_dnew_s = diag_j

    jacobi_identity = tile_matrix_to_match(
        tf.eye(tf.shape(dnew_s_ds)[-1]), dnew_s_ds[:, 0, ...])[:, None]
    m = tf.transpose(tf.concat([dnew_s_ds, jacobi_identity], axis=1), (1, 0, 2, 3, 4))
    z_previous = tf.concat((initial_state.z[:, None, :], z[:, :-1, :]), 1)
    in_rec = [inputs, z_previous]

    def cum_backward(_previous, _inputs):
        _lp, _d, _partials = _inputs
        _new_lp = _lp + tf.einsum('bns,bnsk->bnk', _previous[0], _d)
        _de_db = [tf.einsum('bjs,bjs->bj', _new_lp , _p) for _p in _partials]
        return _new_lp, _de_db

    partials_for_scan = [tf.transpose(p, (1, 0, 2, 3)) for p in partials_wrt_biases]
    scan_inputs = [tf.transpose(learning_signals[..., None] * dnew_z_dnew_s, (1, 0, 2, 3)), m[1:], partials_for_scan]
    _, de_dbiases = tf.scan(
        cum_backward, scan_inputs, (tf.zeros_like(s[:, 0]),
                                    [tf.zeros_like(z[:, 0]), tf.zeros_like(z[:, 0])]), reverse=True)
    de_dbiases = [tf.transpose(p, (1, 0, 2)) for p in de_dbiases]

    prop_gradients = [tf.einsum('btj,bti->bij', d_b, inp) * mask
                      for d_b, inp, mask in zip(de_dbiases, in_rec, cell.dw_val_dw_var)]
    return prop_gradients


def get_signed_weights_reflect(signs, n_post, dtype = tf.float32):

    n_pre = len(signs)

    init_w_var = np.random.randn(n_pre, n_post) / np.sqrt(n_pre)
    init_w_var = tf.cast(init_w_var, dtype)
    w_val = tf.get_variable('w', initializer=init_w_var)

    # constrain e and i
    matrix_signs = tf.sign(w_val)
    pos_matrix = w_val * matrix_signs

    final_matrix = signs[:,None] * pos_matrix
    return final_matrix, w_val


def get_signed_weights_clipped(signs, n_post, dtype = tf.float32):
    n_pre = len(signs)

    init_w_var = np.random.randn(n_pre, n_post) / np.sqrt(n_pre)
    w_var = tf.cast(init_w_var, dtype)
    w_val = tf.get_variable('w', initializer=w_var)

    thresh = signs[:,None] * w_val
    thresh = tf.nn.relu(thresh)
    final_matrix = signs[:,None] * thresh

    return final_matrix, w_val


def spike_encode(input_component, minn, maxx, n_input_code=100, max_rate_hz=200, dt=1, n_dt_per_step=None):
    """
    Population-rate encode analog values

    :param input_component: tensor of analog values
    :param minn: minimum value that this population can encode
    :param maxx: maximum value that this population can encode
    :param n_input_code: number of neurons that encode this value
    :param max_rate_hz: maximum rate for tuned neurons
    :param dt:
    :param n_dt_per_step: number of time steps a single analog value is encoded
    :return: A spike tensor that encodes the analog values
    """
    if 110 < n_input_code < 210:  # 100
        factor = 20
    elif 90 < n_input_code < 110:  # 100
        factor = 10
    elif 15 < n_input_code < 25:  # 20
        factor = 4
    else:
        factor = 2

    sigma_tuning = (maxx - minn) / n_input_code * factor
    mean_line = tf.cast(tf.linspace(minn - 2. * sigma_tuning, maxx + 2. * sigma_tuning, n_input_code), tf.float32)
    max_rate = max_rate_hz / 1000
    max_prob = max_rate * dt

    step_neuron_firing_prob = max_prob * tf.exp(-(mean_line[None, None, :] - input_component[..., None]) ** 2 /
                                                (2 * sigma_tuning ** 2))

    if n_dt_per_step is not None:
        spike_code = tf.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample(n_dt_per_step)
        dims = len(spike_code.get_shape())
        r = list(range(dims))
        spike_code = tf.transpose(spike_code, r[1:-1] + [0, r[-1]])
    else:
        spike_code = tf.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample()

    spike_code = tf.cast(spike_code, tf.float32)
    return spike_code


def exp_convolve(tensor, decay):
    with tf.name_scope('exp_convolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor
