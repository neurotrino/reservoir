import numpy as np
import tensorflow as tf
import ConnMatGenerator as connmat
import EIConnMatGenerator as EIconnmat


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    """
    originally,
    :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
    so we must make sure our membrane dynamics (with negative real valued thresholds, etc.) is consistent with this voltage-scaling spike generation mechanic
    in this case, we are normalizing using -(thr-V)/(thr-EL), which is a variation on the way one would normalize x between 0 and 1 using (x-min)/(max-min)
    (it would be a case of -(max-x)/(max-min)
    :param dampening_factor: parameter to stabilize learning
    """
    z_ = tf.greater(v_scaled, 0.) # returns bool of whether v_scaled is above thr or not, since it would be equal to 0 at thr
    z_ = tf.cast(z_, tf.float32) # cast as number [0, 1]

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


class LogNormal(tf.keras.initializers.Initializer):
    def __init__(self, mean, stddev, units, p):
        self.mean = mean
        self.stddev = stddev
        self.units = units
        self.p = p

    def __call__(self, shape, dtype=None):
        connmat_generator = connmat.ConnectivityMatrixGenerator(self.units, self.p, self.mean, self.stddev)
        initial_weights_mat = connmat_generator.run_generator()
        return tf.convert_to_tensor(tf.cast(initial_weights_mat, tf.float32))
        #normdist = tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
        #return tf.math.exp(normdist)


class LIFCell(tf.keras.layers.Layer):
    def __init__(self, units, thr, EL, tau, dt, n_refractory, dampening_factor, p, mu, sigma):
        super().__init__()
        self.units = units

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._n_refractory = n_refractory

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        self.threshold = thr
        self.EL = EL
        self._dampening_factor = dampening_factor
        self.p = p
        self.mu = mu
        self.sigma = sigma

        #                  voltage, refractory, previous spikes
        self.state_size = (units, units, units)

    def zero_state(self, batch_size, dtype=tf.float32):
        # voltage
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL
        # refractory
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # spike
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return v0, r0, z_buf0

    def build(self, input_shape):
        # using uniform weight dist for inputs as opposed to RandomNormal(mean=1., stddev=1. / np.sqrt(input_shape[-1] + self.units))
        #self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             #initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)), name='input_weights')
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=0.4), name='input_weights')

        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool) # disconnect self-recurrent weights

        # self.recurrent_weights = self.add_weight(shape=(self.units, self.units), initializer = LogNormal(self.mu, self.sigma, self.units, self.p), trainable=True, name='recurrent_weights')

        self.recurrent_weights = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(gain=.7),
            initializer = tf.keras.initializers.RandomNormal();
            trainable = True,
            name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses

        # weights are lognormal, see ConnMatGenerator.py > def make_weighted(self)
        connmat_generator = connmat.ConnectivityMatrixGenerator(self.units, self.p, self.mu, self.sigma)
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # not currently using bias currents
        #self.bias_currents = self.add_weight(shape=(self.units,),
                                             #initializer=tf.keras.initializers.Zeros(),
                                             #name='bias_currents')
        super().build(input_shape)

    def call(self, inputs, state):
        old_v = state[0]
        old_r = state[1]
        old_z = state[2]

        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, no_autapse_w_rec)
        # to circumvent the problem of voltage reset, we have a subtractive current applied if a spike occurred in previous time step
        # i_reset = -self.threshold * old_z # in the toy-valued case, we can just subtract threshold which was 1, to return to baseline 0, or approximately baseline
        # now to have the analogous behavior using real voltage values, we must subtract the difference between thr and EL
        i_reset = -(self.threshold-self.EL) * old_z # approx driving the voltage 20 mV more negative
        input_current = i_in + i_rec + i_reset # + self.bias_currents[None]

        # previously, whether old_v was below or above 0, you would still decay gradually back to 0
        # decay was dependent on the distance between your old voltage and resting 0
        # the equation was simply new_v = self._decay * old_v + input_current
        # we are now writing it with the same concept: the decay is dependent on the distance between old voltage and rest at -70mV
        # that decay is then added to the resting value
        # in the same way that decay was previously implicitly added to 0 (rest)
        # this ensures the same basic behavior s.t. if you're above EL, you hyperpolarize to EL
        # and if you are below EL, you depolarize to EL
        new_v = self.EL + (self._decay) * (old_v - self.EL) + input_current

        is_refractory = tf.greater(old_r, 0)
        #v_scaled = (new_v - self.threshold) / self.threshold
        v_scaled = -(self.threshold-new_v) / (self.threshold-self.EL)
        new_z = spike_function(v_scaled, self._dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(
            old_r - 1 + tf.cast(new_z * self._n_refractory, tf.int32),
            0,
            self._n_refractory)

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state


class SpikeRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, target_rate, rate_cost): # rate in spikes/ms for ease
        self._rate_cost = rate_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.threshold

        rate = tf.reduce_mean(spike, axis=(0, 1))
        # av = Second * tf.reduce_mean(z, axis=(0, 1)) / flags.dt
        #regularization_coeff = tf.Variable(np.ones(flags.n_neurons) * flags.reg_fr, dtype=tf.float32, trainable=False)
        #loss_reg_fr = tf.reduce_sum(tf.square(rate - flags.target_rate) * regularization_coeff)
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        return inputs

class SpikeVoltageRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, rate_cost=.1, voltage_cost=.01, target_rate=.02): # rate in spikes/ms for ease
        self._rate_cost = rate_cost
        self._voltage_cost = voltage_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.threshold

        rate = tf.reduce_mean(spike, axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        v_pos = tf.square(tf.clip_by_value(tf.nn.relu(voltage - upper_threshold), 0., 1.))
        v_neg = tf.square(tf.clip_by_value(tf.nn.relu(-voltage - self._cell.threshold), 0., 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        self.add_loss(voltage_loss)
        self.add_metric(voltage_loss, name='voltage_loss', aggregation='mean')
        return inputs

class Adex(tf.keras.layers.Layer):
    def __init__(self, n_neurons, n_in, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p):

        if tauw is None: raise ValueError("Time constant for adaptive bias must be set.")
        if a is None: raise ValueError("a parameter for adaptive bias must be set.")

        super().__init__()

        self._dt = float(dt)

        self.units = n_neurons
        self.n_in = n_in
        self.threshold = thr
        self.n_refrac = n_refrac
        self.dampening_factor = dampening_factor
        self.tauw = tauw
        self.a = a
        self.b = b
        self.gL = gL
        self.EL = EL
        self.C = C
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.p = p
        self.dt_gL__C = self._dt * self.gL / self.C
        self.dt_a__tauw = self._dt * self.a / self.tauw

        self.input_weights = None
        # self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        #                  voltage,    refractory, adaptation, spikes (spiking or not)
        self.state_size = (self.units, self.units, self.units, self.units)
        #                  voltage,     spikes
        self.output_size = (self.units, self.units)

    def zero_state(self, batch_size, dtype=tf.float32):
        # Voltage (all at EL)
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL  # Do we want to start with random V?
        # Refractory (all 0)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # Adaptation (all 0)
        w0 = tf.zeros((batch_size, self.units), tf.float32)
        # Spike (all not spiking)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return [v0, r0, w0, z_buf0]

    def build(self, input_shape):

        # Create the input weights which should be of the form:
        #   np.array([[input1toN1, ..., input1toNn], ..., [inputktoN1, ..., inputktoNn]], dtype=np.float32)
        # Not sure why this choice of distribution; included also uniform used in LIFCell model
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=0.5),
                                             name='input_weights')

        # Create the recurrent weights, their value here is not important
        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses

        connmat_generator = connmat.ConnectivityMatrixGenerator(self.units, self.p)
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # To make sure that all self-connections are 0 after call
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)

        # Bias_currents; commented out because we are not using it and it might affect the way I am assigning the weights
        # self.bias_currents = self.add_weight(shape=(self.units,),
        #                                      initializer=tf.keras.initializers.Zeros(),
        #                                      name='bias_currents')

        super().build(input_shape)

    def call(self, inputs, state):

        # Old states
        old_v = state[0]
        old_r = state[1]
        old_w = state[2]
        old_z = state[3]

        # No self-connections (diagonal in disconnect_mask is all True so diagonal in recurrent_weights will be like the diagonal in zeros)
        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        # Calculate input current
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, no_autapse_w_rec)
        # There is no reset current because we are setting new_V to V_reset if old_z > 0.5
        i_t = i_in + i_rec  # + self.bias_currents[None]

        # Update voltage
        exp_terms = tf.clip_by_value(tf.exp((old_v - self.threshold)/self.deltaT), -1e6, 30 / self.dt_gL__C)  # These min and max values were taken from tf1
        exp_terms = tf.stop_gradient(exp_terms)  # I think we need this but I am not 100% sure
        new_v = old_v - (self.dt_gL__C * (old_v - self.EL)) + (self.dt_gL__C * self.deltaT * exp_terms) + ((i_t - old_w) * self._dt / self.C)
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)

        # Update adaptation term
        new_w = old_w - ((self._dt / self.tauw) * old_w) + (self.dt_a__tauw * (old_v - self.EL))
        new_w += self.b * old_z

        # Determine if the neuron is spiking
        is_refractory = tf.greater(old_r, 0)
        # v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.threshold-new_v) / (self.threshold-self.EL)
        new_z = spike_function(v_scaled, self.dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Determine if the neuron is in a refractory period (and remaining time if yes)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self.n_refrac, tf.int32), 0, self.n_refrac)

        # New state and output
        new_state = (new_v, new_r, new_w, new_z)
        output = (new_v, new_z)

        return output, new_state

class Adex_EI(tf.keras.layers.Layer):

    def __init__(self, n_neurons, frac_e, n_in, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p_ee, p_ei, p_ie, p_ii):

        if tauw is None: raise ValueError("Time constant for adaptive bias must be set.")
        if a is None: raise ValueError("a parameter for adaptive bias must be set.")
        if (frac_e * n_neurons) % 1 != 0: raise ValueError("The resulting number of excitatory neurons should be an integer.")

        super().__init__()

        self._dt = float(dt)

        self.units = n_neurons
        self.n_excite = frac_e * self.units
        self.n_inhib = self.units - self.n_excite
        self.n_in = n_in
        self.threshold = thr
        self.n_refrac = n_refrac
        self.dampening_factor = dampening_factor
        self.tauw = tauw
        self.a = a
        self.b = b
        self.gL = gL
        self.EL = EL
        self.C = C
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.p_ee = p_ee
        self.p_ei = p_ei
        self.p_ie = p_ie
        self.p_ii = p_ii
        self.dt_gL__C = self._dt * self.gL / self.C
        self.dt_a__tauw = self._dt * self.a / self.tauw

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        #                  voltage,    refractory, adaptation, spikes (spiking or not)
        self.state_size = (self.units, self.units, self.units, self.units)
        #                   voltage,    spikes
        self.output_size = (self.units, self.units)

    def zero_state(self, batch_size, dtype=tf.float32):
        # Voltage (all at EL)
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL  # Do we want to start with random V?
        # Refractory (all 0)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # Adaptation (all 0)
        w0 = tf.zeros((batch_size, self.units), tf.float32)
        # Spike (all not spiking)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return [v0, r0, w0, z_buf0]

    def build(self, input_shape):

        # Create the input weights which should be of the form:
        #   np.array([[input1toN1, ..., input1toNn], ..., [inputktoN1, ..., inputktoNn]], dtype=np.float32)
        # Not sure why this choice of distribution; included also uniform used in LIFCell model
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1),
                                             name='input_weights')
        # Create the recurrent weights, their value here is not important
        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses
        EIconnmat_generator = EIconnmat.ConnectivityMatrixGenerator(self.n_excite, self.n_inhib, self.p_ee, self.p_ei, self.p_ie, self.p_ii)
        initial_weights_mat = EIconnmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # To make sure that all self-connections are 0 after call
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)

        # Store the initial signs for later
        self.rec_sign = tf.sign(self.recurrent_weights)

        # Bias_currents; commented out because we are not using it and it might affect the way I am assigning the weights
        # self.bias_currents = self.add_weight(shape=(self.units,),
        #                                      initializer=tf.keras.initializers.Zeros(),
        #                                      name='bias_currents')

        super().build(input_shape)

    def call(self, inputs, state):

        # Old states
        old_v = state[0]
        old_r = state[1]
        old_w = state[2]
        old_z = state[3]

        # No self-connections (diagonal in disconnect_mask is all True so diagonal in recurrent_weights will be like the diagonal in zeros)
        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        # If the sign of a weight changed, make the weight 0
        constrained_w_rec = tf.where(self.rec_sign * no_autapse_w_rec >= 0, no_autapse_w_rec, 0)

        # Calculate input current
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, constrained_w_rec)
        # There is no reset current because we are setting new_V to V_reset if old_z > 0.5
        i_t = i_in + i_rec  # + self.bias_currents[None]

        # Update voltage
        exp_terms = tf.clip_by_value(tf.exp((old_v - self.threshold)/self.deltaT), -1e6, 30 / self.dt_gL__C)  # These min and max values were taken from tf1
        exp_terms = tf.stop_gradient(exp_terms)  # I think we need this but I am not 100% sure
        new_v = old_v - (self.dt_gL__C * (old_v - self.EL)) + (self.dt_gL__C * self.deltaT * exp_terms) + ((i_t - old_w) * self._dt / self.C)
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)

        # Update adaptation term
        new_w = old_w - ((self._dt / self.tauw) * old_w) + (self.dt_a__tauw * (old_v - self.EL))
        new_w += self.b * old_z

        # Determine if the neuron is spiking
        is_refractory = tf.greater(old_r, 0)
        # v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.threshold-new_v) / (self.threshold-self.EL)
        new_z = spike_function(v_scaled, self.dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Determine if the neuron is in a refractory period (and remaining time if yes)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self.n_refrac, tf.int32), 0, self.n_refrac)

        # New state and output
        new_state = (new_v, new_r, new_w, new_z)
        output = (new_v, new_z)

        return output, new_state

class AdexCS(tf.keras.layers.Layer):
    def __init__(self, n_neurons, n_in, thr, n_refrac, dt, dampening_factor, tauw, a, b, gL, EL, C, deltaT, V_reset, p, tauS, VS):

        if tauw is None: raise ValueError("Time constant for adaptive bias must be set.")
        if a is None: raise ValueError("a parameter for adaptive bias must be set.")

        super().__init__()

        self._dt = float(dt)

        self.units = n_neurons
        self.n_in = n_in
        self.threshold = thr
        self.n_refrac = n_refrac
        self.dampening_factor = dampening_factor
        self.tauw = tauw
        self.a = a
        self.b = b
        self.gL = gL
        self.EL = EL
        self.C = C
        self.deltaT = deltaT
        self.V_reset = V_reset
        self.p = p
        self.tauS = tauS
        self.VS = VS
        self.dt_gL__C = self._dt * self.gL / self.C
        self.dt_a__tauw = self._dt * self.a / self.tauw

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        #                  voltage,    refractory, adaptation, synaptic conductance, spikes (spiking or not)
        self.state_size = (self.units, self.units, self.units, self.units, self.units)
        #                  voltage,     spikes
        self.output_size = (self.units, self.units)

    def zero_state(self, batch_size, dtype=tf.float32):
        # Voltage (all at EL)
        v0 = tf.zeros((batch_size, self.units), dtype) + self.EL  # Do we want to start with random V?
        # Refractory (all 0)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        # Adaptation (all 0)
        w0 = tf.zeros((batch_size, self.units), tf.float32)
        # Conductance (all 0)
        g0 = tf.zeros((batch_size, self.units), tf.float32)
        # Spike (all not spiking)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return [v0, r0, w0, g0, z_buf0]

    def build(self, input_shape):

        # Create the input weights which should be of the form:
        #   np.array([[input1toN1, ..., input1toNn], ..., [inputktoN1, ..., inputktoNn]], dtype=np.float32)
        # Not sure why this choice of distribution; included also uniform used in LIFCell model
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        '''
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                             name='input_weights')

        # Create the recurrent weights, their value here is not important
        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 name='recurrent_weights')

        # Set the desired values for recurrent weights while accounting for p
        # Input weights remain the same
        # initial_weights_mat should be of the same form self.recurrent_weight.value(), i.e.:
        #   np.array([[N1toN1, ..., N1toNn], ..., [NntoN1, ..., NntoNn]], dtype=np.float32)
        # !!!!! might need to change how we set the weights because current based synapses
        connmat_generator = connmat.ConnectivityMatrixGenerator(self.units, self.p)
        initial_weights_mat = connmat_generator.run_generator()
        self.set_weights([self.input_weights.value(), initial_weights_mat])

        # To make sure that all self-connections are 0 after call
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)

        # Bias_currents; commented out because we are not using it and it might affect the way I am assigning the weights
        # self.bias_currents = self.add_weight(shape=(self.units,),
        #                                      initializer=tf.keras.initializers.Zeros(),
        #                                      name='bias_currents')

        super().build(input_shape)

    def call(self, inputs, state):

        # Old states
        old_v = state[0]
        old_r = state[1]
        old_w = state[2]
        old_g = state[3]
        old_z = state[4]

        # No self-connections (diagonal in disconnect_mask is all True so diagonal in recurrent_weights will be like the diagonal in zeros)
        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        # Calculate input current
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, no_autapse_w_rec)
        # There is no reset current because we are setting new_V to V_reset if old_z > 0.5
        # i_t = i_in + i_rec  # + self.bias_currents[None]
        i_w = tf.reduce_sum(i_rec)
        gS = old_g + i_w  # Need to change this but that's the idea

        # Update voltage
        exp_terms = tf.clip_by_value(tf.exp((old_v - self.threshold)/self.deltaT), -1e6, 30 / self.dt_gL__C)  # These min and max values were taken from tf1
        exp_terms = tf.stop_gradient(exp_terms)  # I think we need this but I am not 100% sure
        new_v = old_v - (self.dt_gL__C * (old_v - self.EL)) + (self.dt_gL__C * self.deltaT * exp_terms) + ((i_in + (gS * (self.VS - old_v)) - old_w) * self._dt / self.C)
        new_v = tf.where(old_z > .5, tf.ones_like(new_v) * self.V_reset, new_v)

        # Update adaptation term
        new_w = old_w - ((self._dt / self.tauw) * old_w) + (self.dt_a__tauw * (old_v - self.EL))
        new_w += self.b * old_z

        # Update synaptic conductance
        new_g = gS - ((self._dt / self.tauS) * gS)

        # Determine if the neuron is spiking
        is_refractory = tf.greater(old_r, 0)
        # v_scaled = (new_v - self.thr) / self.thr
        v_scaled = -(self.threshold-new_v) / (self.threshold-self.EL)
        new_z = spike_function(v_scaled, self.dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Determine if the neuron is in a refractory period (and remaining time if yes)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self.n_refrac, tf.int32), 0, self.n_refrac)

        # New state and output
        new_state = (new_v, new_r, new_w, new_g, new_z)
        output = (new_v, new_z)

        return output, new_state
