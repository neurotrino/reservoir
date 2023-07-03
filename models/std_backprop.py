"""TODO: module docs

Backprop
"""

# external ----
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import os

# internal ----
from models.common import *
import utils.config

# from models.neurons.adex import *
from models.neurons.lif import *
from utils.connmat import (
    ExInOutputMatrixGenerator as OMG,
)  # for generating output weights this time
from utils.config import subconfig
from utils.misc import SwitchedDecorator

DEBUG_MODE = True

switched_tf_function = SwitchedDecorator(tf.function)
switched_tf_function.enabled = not DEBUG_MODE


class ModifiedDense(tf.keras.layers.Layer):
    @tf.custom_gradient
    def matmul_random_feedback(self, filtered_z, W_out_arg, B_out_arg):

        logits = tf.einsum("btj,jk->btk", filtered_z, W_out_arg)

        def grad(dy):
            dloss_dW_out = tf.einsum("bij,bik->jk", filtered_z, dy)
            dloss_dfiltered_z = tf.einsum("bik,jk->bij", dy, B_out_arg)
            dloss_db_out = tf.zeros_like(B_out_arg)

            return [dloss_dfiltered_z, dloss_dW_out, dloss_db_out]

        return logits, grad

    def __init__(
        self,
        cfg,
        num_neurons,
        inhib_multiplier,
        p_eo,
        p_io,
        frac_e,  # architecture
        mu,
        sigma,  # weight distribution
        *args,
        **kwargs,  # parameters for keras.layers.Dense
    ):
        super(ModifiedDense, self).__init__(*args, **kwargs)

        self.cfg = cfg
        self.num_neurons = num_neurons  # [!] shouldn't need; Dense can
        #     get d1, we should too
        if self.cfg["model"].cell.categorical_output:
            self.n_out = 2
        elif self.cfg["model"].cell.likelihood_output:
            self.n_out = 2
        else:
            self.n_out = 1
        self.inhib_multiplier = inhib_multiplier
        self.p_eo = p_eo
        self.p_io = p_io
        self.frac_e = frac_e
        self.mu = mu
        self.sigma = sigma

        self.ecount = round(self.frac_e * self.num_neurons)
        self.icount = self.num_neurons - self.ecount

        # Number of zeros in W_out to try to maintain if output
        # rewiring is enabled
        #
        # [!] don't know why this was implemented per-layer; see #55
        #     for a unified implementation
        self.output_target_zcount = None
        self.oweights = None

    def build(self, input_shape):

        if self.oweights is None:

            self.oweights = self.add_weight(
                name="output",
                initializer=tf.keras.initializers.Zeros(),
                shape=(self.num_neurons, self.n_out),
                trainable=self.trainable,
            )
            output_connmat_generator = OMG(
                n_excit=self.ecount,
                n_inhib=self.icount,
                n_out=self.n_out,
                inhib_multiplier=self.inhib_multiplier,
                p_from_e=self.p_eo,
                p_from_i=self.p_io,
                mu=self.mu,
                sigma=self.sigma,
            )
            initial_oweights = output_connmat_generator.run_generator()
            self.oweights.assign_add(
                initial_oweights * self.cfg["model"].cell.output_multiplier
            )

            # save initial output weights
            np.save(
                os.path.join(
                    self.cfg["save"].main_output_dir, "output_preweights.npy"
                ),
                initial_oweights,
            )

            # set output weights from generators
            #
            # [!] don't know why this was implemented per-layer; see #55
            #     for a unified implementation
            self.output_sign = tf.sign(self.oweights)

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return tf.einsum("btj,jk->btk", inputs, self.oweights)

    @tf.function
    def rewire(self, input_id=None):
        if self.output_target_zcount is None:
            # When no target number of zeros is recorded, count how
            # many zeros there currently are, save that for future
            # comparison, and return
            self.output_target_zcount = len(tf.where(self.oweights == 0))
            logging.debug(
                f"output matrix will maintain "
                + f"{self.output_target_zcount} zeros"
            )
            return

        zero_indices = tf.where(self.oweights == 0)
        num_new_zeros = tf.shape(zero_indices)[0] - self.output_target_zcount

        # Replace any new zeros (not necessarily in the same spot)
        if num_new_zeros > 0:
            logging.debug(
                f"found {num_new_zeros} new zeros for a total of "
                + f"{len(zero_indices)} zeros"
            )

            # Generate a list of non-zero replacement weights
            new_weights = np.random.lognormal(
                self.mu, self.sigma, num_new_zeros
            )
            new_weights[np.where(new_weights == 0)] += 0.01

            # Randomly select zero-weight indices (without replacement)
            # if we want no in-to-out direct path, do not permit selection from
            # the input_ids that happen to also be zero
            # [?] use tf instead of np
            meta_indices = np.random.choice(
                len(zero_indices), num_new_zeros, False
            )
            if cfg["model"].cell.no_input_to_output:
                # do not use any of the input ids that happen to be zero originators for the output
                noninput_zero_indices = np.setdiff1d(zero_indices, input_id)
                # find indices that ARE zero indices and also NOT receiving input
                # and thereby valid choices for rewiring the output layer
                # (in this way the output and input layers become more separated
                # over the course of training, even tho they start out with no
                # constraints on overlap; revisit that later)
                if len(noninput_zero_indices) < num_new_zeros:
                    # figure out how many more indices are required
                    compromise_count = num_new_zeros - len(noninput_zero_indices)
                    noninput_zero_indices = np.setdiff1d(zero_indices, np.random.shuffle(input_id)[:len(input_id)-compromise_count])
                zero_indices = tf.gather(noninput_zero_indices, meta_indices)
            else:
                zero_indices = tf.gather(zero_indices, meta_indices)

            # Invert and scale inhibitory neurons
            # [!] Eventually prefer to use .in_mask instead of a loop
            for i in range(len(zero_indices)):
                if zero_indices[i][0] >= self.ecount:
                    new_weights[i] *= self.inhib_multiplier

            # Update recurrent weights
            x = tf.tensor_scatter_nd_update(
                tf.zeros(self.oweights.shape), zero_indices, new_weights
            )
            logging.debug(
                f"{tf.math.count_nonzero(x)} non-zero values generated in "
                + "output weight patch"
            )

            # as of version 2.6.0, tensorflow does not support in-place
            # operation of tf.tensor_scatter_nd_update(), so we just
            # add it to our recurrent weights, which works because
            # scatter_nd_update, only has values in places where
            # recurrent weights are zero
            self.oweights.assign_add(x)
            logging.debug(
                f"{tf.math.count_nonzero(self.oweights)} non-zeroes in "
                + " output layer after adjustments"
            )

        # update output_sign
        self.output_sign = tf.sign(self.oweights)

    def get_config(self):
        # Return a JSON-serializable configuration of this object.
        # The output of this method is the input to `.from_config()`.
        parent_config = super(ModifiedDense, self).get_config()
        self_config = {
            "num_neurons": self.num_neurons,
            "p_eo": self.p_eo,
            "p_io": self.p_io,
            "frac_e": self.frac_e,
            "mu": self.mu,
            "sigma": self.sigma,
        }
        return {**parent_config, **self_config}  # '|' op in Python 3.9


class Model(BaseModel):
    """Generic prototyping model designed to test new features and
    provide an example to people learning the research infrastructure.
    """

    def __init__(self, cfg, training=True):
        """..."""
        super().__init__()
        self.loaded_weights = None

        self.training = training

        # Attribute assignments
        self.cfg = cfg
        cell_cfg = self.cfg["model"].cell
        train_cfg = self.cfg["train"]

        cell_type = eval(cfg["model"].cell.type)  # neuron (class)
        self.cell = cell_type(
            subconfig(  # neuron (object)
                cfg, cfg["model"].cell, old_label="model", new_label="cell"
            )
        )
        logging.info(f"cell type set to {cell_type.__name__}")

        if (
            self.cfg["model"].cell.likelihood_output
            or self.cfg["model"].cell.categorical_output
        ):
            self.n_out = 2
        else:
            self.n_out = 1

        # Layer definitions
        self.rnn1 = tf.keras.layers.RNN(self.cell, return_sequences=True)

        if self.cfg["model"].cell.define_output_w:
            self.dense1 = ModifiedDense(
                self.cfg,
                cell_cfg.units,
                cell_cfg.inhib_multiplier,
                cell_cfg.p_eo,
                cell_cfg.p_io,
                cell_cfg.frac_e,
                cell_cfg.mu,
                cell_cfg.sigma,
                trainable=train_cfg.output_trainable,
                dtype=tf.float32,
            )
            self.dense1.trainable = self.training
        else:
            self.dense1 = tf.keras.layers.Dense(self.n_out)
            self.dense1.trainable = (
                self.training and self.cfg["train"].output_trainable
            )

        self.layers = [  # gather in a list for later convenience
            self.rnn1,
            self.dense1,
        ]


    @switched_tf_function
    def noise_weights(self, mean=1.0, stddev=0.1):
        """Add noise to the recurrent weights."""
        weights = self.rnn1.get_weights()

        iweights = weights[0]
        rweights = weights[1]

        gain_matrix = tf.clip_by_value(
            tf.random.normal(rweights.shape, mean, stddev), -1, 1
        )
        noised_weights = rweights * gain_matrix

        self.rnn1.set_weights([iweights, noised_weights])


    @switched_tf_function
    def call(self, inputs, training=False):
        """..."""

        # [!] is it okay that I got rid of tf.identity for the outputs?
        # [!] is it a problem that I'm putting cell.initial_state here?
        voltages, spikes = self.rnn1(
            inputs,
            initial_state=self.cell.zero_state(self.cfg["train"].batch_size),
            training=self.training
        )
        prediction = self.dense1(spikes)
        prediction = exp_convolve(prediction, axis=1)

        try:
            if not self.training and self.loaded_weights is not None:
                [w0, w1, w2] = self.loaded_weights

                self.cell.input_weights.assign(w0)
                self.cell.recurrent_weights.assign(w1)

                dense_weights = self.dense1.get_weights()
                self.dense1.set_weights([w2])
        except Exception as e:
            logging.warning(f"issue loading saved weights: {e}")

        return voltages, spikes, prediction


    @classmethod
    def from_disk(cls, npz_filename, cfg_filename, train=False):
        """Load a trained model."""
        data = np.load(npz_filename)
        cfg = utils.config.load_hjson_config(cfg_filename)

        w0 = data["tv0.postweights"][-1]  # input connections
        w1 = data["tv1.postweights"][-1]  # recurrent connections
        w2 = data["tv2.postweights"][-1]  # output connections

        model = cls(cfg)
        model.loaded_weights = [w0, w1, w2] # hacky temp workaround
        model.training = train

        return model
