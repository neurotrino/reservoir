"""Connectivity matrices."""

# version: Python 3.7
__author__ = "Tarek, Isabel"

from os import makedirs, path

import csv
import logging
import numpy as np
import tensorflow as tf

# ┬───────────────────────────────────────────────────────────────────────────╮
# ┤ Output Connectivity Matrix Generator                                             │
# ┴───────────────────────────────────────────────────────────────────────────╯


class ExInOutputMatrixGenerator(object):
    def __init__(
        self, n_excit, n_inhib, n_out, inhib_multiplier, p_from_e, p_from_i, mu, sigma
    ):

        self.n_excit = n_excit
        self.n_inhib = n_inhib
        self.inhib_multiplier = inhib_multiplier
        self.n_in = n_excit + n_inhib
        self.n_out = n_out
        self.p_from_e = p_from_e
        self.p_from_i = p_from_i

        self.output_mat = np.zeros((self.n_in, self.n_out))

        self.output_weight = np.zeros((self.n_in, self.n_out))
        self.mu = mu
        self.sigma = sigma

        # total count of output connections from inhib units
        self.k_ie = int(round(p_from_i * self.n_inhib))
        # total count of output connections from excit units
        self.k_ee = int(round(p_from_e * self.n_excit))

    def run_generator(self):

        # Generate connectivity matrix and check it's successful
        try:
            if not self.generate_conn_mat():
                raise Exception("failed to generate connectivity matrix")
            logging.info("generated E/I connectivity matrix")

            if not self.make_weighted():
                raise Exception("failed to weight output connectivity matrix")
            logging.info("output connectivity matrix weighted")

            return self.output_weight * 0.1

        except Exception as e:
            logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:

            # E to output connections
            for n in range(0, self.n_out):
                for a in range(0, self.k_ee):
                    rand = np.random.randint(0, self.n_excit)
                    while self.output_mat[rand][n] == 1:
                        rand = np.random.randint(0, self.n_excit)
                    self.output_mat[rand][n] = 1

            # I to output connections
            for n in range(0, self.n_out):
                for a in range(0, self.k_ie):
                    rand = np.random.randint(0, self.n_inhib)
                    while self.output_mat[rand + self.n_excit][n] == 1:
                        rand = np.random.randint(0, self.n_inhib)
                    self.output_mat[rand + self.n_excit][n] = 1

            return True

        except Exception as e:
            logging.exception(e)
            return False

    def make_weighted(self):

        try:
            # Generate random weights and fill connectivity matrix
            for i in range(0, self.n_in):
                for j in range(0, self.n_out):
                    if self.output_mat[i][j] == 1:
                        self.output_weight[i][j] = np.random.lognormal(
                            self.mu, self.sigma
                        )
                        if self.n_in > i > (self.n_in - self.n_inhib):
                            self.output_weight[i][j] *= self.inhib_multiplier
            return True

        except Exception as e:
            logging.exception(e)
            return False


# ┬───────────────────────────────────────────────────────────────────────────╮
# ┤ Input Connectivity Matrix Generator                                             │
# ┴───────────────────────────────────────────────────────────────────────────╯


class InputMatrixGenerator(object):
    def __init__(self, n_in, n_neurons, p_input, mu, sigma, input_multiplier):

        self.n_neurons = n_neurons
        self.n_in = n_in
        self.n_rec = n_neurons

        # initialize connectivity matrix
        self.input_mat = np.zeros((self.n_in, self.n_rec))

        # initialize weight matrix
        self.input_weight = np.zeros((self.n_in, self.n_rec))
        self.mu = mu
        self.sigma = sigma
        self.input_multiplier = input_multiplier

        # calculate total number of connections per input unit
        self.k = int(round(p_input * self.n_rec))

    def run_generator(self):
        try:
            # Generate input matrix and check it's successful
            if not self.generate_conn_mat():
                raise Exception(
                    "failed to generate input connectivity matrix"
                )
            logging.info("input connectivity matrix generated")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception("failed to weight input connectivity matrix")
            logging.info("input connectivity matrix weighted")

            return self.input_weight

        except Exception as e:
            logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:
            for n in range(0, self.n_in):
                for a in range(0, self.k):
                    rand = np.random.randint(0, self.n_rec)
                    while self.input_mat[n][rand] == 1:
                        rand = np.random.randint(0, self.n_rec)
                    self.input_mat[n][rand] = 1

            return True

        except Exception as e:
            logging.exception(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_in):
                for j in range(0, self.n_rec):
                    if self.input_mat[i][j] == 1:
                        self.input_weight[i][j] = np.random.lognormal(
                            self.mu, self.sigma
                        )
            self.input_weight *= self.input_multiplier / 10

            return True

        except Exception as e:
            logging.exception(e)
            return False


# ┬───────────────────────────────────────────────────────────────────────────╮
# ┤ Connectivity Matrix Generator                                             │
# ┴───────────────────────────────────────────────────────────────────────────╯


class ConnectivityMatrixGenerator(object):
    def __init__(self, n_neurons, p, mu, sigma):

        self.n_neurons = n_neurons

        # Initialize connectivity matrix
        self.conn_mat = np.zeros((self.n_neurons, self.n_neurons))

        # Initialize weight matrix
        self.weight_mat = np.zeros((self.n_neurons, self.n_neurons))

        self.mu = mu
        self.sigma = sigma

        # Calculate total number of connections per neuron (remove
        # neuron from target if included (ee and ii))
        self.k = int(round(p * (self.n_neurons - 1)))

    def run_generator(self):
        try:
            # Generate connectivity matrix and check it's successful
            if not self.generate_conn_mat():
                raise Exception("failed to generate connectivity matrix")
            logging.info("connectivity matrix generated")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception("failed to weight connectivity matrix")
            logging.info("connectivity matrix weighted")

            return self.weight_mat / 10  # hard-coded decrease weights by /10;
            # later on we will need to change the mu and sigma to
            # reflect current rather than conductance anyway

        except Exception as e:
            logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:
            for n in range(0, self.n_neurons):
                for a in range(0, self.k):
                    rand = np.random.randint(0, self.n_neurons)
                    while rand == n or self.conn_mat[n][rand] == 1:
                        rand = np.random.randint(0, self.n_neurons)
                    self.conn_mat[n][rand] = 1

            return True

        except Exception as e:
            logging.exception(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn_mat[i][j] == 1:
                        self.weight_mat[i][j] = np.random.lognormal(
                            self.mu, self.sigma
                        )

            return True

        except Exception as e:
            logging.exception(e)
            return False


# ┬───────────────────────────────────────────────────────────────────────────╮
# ┤ Excitatory/Inhibitory Connectivity Matrix Generator                       │
# ┴───────────────────────────────────────────────────────────────────────────╯


class ExInConnectivityMatrixGenerator(object):
    def __init__(self, n_excite, n_inhib, inhib_multiplier, p_ee, p_ei, p_ie, p_ii, mu, sigma):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_neurons = n_excite + n_inhib

        self.inhib_multiplier = inhib_multiplier

        self.mu = mu
        self.sigma = sigma

        # Initialize connectivity matrix
        self.conn_mat = np.zeros((self.n_neurons, self.n_neurons))

        # Initialize weight matrix
        self.weight_mat = np.zeros((self.n_neurons, self.n_neurons))

        # Calculate total number of connections per neuron (remove
        # neuron from target if included (ee and ii))
        self.k_ii = int(round(p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(p_ei * self.n_inhib))
        self.k_ie = int(round(p_ie * self.n_excite))
        self.k_ee = int(round(p_ee * (self.n_excite - 1)))

    def run_generator(self):

        try:

            # Generate connectivity matrix and check it's successful
            if not self.generate_conn_mat():
                raise Exception("failed to generate connectivity matrix")
            logging.info("generated E/I connectivity matrix")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception("failed to weight connectivity matrix")
            logging.info("weighted connectivity matrix")

            return (
                self.weight_mat / 10
            )  # again doing the hard-coded divide by 10 to make weights in the range that seems most trainable

        except Exception as e:
            logging.exception(e)
            return False

    def generate_conn_mat(self):

        try:

            # E to E connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ee):
                    rand = np.random.randint(0, self.n_excite)
                    while rand == n or self.conn_mat[n][rand] == 1:
                        rand = np.random.randint(0, self.n_excite)
                    self.conn_mat[n][rand] = 1

            # E to I connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ei):
                    rand = np.random.randint(
                        self.n_excite, self.n_excite + self.n_inhib
                    )
                    while self.conn_mat[n][rand] == 1:
                        rand = np.random.randint(
                            self.n_excite, self.n_excite + self.n_inhib
                        )
                    self.conn_mat[n][rand] = 1

            # I to E connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ie):
                    rand = np.random.randint(0, self.n_excite)
                    while self.conn_mat[n + self.n_excite][rand] == 1:
                        rand = np.random.randint(0, self.n_excite)
                    self.conn_mat[n + self.n_excite][rand] = 1

            # I to I connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ii):
                    rand = np.random.randint(
                        self.n_excite, self.n_excite + self.n_inhib
                    )
                    while (
                        rand == (n + self.n_excite)
                        or self.conn_mat[n + self.n_excite][rand] == 1
                    ):
                        rand = np.random.randint(
                            self.n_excite, self.n_excite + self.n_inhib
                        )
                    self.conn_mat[n + self.n_excite][rand] = 1

            return True

        except Exception as e:
            logging.exception(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn_mat[i][j] == 1:
                        self.weight_mat[i][j] = np.random.lognormal(
                            self.mu, self.sigma
                        )
                        # Make all I 10 times stronger AND NEGATIVE
                        if (
                            self.n_neurons
                            > i
                            > (self.n_neurons - self.n_inhib)
                        ):
                            self.weight_mat[i][j] *= self.inhib_multiplier

            return True

        except Exception as e:
            logging.exception(e)
            return False
