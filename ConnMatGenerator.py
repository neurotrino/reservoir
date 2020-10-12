# version: Python 3.7
__author__ = 'Tarek, Isabel'

import csv
import numpy
from os import path, makedirs


class ConnectivityMatrixGenerator(object):

    def __init__(self, n_neurons, p, mu, sigma):

        self.n_neurons = n_neurons

        # Initialize connectivity matrix
        self.conn_mat = numpy.zeros((self.n_neurons, self.n_neurons))

        # Initialize weight matrix
        self.weight_mat = numpy.zeros((self.n_neurons, self.n_neurons))

        self.mu = mu
        self.sigma = sigma

        # Calculate total number of connections per neuron (remove neuron from target if included (ee and ii))
        self.k = int(round(p * (self.n_neurons - 1)))

    def run_generator(self):

        try:

            # Generate connectivity matrix and check that it's successful
            if not self.generate_conn_mat():
                raise Exception('Failed to generate connectivity matrix.')
            print("Connectivity Matrix Generated")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception('Failed to weight connectivity matrix.')
            print("Weighted")

            return self.weight_mat/10

        except Exception as e:
            print(e)
            return False

    def generate_conn_mat(self):

        try:

            for n in range(0, self.n_neurons):
                for a in range(0, self.k):
                    rand = numpy.random.randint(0, self.n_neurons)
                    while rand == n or self.conn_mat[n][rand] == 1:
                        rand = numpy.random.randint(0, self.n_neurons)
                    self.conn_mat[n][rand] = 1

            return True

        except Exception as e:
            print(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn_mat[i][j] == 1:
                        self.weight_mat[i][j] = (numpy.random.lognormal(self.mu, self.sigma))

            return True

        except Exception as e:
            print(e)
            return False
