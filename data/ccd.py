"""Data for a conherence change detection (CCD) task."""

from data.base import BaseDataGenerator
from scipy.sparse import load_npz

import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Data Serving                                                              │
#┴───────────────────────────────────────────────────────────────────────────╯

class DataGenerator(BaseDataGenerator):
    """Serve coherence change detection (CCD) task data.

    Stores and serves data associated with a CCD task. Data is
    formatted as spike trains (features) and coherence values (labels).
    """
    def __init__(self, cfg):
        """Initialize ccd.DataGenerator object."""
        super().__init__(cfg)

        # Generate initial dataset
        seq_len = cfg['data'].seq_len  # no. inputs
        n_input = cfg['data'].n_input  # dim of input

        #spikes = load_npz(cfg['data'].spike_npz)
        rates = np.load(cfg['data'].rate_npy)
        coherences = load_npz(cfg['data'].coh_npz) # shape 60 x 40800

        #x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))

        # rates are already dense in shape [600, 4080, 16]
        # repeat and shuffle (?)
        # generate spikes from x_rates
        #random_matrix = np.random.rand(rates.shape[0], rates.shape[1], rates.shape[2])
        #spikes = (rates - random_matrix > 0)*1.
        #x = spikes # this step is actually quite rapid and COULD be done in real time as needed
        x = rates
        y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None] # shape 600 x 4080 x 1

        self.dataset = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .repeat(count=1)
            .batch(cfg["train"].batch_size)
            .shuffle(x.shape[0], reshuffle_each_iteration=True)
        )

        # this creates a dataset by batches of size 10 trials, with a total upper limit of 600 trials
        # however, we only ever iterate (in trainer std_single_task) n_batch (10) times for each epoch,
        # so only the first 100 trials are ever used.
        # e.g. the first 10 x 40800 in coherences, and the first 100 x 4080 in y

        # Declare data iterator
        self.iterator = None


    def get(self):
        """Retrieve a complete dataset."""
        return self.dataset


    def next(self):
        """Retrieve a single batch from the dataset."""
        if self.iterator is None:
            self.iterator = iter(self.dataset)
        try:
            return self.iterator.get_next()
        except tf.errors.OutOfRangeError:
            self.iterator = None
            return self.next()


def load_data(cfg):
    """Wrapper returning just the dataset."""
    return DataGenerator(cfg).dataset
