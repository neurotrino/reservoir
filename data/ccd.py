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

        spikes = load_npz(cfg['data'].spike_npz)
        coherences = load_npz(cfg['data'].coh_npz)

        x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
        y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        ).repeat(
            count=1
        ).batch(cfg['train'].batch_size)

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
