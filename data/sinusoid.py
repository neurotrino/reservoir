"""Sinusoidal data with noise.

20 repetitions of truly the exact same input sinusoid. Should
create a set with random small displacements along the x and y
axes.
"""

from data.base import BaseDataGenerator as BaseDataGenerator

import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Data Serving                                                              │
#┴───────────────────────────────────────────────────────────────────────────╯

class DataGenerator(BaseDataGenerator):
    """

    [*] Naming convention of DataGenerator in each data script, because
    when invoked, it should be invoked as `sinusoid.DataGenerator`,
    which is self-documenting
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # Generate a sinusoid pattern
        seq_len = cfg['data'].seq_len
        n_input = cfg['data'].n_input

        x = tf.random.uniform(shape=(seq_len, n_input))[None] * 0.5
        y = tf.sin(tf.linspace(0.0, 4 * np.pi, seq_len))[None, :, None]

        # Repeat the sinusoid pattern enough times for a single epoch,
        # formatted as batches
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        ).repeat(
            count=cfg['train'].batch_size
        ).batch(
            cfg['train'].batch_size
        )

        # Iterator
        self.iterator = None


    def get(self):
        return self.dataset


    def next(self):
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


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Preprocessing                                                             │
#┴───────────────────────────────────────────────────────────────────────────╯

# N/A


#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Postprocessing                                                            │
#┴───────────────────────────────────────────────────────────────────────────╯

# N/A
