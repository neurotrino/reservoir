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

        # Generate initial dataset
        seq_len = cfg['data'].seq_len  # no. inputs
        n_input = cfg['data'].n_input  # dim of input

        x = tf.random.uniform(shape=(seq_len, n_input))[None] * 0.5
        y = tf.sin(tf.linspace(0.0, 4 * np.pi, seq_len))[None, :, None]

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        ).repeat(
            count=cfg['train'].batch_size * cfg['train'].n_batch
        ).batch(cfg['train'].batch_size)

    def get(self):
        return self.dataset


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
