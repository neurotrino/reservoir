"""TODO: module docs"""

from data.base import BaseDataGenerator as BaseDataGenerator

import numpy as np
import tensorflow as tf
import scipy.sparse

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

        # in total: 600 trials
        # spikes: (600x4080, 16)
        spikes = scipy.sparse.load_npz('/home/macleanlab/mufeng/NaturalMotionCNN/CNN_outputs/spike_train.npz')
        # cohs: (60, 40800)
        coherences = scipy.sparse.load_npz('/home/macleanlab/mufeng/NaturalMotionCNN/CNN_outputs/coherences.npz')
        x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
        y = np.array(coherences.todense().reshape((-1, seq_len)))[:,:,None]

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        ).repeat(
            count=1
        ).batch(cfg['train'].batch_size)

    def get(self):
        return self.dataset


def load_data(cfg):
    """Wrapper returning just the dataset."""
    return DataGenerator(cfg).dataset
