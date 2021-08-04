"""Data for a delayed match to category (DMC) task."""

from data.base import BaseDataGenerator
from scipy.sparse import load_npz

import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Data Serving                                                              │
#┴───────────────────────────────────────────────────────────────────────────╯


class DataGenerator(BaseDataGenerator):
    """TODO - DOCS"""
    def __init__(self, cfg):
        super().__init__(cfg)

        # Generate initial dataset
        seq_len = cfg['data'].seq_len  # no. inputs
        n_input = cfg['data'].n_input  # dim of input

        FRAMES_PER_TRIAL=240
        FRAMERATE = 60
        MS_PER_TRIAL = (FRAMES_PER_TRIAL // FRAMERATE) * 1000

        spikes = load_npz(cfg['data'].spike_npz)
        matches = np.load(cfg['data'].match_npy)
        #match_labels_frames = matches.reshape(600, FRAMES_PER_TRIAL)
        # dilate the coherence vectors from frames to ms
        # sanity check: 2400*17 = 10*4080 = 40800ms per movie
        #match_labels_ms = np.repeat(match_labels_frames, int(MS_PER_TRIAL/FRAMES_PER_TRIAL)+1, axis=1)

        x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
        #y = np.array(matches.todense().reshape((-1, seq_len)))[:,:,None]
        #y = match_labels_ms.reshape((-1,seq_len))[:,:,None]
        y = matches.reshape((-1,seq_len))[:,:,None]

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        ).repeat(
            count=1
        ).batch(cfg['train'].batch_size)


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
