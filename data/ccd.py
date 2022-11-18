"""Data for a conherence change detection (CCD) task."""

from data.base import BaseDataGenerator
from scipy.sparse import load_npz

import numpy as np
import tensorflow as tf

# ┬───────────────────────────────────────────────────────────────────────────╮
# ┤ Data Serving                                                              │
# ┴───────────────────────────────────────────────────────────────────────────╯


class DataGenerator(BaseDataGenerator):
    """Serve coherence change detection (CCD) task data.

    Stores and serves data associated with a CCD task. Data is
    formatted as spike trains (features) and coherence values (labels).
    """

    def __init__(self, cfg):
        """Initialize ccd.DataGenerator object."""
        super().__init__(cfg)

        # Generate initial dataset
        seq_len = cfg["data"].seq_len  # no. inputs
        n_input = cfg["data"].n_input  # dim of input

        rates = np.load(cfg["data"].rate_npy)
        coherences = load_npz(cfg["data"].coh_npz)  # shape 60 x 40800

        x = rates
        y = np.array(coherences.todense().reshape((-1, seq_len)))[
            :, :, None
        ]  # shape 600 x 4080 x 1

        if cfg["model"].cell.likelihood_output:
            # depending on y's value (0 or 1) at any seq_len point of each trial,
            # create new 2-row output. first row copies y, second inverts it.
            coh_1 = np.copy(
                y
            )  # 100% coherence subset; bc when actual y=0, the
            # likelihood of 100% coherence is 0, and when y=1, the likelihood of
            # 100% coherence is 1.
            coh_0 = np.zeros(np.shape(coh_1))  # 15% coherence subset; bc when
            # actual y=0, the likelihood of 15% coherence is 1, and when y=1,
            # the likeliihood of 15% coherence is 0.
            # wherever coh_1 was 0, make coh_0 nonzero
            coh_0[np.nonzero(coh_1 == 0)] = 1
            # combine outputs
            y = np.concatenate([coh_0, coh_1], -1)  # (600, 4080, 2)

        if cfg["model"].cell.categorical_output:
            cat = np.copy(y) * 0.5  # zeros remain 0
            cat[cat == 0] = 2.0  # zeros become 2
            # even though these are target values, they are ratios
            # therefore one coherence level does not create greater overall activity than the other
            y = cat

        if cfg["model"].cell.swap_output_labels:
            y[y==0] = 2 # zeros become twos
            y[y==1] = 0 # ones become zeros
            y[y==2] = 1 # twos (originally zeros) become ones; swap successful 

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
