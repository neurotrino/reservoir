import math
import numpy as np


def percent_nonzero(a):
    """Return the percent of array entries which are non-zero."""
    return 100 * (np.where(a != 0)[0].shape[0] / math.prod(a.shape))


def subset(w, p=0.05):
    """
    Return indices for percent `p` of non-zero weights in `w` in
    `np.where()` format.
    """
    if p < 0 or p > 100:
        raise ValueError(f"expected percentage in range [0.0, 1.0], got {p}")

    # Indices of non-zero weights in `w`
    candidates = np.where(w != 0)
    num_candidates = candidates[0].shape[0]

    # Select percent `p` of the non-zero indices
    meta_indices = np.random.choice(
        np.arange(num_candidates),
        int(p * num_candidates),
        replace=False
    )
    return tuple([w[meta_indices] for w in candidates])


def silence_randomly(w, p=0.05):
    """Randomly silence percent `p` of the nonzero weights in `w`."""
    target_indices = subset(w, p)
    w[target_indices] = 0
