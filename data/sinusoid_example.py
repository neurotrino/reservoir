"""TODO: module docs"""

import numpy as np
import tensorflow as tf

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Generators                                                                │
#┴───────────────────────────────────────────────────────────────────────────╯

# N/A

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Preprocessing                                                             │
#┴───────────────────────────────────────────────────────────────────────────╯

# N/A

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Private Functions                                                         │
#┴───────────────────────────────────────────────────────────────────────────╯

# N/A

#┬───────────────────────────────────────────────────────────────────────────╮
#┤ Public Functions                                                          │
#┴───────────────────────────────────────────────────────────────────────────╯

def load_data(config):
    """Sinusoidal data with noise.

    20 repetitions of truly the exact same input sinusoid. Should
    create a set with random small displacements along the x and y
    axes.
    """
    uni = config['uni']
    train_cfg = config['train']

    x = tf.random.uniform(shape=(uni.seq_len, uni.n_input))[None] * 0.5
    y = tf.sin(tf.linspace(0.0, 4 * np.pi, uni.seq_len))[None, :, None]

    # repeat set of x,y over batches to form full dataset
    data = tf.data.Dataset.from_tensor_slices(
        (x, y)
    ).repeat(count=train_cfg.batch_size).batch(train_cfg.n_batch)
    print()
    print()
    print(data)
    print()
    print()
    return data
