# process hdf5 weight files
import numpy as np
import matplotlib.pyplot as plt
import h5py

# eventually create loop
# for f in fullconn_w:
# hf = h5py.File(f, 'r')
hf = h5py.File("test_epoch_10.hdf5",'r')
n1 = hf.get('rnn')
n2 = n1.get('rnn')
lif_cell = n2.get('lif_cell')
in_w = lif_cell.get('input_weights:0')
in_w = np.array(in_w)
rec_w = lif_cell.get('recurrent_weights:0')
rec_w = np.array(rec_w)

# plot to check that distribution of weights is lognormal
# (or that there are a lot of zeros if the matrix is sparse)
plt.hist(rec_w)
plt.show()
plt.savefig('test_epoch_10_weights.png')
