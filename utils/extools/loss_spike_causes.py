# Script for plotting / determining what happens in connectivity to cause spikes in loss during training

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from compare_losses import filenames
from fn_analysis import reciprocity

data_dir = '/data/experiments/'
num_epochs = 1000
epochs_per_file = 10
loss_of_interest="step_loss"

# begin with main lr 0.005, output lr 0.00001
experiment = "fwd-pipeline-inputspikeregen-newl23-onlyoutputlrlower"
savepath = '/data/results/fwd/loss_spike_causes_0.005.png'
# move on to others as you desire

# create four subplots
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# comparison is always for the previous epoch's postweights (current preweights) and current epoch total loss
data_files = filenames(num_epochs, epochs_per_file)
losses = []
eiratio_in = []
eiratio_main = []
eiratio_out = []
recip_main = []

for filename in data_files:
    filepath = os.path.join(data_dir, experiment, "npz-data", filename)
    data = np.load(filepath)
    losses += data[loss_of_interest].tolist()
    w_in = data['tv0.postweights']
    for i in range(0,len(w_in)):
        w = w_in[i]
        e = w[w>0]
        i = w[w<0]
        eiratio_in = np.append(eiratio_in, np.abs(np.sum(e)/np.sum(i)))
    w_main = data['tv1.postweights']
    for i in range(0,len(w_main)):
        w = w_main[i]
        recip_main = np.append(recip_main, reciprocity(w))
        e = w[w>0]
        i = w[w<0]
        eiratio_main = np.append(eiratio_main, np.abs(np.sum(e)/np.sum(i)))
    w_out = data['tv2.postweights']
    for i in range(0,len(w_out)):
        w = w_out[i]
        e = w[w>0]
        i = w[w<0]
        eiratio_out = np.append(eiratio_out, np.abs(np.sum(e)/np.sum(i)))

# plot main e/i ratio (weighted)
ax1.plt.scatter(loss,eiratio_in)
ax1.set_xlabel('loss')
ax1.set_ylabel('input e/i ratio')
ax2.plt.scatter(loss,eiratio_main)
ax2.set_xlabel('loss')
ax2.set_ylabel('main e/i ratio')
ax3.plt.scatter(loss,recip_main)
ax3.set_xlabel('loss')
ax3.set_ylabel('main reciprocity')
ax4.plt.scatter(loss,eiratio_out)
ax4.set_xlabel('loss')
ax4.set_ylabel('output e/i ratio')
fig.suptitle("main lr 0.005, output lr 0.00001")
plt.draw()
plt.savefig(savepath)
plt.clf()
plt.close()

# plot output e/i ratio (weighted)
# plot main density v loss
# plot main reciprocity v loss
# plot main number of transitions from zero t0 nonzero v loss
