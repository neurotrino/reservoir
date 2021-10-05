# plot the loss over time for networks of different sizes and components

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

data_dir = /data/experiments/
experiments = ['ccd_200_lif_sparse','ccd_200_lif_rewiring','ccd_500_lif_sparse','ccd_500_lif_rewiring']

def compare_losses(experiments):
    epoch_groups = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120','121-130','131-140','141-150','151-160','161-170','171-180','181-190','191-200']
    nfiles = len(epoch_groups)
    plt.figure()
    for i in range(len(experiments)):
        loss = []
        for j in range(nfiles):
            fname = data_dir + experiments[i] + '/npz-data/' + epoch_groups[j] + '.npz'
            data = np.load(fname)
            for k in range(len(data['epoch_loss'])):
                loss.append(data['epoch_loss'][k])
        plt.plot(loss, label=experiments[i])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('LIF network on CCD task')
    plt.draw()
    plt.savefig('/data/results/ccd/compare_lif_network_size.png')
