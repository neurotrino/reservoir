# process hdf5 weight files
"""
import sys
sys.path.append("tf2_migration/")
from process_hdf5 import *
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

# eventually create loop
# for f in fullconn_w:
# hf = h5py.File(f, 'r')
def basic_w_vis():
    hf = h5py.File("tf2_testing/LIF/p50/test_epoch_0.hdf5",'r')
    n1 = hf.get('rnn')
    n2 = n1.get('rnn')
    lif_cell = n2.get('lif_cell')
    in_w = lif_cell.get('input_weights:0')
    in_w = np.array(in_w)
    rec_w = lif_cell.get('recurrent_weights:0')
    rec_w = np.array(rec_w)

    # plot to check that distribution of weights is lognormal
    # (or that there are a lot of zeros if the matrix is sparse)
    fig, axes = plt.subplots(2)
    axes[0].hist(in_w)
    axes[0].set_title('input weights')
    axes[1].hist(rec_w)
    axes[1].set_title('recurrent weights')
    fig.subplots_adjust(hspace=.5)
    plt.show()
    plt.savefig('tf2_testing/LIF/p50/test_epoch_0_weights.png')

def LIF_EI_begin_end_compare(epoch):
    begin_fname = "tf2_testing/LIF_EI/begin_epoch_" + str(epoch) + ".hdf5"
    hf = h5py.File(begin_fname,'r')
    n1 = hf.get('rnn')
    n2 = n1.get('rnn')
    lif_ei = n2.get('lif_ei')
    rec_w = lif_ei.get('recurrent_weights:0')
    begin_rec_w = np.array(rec_w)

    print("count of zeros at beginning of epoch:")
    print(begin_rec_w[begin_rec_w==0].shape[0])
    begin_diag = begin_rec_w.diagonal()
    print("count of self-recurrent synapses that are zero")
    print(begin_diag[begin_diag==0].shape[0])

    end_fname = "tf2_testing/LIF_EI/end_epoch_" + str(epoch) + ".hdf5"
    hf = h5py.File(end_fname,'r')
    n1 = hf.get('rnn')
    n2 = n1.get('rnn')
    lif_ei = n2.get('lif_ei')
    rec_w = lif_ei.get('recurrent_weights:0')
    end_rec_w = np.array(rec_w)

    print("count of zeros at end of epoch:")
    print(end_rec_w[end_rec_w==0].shape[0])
    end_diag = end_rec_w.diagonal()
    print("count of self-recurrent synapses that are zero")
    print(end_diag[end_diag==0].shape[0])

    fig, axes = plt.subplots(2)
    axes[0].hist(begin_rec_w)
    axes[0].set_title('epoch beginning weights')
    axes[1].hist(end_rec_w)
    axes[1].set_title('epoch ending weights')
    fig.subplots_adjust(hspace=.5)
    plt.show()
    out_fname = "tf2_testing/LIF_EI/compare_epoch_" + str(epoch) + "_weights.png"
    plt.savefig(out_fname)


def LIF_epoch_begin_end_compare(p, epoch):
    begin_fname = "tf2_testing/LIF/p" + str(int(p*100)) + "/begin_epoch_" + str(epoch) + ".hdf5"
    hf = h5py.File(begin_fname,'r')
    n1 = hf.get('rnn')
    n2 = n1.get('rnn')
    lif_cell = n2.get('lif_cell')
    rec_w = lif_cell.get('recurrent_weights:0')
    begin_rec_w = np.array(rec_w)

    print("count of zeros at beginning of epoch:")
    print(begin_rec_w[begin_rec_w==0].shape[0])
    begin_diag = begin_rec_w.diagonal()
    print("count of self-recurrent synapses that are zero")
    print(begin_diag[begin_diag==0].shape[0])

    end_fname = "tf2_testing/LIF/p" + str(int(p*100)) + "/end_epoch_" + str(epoch) + ".hdf5"
    hf = h5py.File(end_fname,'r')
    n1 = hf.get('rnn')
    n2 = n1.get('rnn')
    lif_cell = n2.get('lif_cell')
    rec_w = lif_cell.get('recurrent_weights:0')
    end_rec_w = np.array(rec_w)

    print("count of zeros at end of epoch:")
    print(end_rec_w[end_rec_w==0].shape[0])
    end_diag = end_rec_w.diagonal()
    print("count of self-recurrent synapses that are zero")
    print(end_diag[end_diag==0].shape[0])

    fig, axes = plt.subplots(2)
    axes[0].hist(begin_rec_w)
    axes[0].set_title('epoch beginning weights')
    axes[1].hist(end_rec_w)
    axes[1].set_title('epoch ending weights')
    fig.subplots_adjust(hspace=.5)
    plt.show()
    out_fname = "tf2_testing/LIF/p" + str(int(p*100)) + "/compare_epoch_" + str(epoch) + "_weights.png"
    plt.savefig(out_fname)
