# Script for calculating changing learning rates across epochs from Adam Optimizer

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

data_dir = "/data/experiments/"
experiment = "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger"
epochs_per_file = 10
epoch_of_interest = "1-10.npz"
batch_of_interest = 1
savepath = "/data/results/Adam_lrs.png"
init_main_lr = 0.001
init_out_lr = 0.00001


def single_epoch_lr(experiment, epoch_of_interest, batch_of_interest):
    # determine learning rates for variables over a single epoch
    datapath = os.path.join(
        data_dir, experiment, "npz-data", epoch_of_interest
    )
    data = np.load(datapath)
    new_in_w = data["tv0.postweights"][batch_of_interest]
    new_main_w = data["tv1.postweights"][batch_of_interest]
    new_out_w = data["tv2.postweights"][batch_of_interest]
    in_grad = data["tv0.gradients"][batch_of_interest]
    main_grad = data["tv1.gradients"][batch_of_interest]
    out_grad = data["tv2.postweights"][batch_of_interest]
    old_in_w = data["tv0.postweights"][batch_of_interest - 1]
    old_main_w = data["tv1.postweights"][batch_of_interest - 1]
    old_out_w = data["tv2.postweights"][batch_of_interest - 1]
    in_lr = np.divide(
        np.abs(new_in_w - old_in_w),
        np.abs(in_grad),
        out=np.zeros_like(new_in_w - old_in_w),
        where=in_grad != 0,
    )
    main_lr = np.divide(
        np.abs(new_main_w - old_main_w),
        np.abs(main_grad),
        out=np.zeros_like(new_main_w - old_main_w),
        where=main_grad != 0,
    )
    out_lr = np.divide(
        np.abs(new_out_w - old_out_w),
        np.abs(out_grad),
        out=np.zeros_like(new_out_w - old_out_w),
        where=out_grad != 0,
    )
    return in_lr, main_lr, out_lr
