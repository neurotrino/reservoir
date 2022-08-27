"""Get changes in the Adam Optimizer's learning rate across epochs."""

import numpy as np
import os

data_dir = "/data/experiments/"
experiment = "fwd-pipeline-inputspikeregen-newl23-owerlr-runlonger"
epochs_per_file = 10
epoch_of_interest = "1-10.npz"
savepath = "/data/results/Adam_lrs.png"
init_main_lr = 0.001
init_out_lr = 0.00001


# [!] This function is incorrect. "batch of interest" is actually
#     "epoch of interest", as the "epoch of interest" variable is a
#     file containing data for multiple epochs
def single_epoch_lr(experiment, epoch_of_interest, batch_of_interest=1):
    """Get learning rate data from a single epoch."""

    # Fetch data
    datapath = os.path.join(
        data_dir, experiment, "npz-data", epoch_of_interest
    )
    data = np.load(datapath)

    # Unpack data
    new_in_w = data["tv0.postweights"][batch_of_interest]
    new_main_w = data["tv1.postweights"][batch_of_interest]
    new_out_w = data["tv2.postweights"][batch_of_interest]
    in_grad = data["tv0.gradients"][batch_of_interest]
    main_grad = data["tv1.gradients"][batch_of_interest]
    out_grad = data["tv2.postweights"][batch_of_interest]
    old_in_w = data["tv0.postweights"][batch_of_interest - 1]
    old_main_w = data["tv1.postweights"][batch_of_interest - 1]
    old_out_w = data["tv2.postweights"][batch_of_interest - 1]

    # Calculate learning rate values
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
