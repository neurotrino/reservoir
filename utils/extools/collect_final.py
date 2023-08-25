"""final plot generation for series of completed experiments; synthesis of things in analyze_final"""

# ---- external imports -------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.ma as ma
import os
import scipy
import sys
import seaborn as sns
import networkx as nx

from scipy.sparse import load_npz
from scipy.stats import kstest

# ---- internal imports -------------------------------------------------------
sys.path.append("../")
sys.path.append("../../")

from utils.misc import *
from utils.extools.analyze_structure import get_degrees
from utils.extools.fn_analysis import *
from utils.extools.MI import *
from utils.extools.analyze_structure import _nets_from_weights
from utils.extools.analyze_dynamics import *
from utils.extools.analyze_final import *

# ---- global variables -------------------------------------------------------
data_dir = '/data/experiments/'
experiment_string = 'run-batch30-specout-onlinerate0.1-savey'
task_experiment_string = 'run-batch30-onlytaskloss'
rate_experiment_string = 'run-batch30-onlyrateloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
i_end = 300

n_input = 16
seq_len = 4080

savepath = '/data/results/experiment1/'

e_only = True
positive_only = False
bin = 10

naive_batch_id = 0
trained_batch_id = 99
coh_lvl = 'coh0'
NUM_EXCI = 240

# Paul Tol's colorblind-friendly palettes for scientific visualization
vibrant = [
    '#0077BB',#blue
    '#33BBEE',#cyan
    '#009988',#teal
    '#EE7733',#orange
    '#CC3311',#red
    '#EE3377',#magenta
    '#BBBBBB'#grey
]

muted = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    '#999933',  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]

light = [
    '#77AADD',#light blue
    '#99DDFF',#light cyan
    '#44BB99',#mint
    '#BBCC33',#pear
    '#AAAA00',#olive
    '#EEDD88',#light yellow
    '#EE8866',#orange
    '#FFAABB',#pink
    '#DDDDDD',# pale grey
]

"""
# snippet of code to go thru a bunch of experiments that all contain a particular string
data_dirs = get_experiments(data_dir, experiment_string)
data_files = filenames(num_epochs, epochs_per_file)
fig, ax = plt.subplots(nrows=5, ncols=1)
# get your usual experiments first
for xdir in data_dirs:
"""

data_files = filenames(num_epochs, epochs_per_file)

#ALL DUAL TRAINED TO BEGIN WITH:
unspec_dirs = ["fwd-pipeline-inputspikeregen"]
spec_output_dirs = ["run-batch30-specout-onlinerate0.1-savey","run-batch30-dualloss-silence","run-batch30-dualloss-swaplabels"]
spec_input_dirs = ["run-batch30-dualloss-specinput0.3-rewire"]
spec_nointoout_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
save_inz_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
save_inz_dirs_task = ["run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
spec_nointoout_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
spec_nointoout_dirs_task = ["run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
all_spring_dual_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
all_save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]

lowerinhib_data_dirs = ["2023-07-03 21.12.39","2023-07-08 17.53.59"]

# this is all a general sort of thing, once you do one (mostly figure out shading and dist comparisons) it'll come easily

def get_unspec_info(exp_dirs=unspec_dirs):
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    in_diff = []
    rec_diff = []
    out_diff = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        filepath = os.path.join(data_dir, xdir, "npz-data", "1-10.npz")
        data = np.load(filepath)
        naive_in_w = data['tv0.postweights'][0]
        naive_w = data['tv1.postweights'][0]
        naive_out_w = data['tv2.postweights'][0]

        filepath = os.path.join(data_dir, xdir, "npz-data", "491-500.npz")
        data = np.load(filepath)
        trained_in_w = data['tv0.postweights'][99]
        trained_w = data['tv1.postweights'][99]
        trained_out_w = data['tv2.postweights'][99]

        # mean difference between weights
        in_diff.append(np.abs(np.mean(trained_in_w - naive_in_w)))
        rec_diff.append(np.abs(np.mean(trained_w - naive_w)))
        out_diff.append(np.abs(np.mean(trained_out_w - naive_out_w)))

    return [in_diff,rec_diff,out_diff]

def tracking_top_weights(exp_dirs=all_spring_dual_dirs,exp_season='spring',percentile=0.90):
    # track top decile of weights
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    # collectors
    p_same = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if not '06.03.22' in np_dir: # do not include that one awful experiment
            exp_path = xdir[-9:-1]

            # get truly naive weights
            filepath = os.path.join(data_dir,xdir,"npz-data","main_preweights.npy")
            w = np.load(filepath)
            thresh = np.quantile(np.abs(w[w > 0]), percentile)
            naive_idx = np.argwhere(w>=thresh)

            filepath = os.path.join(data_dir, xdir, "npz-data", "991-1000.npz")
            data = np.load(filepath)
            w = data['tv1.postweights'][99]
            # get top decile
            thresh = np.quantile(np.abs(w[w > 0]), percentile)
            trained_idx = np.argwhere(w>=thresh)

            aset = set([tuple(x) for x in naive_idx])
            bset = set([tuple(x) for x in trained_idx])
            sames = np.array([x for x in aset & bset])
            p_same.append(len(sames)/len(naive_idx)*100)

    return p_same

def dists_of_all_weights(dual_exp_dir=spec_nointoout_dirs_task,exp_season='spring_task'):
    # aggregate over all experiments of this type
    # plot distributions in naive state
    # plot distributions in trained state
    # input layer to e and i; rec ee ei ie ii,
    # (((later create comparable plot according to tuning)))
    # do for rate trained as well
    # do for task trained as well
    # make everything on comparable axes
    # look for the 5x input multiplier; do something about that (plot separately / not at all?)
    # return means and stds (even if not proper) of the weight distributions, or save in some file

    # from actual experiment now
    for exp_string in dual_exp_dir:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    all_w_in = []
    all_w_rec = []
    all_w_out = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        w_in = []
        w_rec = []
        w_out = []

        # get the truly naive weights
        filepath = os.path.join(data_dir,xdir,"npz-data","input_preweights.npy")
        if 'inputx5' in xdir:
            input_w = np.load(filepath)
        else:
            input_w = np.load(filepath)*5
        filepath = os.path.join(data_dir,xdir,"npz-data","main_preweights.npy")
        w = np.load(filepath)
        filepath = os.path.join(data_dir,xdir,"npz-data","output_preweights.npy")
        output_w = np.load(filepath)

        # aggregate
        # over training time but also vstack into all_w later
        w_in.append(input_w)
        w_rec.append(w)
        w_out.append(output_w)
        # [experiment x epoch x 300 x 300]

        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            if 'inputx5' in xdir:
                input_w = data['tv0.postweights'][0] # just the singular for now; too much data and noise otherwise
            else:
                input_w = data['tv0.postweights'][0]*5
            rec_w = data['tv1.postweights'][0]
            out_w = data['tv2.postweights'][0]

            w_in.append(input_w)
            w_rec.append(rec_w)
            w_out.append(out_w)

        """
        if not "all_w_in" in locals():
            all_w_in = w_in
        else:
            all_w_in = np.vstack([all_w_in, w_in])

        if not "all_w_rec" in locals():
            all_w_rec = w_rec
        else:
            all_w_rec = np.vstack([all_w_rec, w_rec])

        if not "all_w_out" in locals():
            all_w_out = w_out
        else:
            all_w_out = np.vstack([all_w_out, w_out])
        """
        all_w_in.append(w_in)
        all_w_rec.append(w_rec)
        all_w_out.append(w_out)

    all_w_in = np.array(all_w_in)
    all_w_rec = np.array(all_w_rec)
    all_w_out = np.array(all_w_out)

    #return [all_w_in,all_w_rec,all_w_out]

    # CHARACTERIZE DISTRIBUTIONS
    end_idx = np.shape(all_w_in)[1]-1
    means = []
    stds = []
    # in to e
    means.append(np.mean(all_w_in[:,0,:,:e_end]))
    stds.append(np.std(all_w_in[:,0,:,:e_end]))
    # in to i
    means.append(np.mean(all_w_in[:,0,:,e_end:]))
    stds.append(np.std(all_w_in[:,0,:,e_end:]))
    # rec ee
    means.append(np.mean(all_w_rec[:,0,:e_end,:e_end]))
    stds.append(np.std(all_w_rec[:,0,:e_end,:e_end]))
    # rec ei
    means.append(np.mean(all_w_rec[:,0,:e_end,e_end:]))
    stds.append(np.std(all_w_rec[:,0,:e_end,e_end:]))
    # rec ie
    means.append(np.mean(all_w_rec[:,0,e_end:,:e_end]))
    stds.append(np.std(all_w_rec[:,0,e_end:,:e_end]))
    # rec ii
    means.append(np.mean(all_w_rec[:,0,e_end:,e_end:]))
    stds.append(np.std(all_w_rec[:,0,e_end:,e_end:]))
    # e to out
    means.append(np.mean(all_w_out[:,0,:e_end,:]))
    stds.append(np.std(all_w_out[:,0,:e_end,:]))
    # i to out
    means.append(np.mean(all_w_out[:,0,e_end:,:]))
    stds.append(np.std(all_w_out[:,0,e_end:,:]))

    naive_means = means
    naive_stds = stds

    # now do trained
    means = []
    stds = []
    # in to e
    means.append(np.mean(all_w_in[:,end_idx,:,:e_end]))
    stds.append(np.std(all_w_in[:,end_idx,:,:e_end]))
    # in to i
    means.append(np.mean(all_w_in[:,end_idx,:,e_end:]))
    stds.append(np.std(all_w_in[:,end_idx,:,e_end:]))
    # rec ee
    means.append(np.mean(all_w_rec[:,end_idx,:e_end,:e_end]))
    stds.append(np.std(all_w_rec[:,end_idx,:e_end,:e_end]))
    # rec ei
    means.append(np.mean(all_w_rec[:,end_idx,:e_end,e_end:]))
    stds.append(np.std(all_w_rec[:,end_idx,:e_end,e_end:]))
    # rec ie
    means.append(np.mean(all_w_rec[:,end_idx,e_end:,:e_end]))
    stds.append(np.std(all_w_rec[:,end_idx,e_end:,:e_end]))
    # rec ii
    means.append(np.mean(all_w_rec[:,end_idx,e_end:,e_end:]))
    stds.append(np.std(all_w_rec[:,end_idx,e_end:,e_end:]))
    # e to out
    means.append(np.mean(all_w_out[:,end_idx,:e_end,:]))
    stds.append(np.std(all_w_out[:,end_idx,:e_end,:]))
    # i to out
    means.append(np.mean(all_w_out[:,end_idx,e_end:,:]))
    stds.append(np.std(all_w_out[:,end_idx,e_end:,:]))

    # COMPARE DISTRIBUTIONS
    # we want to show how similar and different are the rates of

    in_e_naive = all_w_in[:,0,:,:e_end]
    in_e_trained = all_w_in[:,end_idx,:,:e_end]
    in_i_naive = all_w_in[:,0,:,e_end:]
    in_i_trained = all_w_in[:,end_idx,:,e_end:]

    rec_ee_naive = all_w_rec[:,0,:e_end,:e_end]
    rec_ee_trained = all_w_rec[:,end_idx,:e_end,:e_end]
    rec_ei_naive = all_w_rec[:,0,:e_end,e_end:]
    rec_ei_trained = all_w_rec[:,end_idx,:e_end,e_end:]
    rec_ie_naive = all_w_rec[:,0,e_end:,:e_end]
    rec_ie_trained = all_w_rec[:,end_idx,e_end:,:e_end]
    rec_ii_naive = all_w_rec[:,0,e_end:,e_end:]
    rec_ii_trained = all_w_rec[:,end_idx,e_end:,e_end:]

    out_e_naive = all_w_out[:,0,:e_end,:]
    out_e_trained = all_w_out[:,end_idx,:e_end,:]
    out_i_naive = all_w_out[:,0,e_end:,:]
    out_i_trained = all_w_out[:,end_idx,e_end:,:]

    [D_in_e, p_in_e] = scipy.stats.kstest(in_e_naive.flatten(),in_e_trained.flatten())
    [D_in_i, p_in_i] = scipy.stats.kstest(in_i_naive.flatten(),in_i_trained.flatten())

    [D_rec_ee, p_rec_ee] = scipy.stats.kstest(rec_ee_naive.flatten(),rec_ee_trained.flatten())
    [D_rec_ei, p_rec_ei] = scipy.stats.kstest(rec_ei_naive.flatten(),rec_ei_trained.flatten())
    [D_rec_ie, p_rec_ie] = scipy.stats.kstest(rec_ie_naive.flatten(),rec_ie_trained.flatten())
    [D_rec_ii, p_rec_ii] = scipy.stats.kstest(rec_ii_naive.flatten(),rec_ii_trained.flatten())

    [D_naive_es,p_naive_es] = scipy.stats.kstest(rec_ee_naive.flatten(),rec_ei_naive.flatten())
    [D_naive_is,p_naive_is] = scipy.stats.kstest(rec_ie_naive.flatten(),rec_ii_naive.flatten())
    [D_trained_es,p_trained_es] = scipy.stats.kstest(rec_ee_trained.flatten(),rec_ei_trained.flatten())
    [D_trained_is,p_trained_is] = scipy.stats.kstest(rec_ie_trained.flatten(),rec_ii_trained.flatten())

    [D_out_e, p_out_e] = scipy.stats.kstest(out_e_naive.flatten(),out_e_trained.flatten())
    [D_out_i, p_out_i] = scipy.stats.kstest(out_i_naive.flatten(),out_i_trained.flatten())

    return [naive_means,naive_stds,means,stds,[D_in_e,D_in_i,D_rec_ee,D_rec_ei,D_rec_ie,D_rec_ii,D_out_e,D_out_i],[p_in_e,p_in_i,p_rec_ee,p_rec_ei,p_rec_ie,p_rec_ii,p_out_e,p_out_i],[D_naive_es,D_naive_is,D_trained_es,D_trained_is],[p_naive_es,p_naive_is,p_trained_es,p_trained_is]]

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax = ax.flatten()

    epochs = np.arange(0,np.shape(all_w_rec)[1])

    mean_w_e_in = np.mean(all_w_in[:,:,:,:e_end],(0,2,3))
    std_w_e_in = np.std(all_w_in[:,:,:,:e_end],(0,2,3))
    ax[0].plot(epochs,mean_w_e_in,color='slateblue',label='input to e')
    ax[0].fill_between(epochs,mean_w_e_in-std_w_e_in,mean_w_e_in+std_w_e_in,facecolor='slateblue',alpha=0.4)

    mean_w_i_in = np.mean(all_w_in[:,:,:,e_end:],(0,2,3))
    std_w_i_in = np.std(all_w_in[:,:,:,e_end:],(0,2,3))
    ax[0].plot(epochs,mean_w_i_in,color='orangered',label='input to i')
    ax[0].fill_between(epochs,mean_w_i_in-std_w_i_in,mean_w_i_in+std_w_i_in,facecolor='orangered',alpha=0.4)

    ax[0].set_title('input weights',fontname='Ubuntu')

    mean_w_ee_rec = np.mean(all_w_rec[:,:,:e_end,:e_end],(0,2,3))
    std_w_ee_rec = np.std(all_w_rec[:,:,:e_end,:e_end],(0,2,3))
    ax[1].plot(epochs,mean_w_ee_rec,color='slateblue',label='e to e')
    ax[1].fill_between(epochs,mean_w_ee_rec-std_w_ee_rec,mean_w_ee_rec+std_w_ee_rec,facecolor='slateblue',alpha=0.4)

    mean_w_ei_rec = np.mean(all_w_rec[:,:,:e_end,e_end:],(0,2,3))
    std_w_ei_rec = np.std(all_w_rec[:,:,:e_end,e_end:],(0,2,3))
    ax[1].plot(epochs,mean_w_ei_rec,color='mediumseagreen',label='e to i')
    ax[1].fill_between(epochs,mean_w_ei_rec-std_w_ei_rec,mean_w_ei_rec+std_w_ei_rec,facecolor='mediumseagreen',alpha=0.4)

    mean_w_ie_rec = np.mean(all_w_rec[:,:,e_end:,:e_end],(0,2,3))
    std_w_ie_rec = np.std(all_w_rec[:,:,e_end:,:e_end],(0,2,3))
    ax[1].plot(epochs,mean_w_ie_rec,color='darkorange',label='i to e')
    ax[1].fill_between(epochs,mean_w_ie_rec-std_w_ie_rec,mean_w_ie_rec+std_w_ie_rec,facecolor='darkorange',alpha=0.4)

    mean_w_ii_rec = np.mean(all_w_rec[:,:,e_end:,e_end:],(0,2,3))
    std_w_ii_rec = np.std(all_w_rec[:,:,e_end:,e_end:],(0,2,3))
    ax[1].plot(epochs,mean_w_ii_rec,color='orangered',label='i to i')
    ax[1].fill_between(epochs,mean_w_ii_rec-std_w_ii_rec,mean_w_ii_rec+std_w_ii_rec,facecolor='orangered',alpha=0.4)

    ax[1].set_title('recurrent weights',fontname='Ubuntu')

    mean_w_e_out = np.mean(all_w_out[:,:,:e_end,:],(0,2,3))
    std_w_e_out = np.std(all_w_out[:,:,:e_end,:],(0,2,3))
    ax[2].plot(epochs,mean_w_e_out,color='slateblue',label='e to output')
    ax[2].fill_between(epochs,mean_w_e_out-std_w_e_out,mean_w_e_out+std_w_e_out,facecolor='slateblue',alpha=0.4)

    mean_w_i_out = np.mean(all_w_out[:,:,e_end:,:],(0,2,3))
    std_w_i_out = np.std(all_w_out[:,:,e_end:,:],(0,2,3))
    ax[2].plot(epochs,mean_w_i_out,color='orangered',label='i to output')
    ax[2].fill_between(epochs,mean_w_i_out-std_w_i_out,mean_w_i_out+std_w_i_out,facecolor='orangered',alpha=0.4)

    ax[2].set_title('output weights',fontname='Ubuntu')

    plt.suptitle('Average weights over training time',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylabel('average weights',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    save_fname = spath+'/avg_weights_over_time_test.png'

    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

    # PLOT DISTRIBUTIONS OF NAIVE AND TRAINED WEIGHTS NOW
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()

    # input layer
    end_idx = np.shape(all_w_in)[1]-1
    ax[0].hist(all_w_in[:,0,:,:e_end][all_w_in[:,0,:,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.4,color='slateblue',label='naive in to e')
    ax[0].hist(all_w_in[:,end_idx,:,:e_end][all_w_in[:,end_idx,:,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.8,color='slateblue',label='trained in to e')

    ax[0].hist(all_w_in[:,0,:,e_end:][all_w_in[:,0,:,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.4,color='orangered',label='naive in to i')
    ax[0].hist(all_w_in[:,end_idx,:,e_end:][all_w_in[:,end_idx,:,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.8,color='orangered',label='trained in to i')

    ax[0].set_title('input weights',fontname='Ubuntu')

    # output layer
    ax[1].hist(all_w_out[:,0,:e_end,:][all_w_out[:,0,:e_end,:]!=0].flatten(),density=True,bins=30,alpha=0.4,color='slateblue',label='naive e to out')
    ax[1].hist(all_w_out[:,end_idx,:e_end,:][all_w_out[:,end_idx,:e_end,:]!=0].flatten(),density=True,bins=30,alpha=0.8,color='slateblue',label='trained e to out')

    ax[1].hist(all_w_out[:,0,e_end:,:][all_w_out[:,0,e_end:,:]!=0].flatten(),density=True,bins=30,alpha=0.4,color='orangered',label='naive i to out')
    ax[1].hist(all_w_out[:,end_idx,e_end:,:][all_w_out[:,end_idx,e_end:,:]!=0].flatten(),density=True,bins=30,alpha=0.8,color='orangered',label='trained i to out')

    ax[1].set_title('output weights',fontname='Ubuntu')

    # recurrent layer e units
    ax[2].hist(all_w_rec[:,0,:e_end,:e_end][all_w_rec[:,0,:e_end,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.4,color='slateblue',label='naive e to e')
    ax[2].hist(all_w_rec[:,end_idx,:e_end,:e_end][all_w_rec[:,end_idx,:e_end,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.8,color='slateblue',label='trained e to e')
    ax[2].hist(all_w_rec[:,0,:e_end,e_end:][all_w_rec[:,0,:e_end,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.4,color='mediumseagreen',label='naive e to i')
    ax[2].hist(all_w_rec[:,end_idx,:e_end,e_end:][all_w_rec[:,end_idx,:e_end,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.8,color='mediumseagreen',label='trained e to i')
    ax[2].set_title('recurrent e weights',fontname='Ubuntu')

    # recurrent layer i units
    ax[3].hist(all_w_rec[:,0,e_end:,:e_end][all_w_rec[:,0,e_end:,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.4,color='darkorange',label='naive i to e')
    ax[3].hist(all_w_rec[:,end_idx,e_end:,:e_end][all_w_rec[:,end_idx,e_end:,:e_end]!=0].flatten(),density=True,bins=30,alpha=0.8,color='darkorange',label='trained i to e')
    ax[3].hist(all_w_rec[:,0,e_end:,e_end:][all_w_rec[:,0,e_end:,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.4,color='orangered',label='naive i to i')
    ax[3].hist(all_w_rec[:,end_idx,e_end:,e_end:][all_w_rec[:,end_idx,e_end:,e_end:]!=0].flatten(),density=True,bins=30,alpha=0.8,color='orangered',label='trained i to i')

    ax[3].set_title('recurrent i weights',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylabel('count',fontname='Ubuntu')
        ax[j].set_xlabel('weights',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    save_fname = spath+'/naive_trained_w_dists_test.png'

    plt.subplots_adjust(hspace=0.6,wspace=0.6)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()


def dists_of_all_rates(dual_exp_dir=save_inz_dirs,exp_season='spring'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

    for exp_string in dual_exp_dir:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    rec_0_e_rates = []
    rec_0_i_rates = []
    rec_1_i_rates = []
    rec_0_i_rates = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        coh0_exp_rates = []
        coh1_exp_rates = []
        # eventually sized 100 x units

        # rates over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            spikes = data['spikes'][0]
            # shaped [100 batches x 30 trials x 4080 timesteps x 300 units]
            true_y = data['true_y'][0] # shaped [100 batches x 30 trials x 4080 timesteps]
            for i in range(0,np.shape(true_y)[0]): # for batch
                for j in range(0,np.shape(true_y)[1]): # for trial
                    coh0_idx = np.where(true_y[i][j]==0)[0]
                    coh1_idx = np.where(true_y[i][j]==1)[0]
                    if len(coh0_idx)>0:
                        if not 'coh0_trial_rates' in locals():
                            coh0_trial_rates = np.average(spikes[i][j][coh0_idx],0) # average across time, not units (yet)
                        else:
                            coh0_trial_rates = np.vstack([coh0_trial_rates,np.average(spikes[i][j][coh0_idx],0)]) # stack trials, mean across 4080 timesteps, but preserve units

                    if len(coh1_idx)>0:
                        if not 'coh1_trial_rates' in locals():
                            coh1_trial_rates = np.average(spikes[i][j][coh1_idx],0)
                        else:
                            coh1_trial_rates = np.vstack([coh1_trial_rates,np.average(spikes[i][j][coh1_idx],0)])

            # average across coherence level trials in this file
            coh0_exp_rates.append(np.mean(coh0_trial_rates,0)) # mean across trials, but preserve units; single vector of 300 per file (100 files)
            coh1_exp_rates.append(np.mean(coh1_trial_rates,0))

        rec_0_e_rates.append(coh0_exp_rates[:,:e_end])
        rec_0_i_rates.append(coh0_exp_rates[:,e_end:])
        rec_1_e_rates.append(coh1_exp_rates[:,:e_end])
        rec_1_i_rates.append(coh1_exp_rates[:,e_end:])

    fig, ax = plt.subplots(nrows=2,ncols=1)
    ax = ax.flatten()

    # plot rates over time
    epochs=np.arange(0,np.shape(rec_0_e_rates)[1])
    ax[0].plot(epochs,np.mean(rec_0_e_rates,(0,2)),label='e units',color='slateblue')
    ax[0].fill_between(epochs,np.mean(rec_0_e_rates,(0,2))-np.std(rec_0_e_rates,(0,2)),np.mean(rec_0_e_rates,(0,2))+np.std(rec_0_e_rates,(0,2)),facecolor='slateblue',alpha=0.4)
    ax[0].plot(epochs,np.mean(rec_0_i_rates,(0,2)),label='i units',color='orangered')
    ax[0].fill_between(epochs,np.mean(rec_0_i_rates,(0,2))-np.std(rec_0_i_rates,(0,2)),np.mean(rec_0_i_rates,(0,2))+np.std(rec_0_i_rates,(0,2)),facecolor='orangered',alpha=0.4)
    ax[0].set_title('rates to coherence 0 trials',fontname='Ubuntu')

    ax[1].plot(epochs,np.mean(rec_1_e_rates,(0,2)),label='e units',color='slateblue')
    ax[1].fill_between(epochs,np.mean(rec_1_e_rates,(0,2))-np.std(rec_1_e_rates,(0,2)),np.mean(rec_1_e_rates,(0,2))+np.std(rec_1_e_rates,(0,2)),facecolor='slateblue',alpha=0.4)
    ax[1].plot(epochs,np.mean(rec_1_i_rates,(0,2)),label='i units',color='orangered')
    ax[1].fill_between(epochs,np.mean(rec_1_i_rates,(0,2))-np.std(rec_1_i_rates,(0,2)),np.mean(rec_1_i_rates,(0,2))+np.std(rec_1_i_rates,(0,2)),facecolor='orangered',alpha=0.4)
    ax[1].set_title('rates to coherence 1 trials',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylabel('average rates',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Evolution of rates over training',fontname='Ubuntu')
    plt.subplots_adjust(wspace=0.9, hspace=0.9)
    plt.draw()

    save_fname = spath+'/rates_over_training_test.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

    # plot distributions of naive and trained rates

    rec_0_e_rates[:,0,:e_end]
    rec_1_e_rates[:,0,:e_end]
    rec_0_i_rates[:,0,e_end:]
    rec_1_i_rates[:,0,e_end:]

    fig, ax = plt.subplots(nrows=1,ncols=2)
    ax = ax.flatten()

    end_idx = np.shape(rec_1_e_rates)[1]-1

    ax[0].hist(rec_1_e_rates[:,0,:e_end].flatten(),bins=20,density=True,label='e in response to 1',alpha=0.5,color='slateblue')
    ax[0].hist(rec_0_e_rates[:,0,:e_end].flatten(),bins=20,density=True,label='e in response to 0',alpha=0.5,color='mediumseagreen')
    ax[0].hist(rec_1_i_rates[:,0,e_end:].flatten(),bins=20,density=True,label='i in response to 1',alpha=0.5,color='darkorange')
    ax[0].hist(rec_0_i_rates[:,0,e_end:].flatten(),bins=20,density=True,label='i in response to 0',alpha=0.5,color='orangered')
    ax[0].set_title('naive',fontname='Ubuntu')

    ax[0].hist(rec_1_e_rates[:,end_idx,:e_end].flatten(),bins=20,density=True,label='e in response to 1',alpha=0.5,color='slateblue')
    ax[0].hist(rec_0_e_rates[:,end_idx,:e_end].flatten(),bins=20,density=True,label='e in response to 0',alpha=0.5,color='mediumseagreen')
    ax[0].hist(rec_1_i_rates[:,end_idx,e_end:].flatten(),bins=20,density=True,label='i in response to 1',alpha=0.5,color='darkorange')
    ax[0].hist(rec_0_i_rates[:,end_idx,e_end:].flatten(),bins=20,density=True,label='i in response to 0',alpha=0.5,color='orangered')
    ax[0].set_title('trained',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylabel('average rates',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Distribution of rates',fontname='Ubuntu')
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.draw()

    save_fname = spath+'/naive_trained_rates_dist_test.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

"""

def dists_of_all_synch(exp_dir=spec_input_dirs,exp_season='winter'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

def all_losses_over_training(exp_dir=spec_nointoout_dirs,exp_season='spring'):
    # PRETTY MUCH ALREADY DONE
    # plot mean with shading over time course of training for task and rate loss
    # have a better way to describe this eventually
    # split into change and no-change trials to describe performance? that might be worthwhile
"""

# SEE IF YOU CAN COMPLETE ALL THE BELOW TODAY
# ONE PER HOUR, SUPER DOABLE

def plot_in_v_out_strength(exp_dirs=all_spring_dual_dirs,exp_season='spring'):

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    fig,ax=plt.subplots(nrows=2, ncols=2)

    experiments = get_experiments(data_dir, experiment_string)
    for xdir in experiments:
        # pool together for all experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if not os.path.isfile(os.path.join(np_dir, "991-1000.npz")):
            continue

        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        early_data = np.load(os.path.join(np_dir, "41-50.npz"))
        late_data = np.load(os.path.join(np_dir, "241-250.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        naive_in = naive_data['tv0.postweights'][0]
        early_in = early_data['tv0.postweights'][0]
        late_in = late_data['tv0.postweights'][0]
        trained_in = trained_data['tv0.postweights'][0]

        naive_out = naive_data['tv2.postweights'][0]
        early_out = early_data['tv2.postweights'][0]
        late_out = late_data['tv2.postweights'][0]
        trained_out = trained_data['tv2.postweights'][0]

        # sum inputs for each unit

        ax[0,0].scatter(np.sum(naive_in[:,e_end:],0),np.abs(naive_out[e_end:]).flatten(),s=2,color='gold')
        ax[0,1].scatter(np.sum(early_in[:,e_end:],0),np.abs(early_out[e_end:]).flatten(),s=2,color='darkorange')
        ax[1,0].scatter(np.sum(late_in[:,e_end:],0),np.abs(late_out[e_end:]).flatten(),s=2,color='orangered')
        ax[1,1].scatter(np.sum(trained_in[:,e_end:],0),np.abs(trained_out[e_end:]).flatten(),s=2,color='crimson')

    # Label and title
    ax[0,0].set_title('epoch 0',fontname='Ubuntu')
    ax[0,1].set_title('epoch 5',fontname='Ubuntu')
    ax[1,0].set_title('epoch 25',fontname='Ubuntu')
    ax[1,1].set_title('epoch 100',fontname='Ubuntu')

    ax = ax.flatten()
    for j in range(0,len(ax)):
        ax[j].set_ylabel('absolute output weights',fontname='Ubuntu')
        ax[j].set_xlabel('sum input weights',fontname='Ubuntu')
        ax[j].set_xlim(-4,10)
        #ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle("Evolution of input vs output weights per inhibitory neuron")

    # Draw and save
    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.55)
    save_fname = spath+'/input_v_output_all_quad.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

def rates_over_training(exp_dirs=save_inz_dirs,exp_season='spring'):

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    rec_0_e_rates = []
    rec_0_i_rates = []
    rec_1_e_rates = []
    rec_1_i_rates = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        coh0_exp_rates = []
        coh1_exp_rates = []
        # eventually sized 100 x units

        if not '06.03.22' in np_dir: # do not include that one awful rate experiment

            # rates over time
            for filename in data_files:
                filepath = os.path.join(data_dir, xdir, "npz-data", filename)
                data = np.load(filepath)
                # too much to do anything more than the first (0th) batch in each file
                spikes = data['spikes'][0]
                # shaped [100 batches x 30 trials x 4080 timesteps x 300 units]
                true_y = data['true_y'][0] # shaped [100 batches x 30 trials x 4080 timesteps]
                for i in range(0,np.shape(true_y)[0]): # for each of 30 trials
                    coh0_idx = np.where(true_y[i]==0)[0]
                    coh1_idx = np.where(true_y[i]==1)[0]
                    if len(coh0_idx)>0:
                        if not 'coh0_trial_rates' in locals():
                            coh0_trial_rates = np.average(spikes[i][coh0_idx],0) # average across time, not units (yet)
                        else:
                            coh0_trial_rates = np.vstack([coh0_trial_rates,np.average(spikes[i][coh0_idx],0)]) # stack trials, mean across 4080 timesteps, but preserve units

                    if len(coh1_idx)>0:
                        if not 'coh1_trial_rates' in locals():
                            coh1_trial_rates = np.average(spikes[i][coh1_idx],0)
                        else:
                            coh1_trial_rates = np.vstack([coh1_trial_rates,np.average(spikes[i][coh1_idx],0)])

                # average across coherence level trials in this file
                coh0_exp_rates.append(np.mean(coh0_trial_rates,0)) # mean across trials, but preserve units; single vector of 300 per file (100 files)
                coh1_exp_rates.append(np.mean(coh1_trial_rates,0))

        rec_0_e_rates.append(coh0_exp_rates[:][:e_end])
        rec_0_i_rates.append(coh0_exp_rates[:][e_end:])
        rec_1_e_rates.append(coh1_exp_rates[:][:e_end])
        rec_1_i_rates.append(coh1_exp_rates[:][e_end:])

    return [rec_0_e_rates,rec_0_i_rates,rec_1_e_rates,rec_1_i_rates]

    # spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    # rates_0 = np.vstack([rec_0_e_rates[1:16],rec_0_e_rates[17:]])
    # rates_1 = np.vstack([rec_1_e_rates[1:16],rec_1_e_rates[17:]])
    # np.savez(spath+'/rec_rates.npz',rates_0=rates_0,rates_1=rates_1)
    # data = np.load(spath+'/rec_rates.npz')


def plot_single_exp_rate_over_training(exp_dirs=all_spring_dual_dirs,exp_season='spring'):
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    collect_rates_e_0 = []
    collect_rates_e_1 = []
    collect_rates_i_0 = []
    collect_rates_i_1 = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        rates_0 = []
        rates_1 = []

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            # too much to do anything more than the first (0th) batch in each file
            spikes = data['spikes'][0]
            # shaped [100 batches x 30 trials x 4080 timesteps x 300 units]
            true_y = data['true_y'][0] # shaped [100 batches x 30 trials x 4080 timesteps]
            for i in range(0,len(true_y)): # for each of 30 trials
                if true_y[i][0]==true_y[i][seq_len-1]: # no change in this trial
                    trial_spikes = np.transpose(spikes[i]) # now in shape units x 4080 timesteps
                    # average over time for each unit
                    trial_rates = np.mean(trial_spikes,1)

                    if true_y[i][0]==1:
                        if not 'batch_rates_1' in locals():
                            batch_rates_1 = trial_rates
                        else:
                            batch_rates_1 = np.vstack([batch_rates_1,trial_rates])
                    else:
                        if not 'batch_rates_0' in locals():
                            batch_rates_0 = trial_rates
                        else:
                            batch_rates_0 = np.vstack([batch_rates_0,trial_rates])


            rates_1.append(batch_rates_1)
            rates_0.append(batch_rates_0)
            del batch_rates_0
            del batch_rates_1
            # ultimately this would stack us up at 100 epochs, variable # trials, 300 units

        #return [rates_0,rates_1]

        # because of the variable number of trials per epoch, you may simply have to loop through
        # and manually calculate the mean and the std for plotting
        # that's not the worst thing
        e_0_means = []
        e_0_stds = []
        e_1_means = []
        e_1_stds = []
        i_0_means = []
        i_0_stds = []
        i_1_means = []
        i_1_stds = []

        epochs = np.arange(0,len(rates_0))

        for i in range(0,len(rates_0)): # for each of 100 epochs
            # meaning across units and across batches within that time slot
            e_0_means.append(np.mean(rates_0[i][:,:e_end]))
            e_0_stds.append(np.std(rates_0[i][:,:e_end]))
            e_1_means.append(np.mean(rates_1[i][:,:e_end]))
            e_1_stds.append(np.std(rates_1[i][:,:e_end]))

            i_0_means.append(np.mean(rates_0[i][:,e_end:]))
            i_0_stds.append(np.std(rates_0[i][:,e_end:]))
            i_1_means.append(np.mean(rates_1[i][:,e_end:]))
            i_1_stds.append(np.std(rates_1[i][:,e_end:]))

        e_0_means = np.array(e_0_means)
        e_0_stds = np.array(e_0_stds)
        e_1_means = np.array(e_1_means)
        e_1_stds = np.array(e_1_stds)
        i_0_means = np.array(i_0_means)
        i_0_stds = np.array(i_0_stds)
        i_1_means = np.array(i_1_means)
        i_1_stds = np.array(i_1_stds)

        fig,ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(epochs,e_0_means,label='e units',color='dodgerblue')
        ax[0].fill_between(epochs,e_0_means-e_0_stds,e_0_means+e_0_stds,facecolor='dodgerblue',alpha=0.4)
        ax[0].plot(epochs,i_0_means,label='i units',color='orangered')
        ax[0].fill_between(epochs,i_0_means-i_0_stds,i_0_means+i_0_stds,facecolor='darkorange',alpha=0.4)
        ax[0].suptitle('coherence 0')

        ax[1].plot(epochs,e_1_means,label='e units',color='dodgerblue')
        ax[1].fill_between(epochs,e_1_means-e_1_stds,e_1_means+e_1_stds,facecolor='dodgerblue',alpha=0.4)
        ax[1].plot(epochs,i_1_means,label='i units',color='orangered')
        ax[1].fill_between(epochs,i_1_means-i_1_stds,i_1_means+i_1_stds,facecolor='darkorange',alpha=0.4)
        ax[1].set_title('coherence 1')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
            ax[j].set_xlabel('training epoch',fontname='Ubuntu')
            ax[j].legend(prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Evolution of rates over training',fontname='Ubuntu')
        plt.subplots_adjust(wspace=0.9, hspace=0.7)
        plt.draw()

        save_fname = spath+'/rates_over_training_'+exp_path+'.png'
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()

        collect_rates_e_0.append(e_0_means)
        collect_rates_i_0.append(i_0_means)
        collect_rates_e_1.append(e_1_means)
        collect_rates_i_1.append(i_1_means)

    np.savez(spath+'/collected_rates.npz',collect_rates_e_0=collect_rates_e_0,collect_rates_e_1=collect_rates_e_1,collect_rates_i_0=collect_rates_i_0,collect_rates_i_1=collect_rates_i_1)

    return [collect_rates_e_0,collect_rates_e_1,collect_rates_i_0,collect_rates_i_1] # each shaped [18 experiments x 100 epochs]

    # should probably make a violin plot of e naive vs trained and another of i naive vs trained for the two coherence levels.
    # that's why you are collecting the rates, okay.

    """

                coh0_idx = np.where(true_y[i]==0)[0]
                coh1_idx = np.where(true_y[i]==1)[0]
                if len(coh0_idx)>0:
                    if not 'coh0_trial_rates' in locals():
                        coh0_trial_rates = np.average(spikes[i][coh0_idx],0) # average across time, not units (yet)
                    else:
                        coh0_trial_rates = np.vstack([coh0_trial_rates,np.average(spikes[i][coh0_idx],0)]) # stack trials, mean across 4080 timesteps, but preserve units

                if len(coh1_idx)>0:
                    if not 'coh1_trial_rates' in locals():
                        coh1_trial_rates = np.average(spikes[i][coh1_idx],0)
                    else:
                        coh1_trial_rates = np.vstack([coh1_trial_rates,np.average(spikes[i][coh1_idx],0)])

            # average across coherence level trials in this file
            coh0_exp_rates.append(np.mean(coh0_trial_rates,0)) # mean across trials, but preserve units; single vector of 300 per file (100 files)
            coh1_exp_rates.append(np.mean(coh1_trial_rates,0))
    """

def plot_collected_rates(exp_season='spring'):
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    data = np.load(spath+'/collected_rates.npz')
    collect_rates_e_0 = data['collect_rates_e_0']
    collect_rates_i_0 = data['collect_rates_i_0']
    collect_rates_e_1 = data['collect_rates_e_1']
    collect_rates_i_1 = data['collect_rates_i_1']

    fig,ax = plt.subplots(nrows=2, ncols=1)
    epochs = np.arange(0,np.shape(collect_rates_e_0)[1])
    ax[0].plot(epochs,np.mean(collect_rates_e_0,0),label='e units',color='blue')
    ax[0].fill_between(epochs,np.mean(collect_rates_e_0,0)-np.std(collect_rates_e_0,0),np.mean(collect_rates_e_0,0)+np.std(collect_rates_e_0,0),facecolor='dodgerblue',alpha=0.4)
    ax[0].plot(epochs,np.mean(collect_rates_i_0,0),label='i units',color='orangered')
    ax[0].fill_between(epochs,np.mean(collect_rates_i_0,0)-np.std(collect_rates_i_0,0),np.mean(collect_rates_i_0,0)+np.std(collect_rates_i_0,0),facecolor='darkorange',alpha=0.4)

    ax[1].plot(epochs,np.mean(collect_rates_e_1,0),label='e units',color='blue')
    ax[1].fill_between(epochs,np.mean(collect_rates_e_1,0)-np.std(collect_rates_e_1,0),np.mean(collect_rates_e_1,0)+np.std(collect_rates_e_1,0),facecolor='dodgerblue',alpha=0.4)
    ax[1].plot(epochs,np.mean(collect_rates_i_1,0),label='i units',color='orangered')
    ax[1].fill_between(epochs,np.mean(collect_rates_i_1,0)-np.std(collect_rates_i_1,0),np.mean(collect_rates_i_1,0)+np.std(collect_rates_i_1,0),facecolor='darkorange',alpha=0.4)

    for j in range(0,len(ax)):
        ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Evolution of rates over training',fontname='Ubuntu')
    plt.subplots_adjust(wspace=0.9, hspace=0.9)
    plt.draw()

    save_fname = spath+'/rates_over_training_all.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

def plot_naive_trained_rate_violins(exp_season='spring'):

    [rates_e_0,rates_e_1,rates_i_0,rates_i_1] = get_truly_naive_rates(exp_dirs=all_spring_dual_dirs,exp_season='spring',naive=True)
    naive_rates = [rates_e_0,rates_e_1,rates_i_0,rates_i_1]

    [rates_e_0,rates_e_1,rates_i_0,rates_i_1] = get_truly_naive_rates(exp_dirs=all_spring_dual_dirs,exp_season='spring',naive=False)
    trained_rates = [rates_e_0,rates_e_1,rates_i_0,rates_i_1]

    # make violin plots of naive vs trained rates
    fig,ax=plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    # plot naive vs trained for e
    for j in range(0,len(ax)):
        vplot = ax[j].violinplot(dataset=[naive_rates[j],trained_rates[j]],showmeans=True)
        for i, pc in enumerate(vplot["bodies"], 1):
            if i%2 != 0: # naive
                if j==0: # e0, e1, i0, i1
                    pc.set_facecolor('mediumseagreen')
                if j==1:
                    pc.set_facecolor('dodgerblue')
                if j==2:
                    pc.set_facecolor('darkorange')
                if j==3:
                    pc.set_facecolor('orangered')
                pc.set_alpha(0.4)
            else: # trained
                if j==0:
                    pc.set_facecolor('mediumseagreen')
                if j==1:
                    pc.set_facecolor('dodgerblue')
                if j==2:
                    pc.set_facecolor('darkorange')
                if j==3:
                    pc.set_facecolor('orangered')
                pc.set_alpha(0.7)
            if j>1:
                pc.set_edgecolor('darkorchid')
            else:
                pc.set_edgecolor('slateblue')
    ax[0].set_title('e responses to coherence 0',fontname='Ubuntu')
    ax[1].set_title('e responses to coherence 1',fontname='Ubuntu')
    ax[2].set_title('i responses to coherence 0',fontname='Ubuntu')
    ax[3].set_title('i responses to coherence 1',fontname='Ubuntu')

    plt.suptitle('SNN Activity by Coherence Label',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
        ax[j].set_ylim(-0.005,0.07)
        ax[j].set_xlabel('naive           trained',fontname='Ubuntu')
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        #for tick in ax[j].get_yticklabels():
            #tick.set_fontname("Ubuntu")

    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    save_fname = spath+'/naive_trained_rates_violin.png'

    plt.subplots_adjust(hspace=0.7,wspace=0.7)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

def plot_rates_over_training(exp_season='spring'):

    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    data = np.load(spath+'/rec_rates.npz')
    rates_0 = data['rates_0']
    rates_1 = data['rates_1']

    # just work with what you have
    # each entry of rates_0 and rates_1 is shaped [100,300]
    e_rates_0 = rates_0[:,:,:e_end]
    e_rates_1 = rates_1[:,:,:e_end]
    i_rates_0 = rates_0[:,:,e_end:]
    i_rates_1 = rates_1[:,:,e_end:]

    epochs=np.arange(0,np.shape(e_rates_0)[1])

    # plot separate examples per experiment, since it's too jumbled all together
    for i in range(0,np.shape(e_rates_0)[0]):
        # plot rates over time
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(epochs,np.mean(e_rates_0[i],1),label='e units',color='blue')
        ax[0].fill_between(epochs,np.mean(e_rates_0[i],1)-np.std(e_rates_0[i],1),np.mean(e_rates_0[i],1)+np.std(e_rates_0[i],1),facecolor='dodgerblue',alpha=0.4)
        ax[0].plot(epochs,np.mean(i_rates_0[i],1),label='i units',color='orangered')
        ax[0].fill_between(epochs,np.mean(i_rates_0[i],1)-np.std(i_rates_0[i],1),np.mean(i_rates_0[i],1)+np.std(i_rates_0[i],1),facecolor='darkorange',alpha=0.4)

        ax[1].plot(epochs,np.mean(e_rates_1[i],1),label='e units',color='blue')
        ax[1].fill_between(epochs,np.mean(e_rates_1[i],1)-np.std(e_rates_1[i],1),np.mean(e_rates_1[i],1)+np.std(e_rates_1[i],1),facecolor='dodgerblue',alpha=0.4)
        ax[1].plot(epochs,np.mean(i_rates_1[i],1),label='i units',color='orangered')
        ax[1].fill_between(epochs,np.mean(i_rates_1[i],1)-np.std(i_rates_1[i],1),np.mean(i_rates_1[i],1)+np.std(i_rates_1[i],1),facecolor='darkorange',alpha=0.4)

        for j in range(0,len(ax)):
            ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
            ax[j].set_xlabel('training epoch',fontname='Ubuntu')
            ax[j].legend(prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Evolution of rates over training',fontname='Ubuntu')
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        plt.draw()

        save_fname = spath+'/rates_over_training_exp'+str(i)+'.png'
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()

    #return [e_rates_0, i_rates_0, e_rates_1, i_rates_1]

    """
    # do the statistics you want as well
    naive_e_0 = np.mean(e_rates_0,(0,2))[0]
    naive_i_0 = np.mean(i_rates_0,(0,2))[0]
    naive_e_1 = np.mean(e_rates_1,(0,2))[0]
    naive_i_1 = np.mean(i_rates_1,(0,2))[0]

    trained_e_0 = np.mean(e_rates_0,(0,2))[99]
    trained_i_0 = np.mean(i_rates_0,(0,2))[99]
    trained_e_1 = np.mean(e_rates_1,(0,2))[99]
    trained_i_1 = np.mean(i_rates_1,(0,2))[99]

    # swap out the above to do std

    # naive coherence response comparisons
    [D,p] = scipy.stats.kstest(e_rates_0[:,0,:].flatten(),e_rates_1[:,0,:].flatten())
    [D,p] = scipy.stats.kstest(i_rates_0[:,0,:].flatten(),i_rates_1[:,0,:].flatten())
    # trained coherence response comparisons
    [D,p] = scipy.stats.kstest(e_rates_0[:,99,:].flatten(),e_rates_1[:,99,:].flatten())
    [D,p] = scipy.stats.kstest(i_rates_0[:,99,:].flatten(),i_rates_1[:,99,:].flatten())

    """

def get_truly_naive_rates(exp_dirs=all_spring_dual_dirs,exp_season='spring',naive=True):
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    rates_e_0 = []
    rates_e_1 = []
    rates_i_0 = []
    rates_i_1 = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if not '06.03.22' in np_dir: # do not include that one awful experiment
            exp_path = xdir[-9:-1]

            if naive:
                filepath = os.path.join(data_dir,xdir,"npz-data","1-10.npz")
                batch_id = 0
            else:
                filepath = os.path.join(data_dir,xdir,"npz-data","991-1000.npz")
                batch_id = 99
            data = np.load(filepath)
            spikes = data['spikes'][batch_id] # first batch ever
            true_y = data['true_y'][batch_id]

            for i in range(0,len(true_y)):
                if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change trial
                    coh_lvl = true_y[i][0]
                    trial_spikes = np.transpose(spikes[i])
                    trial_e_rates = np.mean(trial_spikes[:e_end,:])
                    trial_i_rates = np.mean(trial_spikes[e_end:,:])
                    if coh_lvl == 0:
                        rates_e_0.append(trial_e_rates)
                        rates_i_0.append(trial_i_rates)
                    else:
                        rates_e_1.append(trial_e_rates)
                        rates_i_1.append(trial_i_rates)

    return [rates_e_0,rates_e_1,rates_i_0,rates_i_1]

def get_all_final_losses(exp_dirs=save_inz_dirs_rate,exp_season='spring'):
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    # collectors
    task_loss = []
    rate_loss = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if not '06.03.22' in np_dir: # do not include that one awful experiment
            exp_path = xdir[-9:-1]

            filepath = os.path.join(data_dir, xdir, "npz-data", "991-1000.npz")
            data = np.load(filepath)
            rate_loss.append(data['step_rate_loss'][0])
            task_loss.append(data['step_task_loss'][0])

    return [rate_loss, task_loss]


def losses_over_training(exp_dirs=all_spring_dual_dirs,exp_season='spring'):
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        if not '06.03.22' in np_dir: # do not include that one awful experiment
            exp_path = xdir[-9:-1]

            #task_loss = []
            #rate_loss = []

            for filename in data_files:
                filepath = os.path.join(data_dir, xdir, "npz-data", filename)
                data = np.load(filepath)
                # mean across epochs/batches in this file
                #rate_loss.append(np.mean(data['step_rate_loss']))
                #task_loss.append(np.mean(data['step_task_loss']))
                # modify to be single experiment
                rate_loss = data['step_rate_loss']
                task_loss = data['step_task_loss']

                # concat losses together
                if not 'rate_losses' in locals():
                    rate_losses = np.transpose(rate_loss)
                    task_losses = np.transpose(task_loss)
                else:
                    rate_losses = np.vstack([rate_losses,np.transpose(rate_loss)])
                    task_losses = np.vstack([task_losses,np.transpose(task_loss)])

                print(np.shape(rate_losses))

            fig, ax = plt.subplots(nrows=2, ncols=1)
            epochs=np.arange(0,len(rate_losses))
            #ax[0].plot(epochs,task_loss,label='task loss',color='teal')
            #ax[0].plot(epochs,rate_loss,label='rate loss',color='blueviolet')
            ax[0].plot(epochs,np.mean(task_losses,1),label='task loss',color='orangered')
            ax[0].fill_between(epochs,np.mean(task_losses,1)-np.std(task_losses,1),np.mean(task_losses,1)+np.std(task_losses,1),facecolor='orangered',alpha=0.4)
            ax[0].plot(epochs,np.mean(rate_losses,1),label='rate loss',color='darkorange')
            ax[0].fill_between(epochs,np.mean(rate_losses,1)-np.std(rate_losses,1),np.mean(rate_losses,1)+np.std(rate_losses,1),facecolor='darkorange',alpha=0.4)

            for j in range(0,len(ax)):
                ax[j].set_ylabel('loss',fontname='Ubuntu')
                ax[j].set_xlabel('training epoch',fontname='Ubuntu')
                ax[j].legend(prop={"family":"Ubuntu"})
                ax[j].set_ylim(0.0,0.6)
                for tick in ax[j].get_xticklabels():
                    tick.set_fontname("Ubuntu")
                for tick in ax[j].get_yticklabels():
                    tick.set_fontname("Ubuntu")
            plt.suptitle('Example evolution of loss over task-and-rate training',fontname='Ubuntu')
            plt.subplots_adjust(wspace=0.7, hspace=0.5)
            plt.draw()

            save_fname = spath+'/losses_over_training_'+exp_path+'.png'
            plt.savefig(save_fname,dpi=300)
            # Teardown
            plt.clf()
            plt.close()

            del rate_losses
            del task_losses


def demo_input_spikes_output(exp_dirs=all_save_inz_dirs,exp_season='spring'):
    # randomly go thru and try to pick a good naive and good trained example for display
    # make the colors aesthetic, please
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'

    # custom colormaps for e and i spike rasters
    e_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","white"])
    i_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","white"])

    for xdir in exp_data_dirs: # loop through experiments
        if '15.34.00' in xdir: # choose arbitrary experiment for now
        #if not '06.03.22' in xdir: # do not include that one awful rate experiment
            exp_path = xdir[-9:-1]
            xpath = spath + '/' + exp_path
            if not os.path.isdir(xpath):
                os.makedirs(xpath)

            np_dir = os.path.join(data_dir, xdir, "npz-data")

            """
            # load naive
            data = np.load(np_dir+'/1-10.npz')
            true_y = data['true_y'][0]
            #for i in range(0,len(true_y)): # just do the first few for now
            for i in range(0,len(true_y)):
                if true_y[i][0]!=true_y[i][seq_len-1]: # i is a change trial
                    pred_y = data['pred_y'][0][i]
                    spikes = data['spikes'][0][i]
                    in_spikes = data['inputs'][0][i]
                    diffs = np.diff(true_y[i],axis=0)
                    # t_change is the first timestep of the new coherence level
                    t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1

                    # plot input spikes, recurrent spikes, output overlaid with target
                    fig, ax = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [1, 2]}) #,gridspec_kw={'height_ratios': [1, 12, 3, 6]},figsize=(8,10))

                    spike_data = np.transpose(in_spikes)
                    in_spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            in_spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            in_spike_times.append([])

                    ax[0].eventplot(in_spike_times,colors='silver')
                    ax[0].vlines(t_change,ymin=0,ymax=16,color='red',label='t change')
                    ax[0].set_ylabel('inputs',fontname='Ubuntu')
                    ax[0].set_title('input spikes',fontname='Ubuntu')

                    ax[1].plot(pred_y,color='mediumseagreen',alpha=0.6,label='output')
                    ax[1].plot(true_y[i],color='mediumblue',alpha=0.6,label='target')
                    ax[1].vlines(t_change,ymin=np.min(true_y[i]),ymax=np.max(true_y[i]),alpha=1.0,color='red',label='t change')
                    ax[1].set_ylabel('coherence level',fontname='Ubuntu')
                    ax[1].set_title('SNN output',fontname='Ubuntu')
                    ax[1].legend(prop={"family":"Ubuntu"})

                    plt.suptitle('Example naive trial',fontname='Ubuntu')
                    for j in range(0,len(ax)):
                        ax[j].set_xlabel('time (ms)',fontname='Ubuntu')
                        for tick in ax[j].get_xticklabels():
                            tick.set_fontname("Ubuntu")
                        for tick in ax[j].get_yticklabels():
                            tick.set_fontname("Ubuntu")

                    save_fname = xpath+'/'+exp_path+'_naive_inout_trial'+str(i)+'.png'

                    plt.subplots_adjust(hspace=0.5,wspace=0.7)
                    plt.draw()
                    plt.savefig(save_fname,dpi=300)
                    # Teardown
                    plt.clf()
                    plt.close()

                    # separate figure for main e and i units
                    fig, ax = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [4, 1]})

                    #sns.heatmap(np.transpose(spikes[:,:e_end]),cmap=e_cmap,cbar=False,xticklabels=False,yticklabels=False,ax=ax[0])
                    spike_data = np.transpose(spikes[:,:e_end])
                    spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            spike_times.append([])

                    ax[0].eventplot(spike_times,colors='dodgerblue')
                    ax[0].vlines(t_change,ymin=0,ymax=240,color='red',label='t change')
                    ax[0].set_ylabel('e units',fontname='Ubuntu')
                    ax[0].set_title('excitatory SNN spikes',fontname='Ubuntu')

                    #sns.heatmap(np.transpose(spikes[:,e_end:]),cmap=i_cmap,cbar=False,xticklabels=False,yticklabels=False,ax=ax[1])

                    spike_data = np.transpose(spikes[:,e_end:])
                    spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            spike_times.append([])

                    ax[1].eventplot(spike_times,colors='darkorange')
                    ax[1].vlines(t_change,ymin=0,ymax=60,color='red',label='t change')
                    ax[1].set_ylabel('i units',fontname='Ubuntu')
                    ax[1].set_title('inhibitory SNN spikes',fontname='Ubuntu')

                    plt.suptitle('Example naive trial',fontname='Ubuntu')
                    for j in range(0,len(ax)):
                        ax[j].set_xlabel('time (ms)',fontname='Ubuntu')
                        for tick in ax[j].get_xticklabels():
                            tick.set_fontname("Ubuntu")
                        for tick in ax[j].get_yticklabels():
                            tick.set_fontname("Ubuntu")

                    save_fname = xpath+'/'+exp_path+'_naive_main_trial'+str(i)+'.png'

                    plt.subplots_adjust(hspace=0.5,wspace=0.7)
                    plt.draw()
                    plt.savefig(save_fname,dpi=300)
                    # Teardown
                    plt.clf()
                    plt.close()
            """


            # repeat for trained
            data = np.load(np_dir+'/991-1000.npz')
            true_y = data['true_y'][99]
            #for i in range(0,len(true_y)): # just do the first few for now
            for i in range(0,len(true_y)):
                if true_y[i][0]!=true_y[i][seq_len-1]: # i is a change trial
                    pred_y = data['pred_y'][99][i]
                    spikes = data['spikes'][99][i]
                    in_spikes = data['inputs'][99][i]
                    diffs = np.diff(true_y[i],axis=0)
                    # t_change is the first timestep of the new coherence level
                    t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1

                    # plot input spikes, recurrent spikes, output overlaid with target
                    fig, ax = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [1, 2]}) #,gridspec_kw={'height_ratios': [1, 12, 3, 6]},figsize=(8,10))

                    #sns.heatmap(np.transpose(in_spikes),cmap='Greys_r',cbar=False,xticklabels=False,yticklabels=False,ax=ax[0])
                    # convert to event times
                    spike_data = np.transpose(in_spikes)
                    in_spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            in_spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            in_spike_times.append([])

                    ax[0].eventplot(in_spike_times,colors='silver')
                    ax[0].vlines(t_change,ymin=0,ymax=16,color='red',label='t change')
                    ax[0].set_ylabel('inputs',fontname='Ubuntu')
                    ax[0].set_title('input spikes',fontname='Ubuntu')

                    ax[1].plot(pred_y,color='mediumseagreen',alpha=0.6,label='output')
                    ax[1].plot(true_y[i],color='mediumblue',alpha=0.6,label='target')
                    ax[1].vlines(t_change,ymin=np.min(pred_y),ymax=np.max(pred_y),alpha=1.0,color='red',label='t change')
                    ax[1].set_ylabel('coherence level',fontname='Ubuntu')
                    ax[1].set_title('SNN output',fontname='Ubuntu')
                    ax[1].legend(prop={"family":"Ubuntu"})

                    plt.suptitle('Example trained trial',fontname='Ubuntu')
                    for j in range(0,len(ax)):
                        ax[j].set_xlabel('time (ms)',fontname='Ubuntu')
                        for tick in ax[j].get_xticklabels():
                            tick.set_fontname("Ubuntu")
                        for tick in ax[j].get_yticklabels():
                            tick.set_fontname("Ubuntu")

                    save_fname = xpath+'/'+exp_path+'_trained_inout_trial'+str(i)+'.png'

                    plt.subplots_adjust(hspace=0.5,wspace=0.7)
                    plt.draw()
                    plt.savefig(save_fname,dpi=300)
                    # Teardown
                    plt.clf()
                    plt.close()


                    # separate figure for main e and i units
                    fig, ax = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [4, 1]})

                    #sns.heatmap(np.transpose(spikes[:,:e_end]),cmap=e_cmap,cbar=False,xticklabels=False,yticklabels=False,ax=ax[0])

                    spike_data = np.transpose(spikes[:,:e_end])
                    spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            spike_times.append([])

                    ax[0].eventplot(spike_times,colors='dodgerblue')
                    ax[0].vlines(t_change,ymin=0,ymax=240,color='red',label='t change')
                    ax[0].set_ylabel('e units',fontname='Ubuntu')
                    ax[0].set_title('excitatory SNN spikes',fontname='Ubuntu')

                    #sns.heatmap(np.transpose(spikes[:,e_end:]),cmap=i_cmap,cbar=False,xticklabels=False,yticklabels=False,ax=ax[1])

                    spike_data = np.transpose(spikes[:,e_end:])
                    spike_times = []
                    for j in range(0,np.shape(spike_data)[0]):
                        if len(np.argwhere(spike_data[j,:]==1))>0:
                            spike_times.append(np.concatenate(np.argwhere(spike_data[j,:]==1)).ravel().tolist())
                        else:
                            spike_times.append([])

                    ax[1].eventplot(spike_times,colors='darkorange')
                    ax[1].vlines(t_change,ymin=0,ymax=60,color='red',label='t change')
                    ax[1].set_ylabel('i units',fontname='Ubuntu')
                    ax[1].set_title('inhibitory SNN spikes',fontname='Ubuntu')

                    plt.suptitle('Example trained trial',fontname='Ubuntu')
                    for j in range(0,len(ax)):
                        ax[j].set_xlabel('time (ms)',fontname='Ubuntu')
                        for tick in ax[j].get_xticklabels():
                            tick.set_fontname("Ubuntu")
                        for tick in ax[j].get_yticklabels():
                            tick.set_fontname("Ubuntu")

                    save_fname = xpath+'/'+exp_path+'_trained_main_trial'+str(i)+'.png'

                    plt.subplots_adjust(hspace=0.5,wspace=0.7)
                    plt.draw()
                    plt.savefig(save_fname,dpi=300)
                    # Teardown
                    plt.clf()
                    plt.close()



def input_channel_violin_plots(exp_dirs=all_save_inz_dirs,exp_season='spring',fromfile=True):
    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    if not fromfile:
        # include input rates from CNN (original CNN output file)
        spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
        x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
        # determine each of the 16 channels' average rates over 600 x 4080 trials
        # separate according to coherence level!
        coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
        y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]
        y = np.squeeze(y)

        # from actual experiment now
        for exp_string in exp_dirs:
            if not 'exp_data_dirs' in locals():
                exp_data_dirs = get_experiments(data_dir, exp_string)
            else:
                exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

        # aggregate across all experiments and all trials
        data_files = filenames(num_epochs, epochs_per_file)

        for xdir in exp_data_dirs: # loop through experiments
            np_dir = os.path.join(data_dir, xdir, "npz-data")

            for filename in data_files:
                filepath = os.path.join(data_dir, xdir, "npz-data", filename)

                data = np.load(filepath)
                # simply too many if we don't just take the final batch
                in_spikes = data['inputs'][99]
                true_y = data['true_y'][99]
                if '-swaplabels' in xdir: # not unswaplabels
                    true_y = ~true_y.astype(int) + 2

                true_y = np.reshape(true_y,[np.shape(true_y)[0],seq_len])
                in_spikes = np.reshape(in_spikes,[np.shape(in_spikes)[0],seq_len,np.shape(in_spikes)[2]])

                y=np.vstack([y,true_y])
                x=np.vstack([x,in_spikes])

        # for each of ALL trials (from CNN and experimental)
        for i in range(0,np.shape(y)[0]):
        # for each of 4080 time steps
        # determine if coherence 1 or 0
            coh0_idx = np.where(y[i]==0)[0]
            coh1_idx = np.where(y[i]==1)[0]
        # take average rates across that trial's timepoints for the same coherence level and append
            if len(coh0_idx)>0:
                if not 'coh0_channel_trial_rates' in locals():
                    coh0_channel_trial_rates = np.average(x[i][coh0_idx],0)
                else:
                    coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(x[i][coh0_idx],0)])

            if len(coh1_idx)>0:
                if not 'coh1_channel_trial_rates' in locals():
                    coh1_channel_trial_rates = np.average(x[i][coh1_idx],0)
                else:
                    coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(x[i][coh1_idx],0)])

        # average across all trials for a given channel (16)
        coh0_rates = np.average(coh0_channel_trial_rates,0)
        coh1_rates = np.average(coh1_channel_trial_rates,0)

        coh0_channels = np.where(coh1_rates<coh0_rates)[0]
        coh1_channels = np.where(coh1_rates>coh0_rates)[0]

    else:
        data_path = spath+'/input_channel_trial_rates.npz'
        tuning_data = np.load(data_path)
        coh0_channels = tuning_data['coh0_channels']
        coh1_channels = tuning_data['coh1_channels']
        coh0_channel_trial_rates = tuning_data['coh0_channel_trial_rates']
        coh1_channel_trial_rates = tuning_data['coh1_channel_trial_rates']

    # make violin plots

    fig, ax = plt.subplots(nrows=2,ncols=4)

    # first do coh0_channels
    # next do coh1_channels
    # plot both coh0_channel_trial_rates' mean median std and same for coh1_channel_trial_rates in each subplot

    ax = ax.flatten()
    vplot_coh0_colors = ['YellowGreen','OliveDrab','LimeGreen','ForestGreen','MediumSeaGreen','LightSeaGreen','Teal','SteelBlue']
    vplot_coh1_colors = ['BlueViolet','Purple','MediumVioletRed','HotPink','DeepPink','Crimson','OrangeRed','DarkOrange']
    for i in range(0,len(coh0_channels)):
        vplot = ax[i].violinplot(dataset=[coh0_channel_trial_rates[:,coh0_channels[i]],coh1_channel_trial_rates[:,coh0_channels[i]]], showmeans=True)
        for i, pc in enumerate(vplot["bodies"], 1):
            if i%2 != 0: # multiple colors for coherence level we are focusing on
                pc.set_facecolor('yellowgreen')
            else: # same color for the non-preferred one
                pc.set_facecolor('teal') # the other would be pc.set_facecolor('mediumaquamarine')
            pc.set_edgecolor('dodgerblue')
            pc.set_alpha(0.6)

    plt.suptitle('Low-coherence-tuned input channels',fontname='Ubuntu')
    #labels = ['low', 'high']
    for j in range(0,len(ax)):
        ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
        #ax[j].set_xticklabels(['low','high'],fontname='Ubuntu')
        #ax[j].set_xlim(0.25, len(labels) + 0.75)
        ax[j].set_ylim(0.0,0.7)
        ax[j].set_xlabel('low  high',fontname='Ubuntu')
        ax[j].set_title('channel '+str(coh0_channels[j]+1),fontname='Ubuntu')
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        #for tick in ax[j].get_yticklabels():
            #tick.set_fontname("Ubuntu")

    save_fname = spath+'/input_channel_coh0_rates_violin.png'

    plt.subplots_adjust(hspace=0.5,wspace=1.0)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

    # do the same of coherence 1 tuned input channels
    # make sure to use the same y axis min and max as well
    fig, ax = plt.subplots(nrows=2,ncols=4)
    ax = ax.flatten()
    for i in range(0,len(coh0_channels)):
        vplot = ax[i].violinplot(dataset=[coh0_channel_trial_rates[:,coh1_channels[i]],coh1_channel_trial_rates[:,coh1_channels[i]]], showmeans=True)
        for i, pc in enumerate(vplot["bodies"], 1):
            if i%2 != 0: # multiple colors for coherence level we are focusing on
                pc.set_facecolor('dodgerblue')
            else: # same color for the non-preferred one
                pc.set_facecolor('blueviolet')
            pc.set_edgecolor('dodgerblue')
            pc.set_alpha(0.6)

    plt.suptitle('High-coherence-tuned input channels',fontname='Ubuntu')
    #labels = ['low', 'high']
    for j in range(0,len(ax)):
        ax[j].set_ylabel('rate (spikes/ms)',fontname='Ubuntu')
        #ax[j].set_xticklabels(['low','high'],fontname='Ubuntu')
        #ax[j].set_xlim(0.25, len(labels) + 0.75)
        ax[j].set_ylim(0.0,0.7)
        ax[j].set_xlabel('low  high',fontname='Ubuntu')
        ax[j].set_title('channel '+str(coh1_channels[j]+1),fontname='Ubuntu')
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        #for tick in ax[j].get_yticklabels():
            #tick.set_fontname("Ubuntu")

    save_fname = spath+'/input_channel_coh1_rates_violin.png'

    plt.subplots_adjust(hspace=0.5,wspace=1.0)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()


def dists_of_input_rates(exp_dirs=all_save_inz_dirs,exp_season='spring',make_plots=False):
    # also bring in the rate trained ones too. just anything that contains saveinz; also the original CNN outputs too

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    # from CNN (original CNN output file)
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]
    y = np.squeeze(y)

    # from actual data
    # from actual experiment now
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)

            data = np.load(filepath)
            # simply too many if we don't just take the final batch
            in_spikes = data['inputs'][99]
            true_y = data['true_y'][99]
            if '-swaplabels' in xdir: # not unswaplabels
                true_y = ~true_y.astype(int) + 2

            true_y = np.reshape(true_y,[np.shape(true_y)[0],seq_len])
            in_spikes = np.reshape(in_spikes,[np.shape(in_spikes)[0],seq_len,np.shape(in_spikes)[2]])

            y=np.vstack([y,true_y])
            x=np.vstack([x,in_spikes])

    # for each of ALL trials (from CNN and experimental)
    for i in range(0,np.shape(y)[0]):
    # for each of 4080 time steps
    # determine if coherence 1 or 0
        coh0_idx = np.where(y[i]==0)[0]
        coh1_idx = np.where(y[i]==1)[0]
    # take average rates across that trial's timepoints for the same coherence level and append
        if len(coh0_idx)>0:
            if not 'coh0_channel_trial_rates' in locals():
                coh0_channel_trial_rates = np.average(x[i][coh0_idx],0)
            else:
                coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(x[i][coh0_idx],0)])

        if len(coh1_idx)>0:
            if not 'coh1_channel_trial_rates' in locals():
                coh1_channel_trial_rates = np.average(x[i][coh1_idx],0)
            else:
                coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(x[i][coh1_idx],0)])

    # average across all trials for a given channel (16)
    coh0_rates = np.average(coh0_channel_trial_rates,0)
    coh1_rates = np.average(coh1_channel_trial_rates,0)

    coh0_channels = np.where(coh1_rates<coh0_rates)[0]
    coh1_channels = np.where(coh1_rates>coh0_rates)[0]

    # plot the distribution of rates of each channel in response to the two coherence levels
    # do so for all 16 channels together (and measure, compare their distributions)
    # do so for each channel separately and describe just how close their firing rates are to each coherence level

    if make_plots:
        # PLOT RATES OF TUNED CHANNELS TOGETHER FOR COH0 AND COH1 TRIALS
        # stack over subplots: coh0 tuned on top, coh 1 tuned on bottom
        # adjust axes to be the same

        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()

        ax[0].hist(coh0_channel_trial_rates[:,coh0_channels],density=True,bins=30,alpha=0.7,label='rates to coh 0 trials')
        ax[0].hist(coh1_channel_trial_rates[:,coh0_channels],density=True,bins=30,alpha=0.7,label='rates to coh 1 trials')
        ax[0].set_title('Coherence 0 tuned input channels',fontname='Ubuntu')

        ax[1].hist(coh0_channel_trial_rates[:,coh1_channels],density=True,bins=30,alpha=0.7,label='rates to coh 0 trials')
        ax[1].hist(coh1_channel_trial_rates[:,coh1_channels],density=True,bins=30,alpha=0.7,label='rates to coh 1 trials')
        ax[1].set_title('Coherence 1 tuned input channels',fontname='Ubuntu')

        # hopefully can visually (and numerically) see that input channels don't differ all that much in their responses
        # even though they are sliiiightly tuned

        # PLOT RATES OF ALL CHANNELS TO COH0 AND COH1 (regardless of tuning)
        ax[2].hist(coh0_channel_trial_rates,density=True,bins=30,alpha=0.7)
        ax[2].set_title("All channels' rates to coh 0 trials",fontname='Ubuntu')

        ax[3].hist(coh1_channel_trial_rates,density=True,bins=30,alpha=0.7)
        ax[3].set_title("All channels' rates to coh 1 trials",fontname='Ubuntu')

        plt.suptitle('Responses of 16 input channels to different coherences',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('density',fontname='Ubuntu')
            ax[j].set_xlabel('rates (Hz)',fontname='Ubuntu')
            if j < 2:
                ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        save_fname = spath+'/input_channel_rates.png'

        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()

    # KS TEST COMPARISONS NOW
    return [coh0_channel_trial_rates,coh1_channel_trial_rates]

    """
    # compare for within each of 16 channels
    Ds = []
    ps = []
    for i in range(0,np.shape(coh0_channel_trial_rates)[1]):
...     [D,p]=scipy.stats.kstest(coh0_channel_trial_rates[:,i],coh1_channel_trial_rates[:,i])
...     Ds.append(D)
...     ps.append(p)

    # for all together
    [D,p] = scipy.stats.kstest(coh0_channel_trial_rates.flatten(),coh1_channel_trial_rates.flatten())

    # for within coherence tunings
    [D,p] = scipy.stats.kstest(coh0_channel_trial_rates[:,coh0_channels].flatten(),coh1_channel_trial_rates[:,coh0_channels].flatten())
    # For the group that are considered tuned altogether to 1, rates to coherence 0 vs 1
    [D,p] = scipy.stats.kstest(coh0_channel_trial_rates[:,coh1_channels].flatten(),coh1_channel_trial_rates[:,coh1_channels].flatten())

    np.mean(coh0_channel_trial_rates[:,coh1_channels])
    np.mean(coh1_channel_trial_rates[:,coh1_channels])
    np.mean(coh0_channel_trial_rates[:,coh0_channels])
    np.mean(coh1_channel_trial_rates[:,coh0_channels])
    np.std(coh0_channel_trial_rates[:,coh1_channels])
    np.std(coh1_channel_trial_rates[:,coh1_channels])
    np.std(coh0_channel_trial_rates[:,coh0_channels])
    np.std(coh1_channel_trial_rates[:,coh0_channels])
"""


def get_input_tuning_from_CNN():
    spikes = load_npz('/data/datasets/CNN_outputs/spike_train_mixed_limlifetime_abs.npz')
    x = np.array(spikes.todense()).reshape((-1, seq_len, n_input))
    # determine each of the 16 channels' average rates over 600 x 4080 trials
    # separate according to coherence level!
    coherences = load_npz('/data/datasets/CNN_outputs/ch8_abs_ccd_coherences.npz')
    y = np.array(coherences.todense().reshape((-1, seq_len)))[:, :, None]

    # for each of 600 trials
    for i in range(0,np.shape(y)[0]):
    # for each of 4080 time steps
    # determine if coherence 1 or 0
        coh0_idx = np.where(y[i]==0)[0]
        coh1_idx = np.where(y[i]==1)[0]
    # take average rates across that trial's timepoints for the same coherence level and append
        if len(coh0_idx)>0:
            if not 'coh0_channel_trial_rates' in locals():
                coh0_channel_trial_rates = np.average(x[i][coh0_idx],0)
            else:
                coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(x[i][coh0_idx],0)])

        if len(coh1_idx)>0:
            if not 'coh1_channel_trial_rates' in locals():
                coh1_channel_trial_rates = np.average(x[i][coh1_idx],0)
            else:
                coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(x[i][coh1_idx],0)])

    coh0_rates = np.average(coh0_channel_trial_rates,0)
    coh1_rates = np.average(coh1_channel_trial_rates,0)
    coh1_idx = np.where(coh1_rates>coh0_rates)[0]
    coh0_idx = np.where(coh1_rates<coh0_rates)[0]

    return [coh0_idx,coh1_idx]

def get_input_tuning_single_exp(xdir):

    for filename in data_files:
        filepath = os.path.join(data_dir, xdir, "npz-data", filename)
        data = np.load(filepath)
        input_z = data['inputs']
        # shaped [100 batches x 30 trials x 4080 timesteps x 16 units]
        true_y = data['true_y'] # shaped [100 batches x 30 trials x 4080 timesteps]
        for i in range(0,np.shape(true_y)[0]): # for batch
            for j in range(0,np.shape(true_y)[1]): # for trial
                coh0_idx = np.where(true_y[i][j]==0)[0]
                coh1_idx = np.where(true_y[i][j]==1)[0]
                if len(coh0_idx)>0:
                    if not 'coh0_channel_trial_rates' in locals():
                        coh0_channel_trial_rates = np.average(input_z[i][j][coh0_idx],0)
                    else:
                        coh0_channel_trial_rates = np.vstack([coh0_channel_trial_rates,np.average(input_z[i][j][coh0_idx],0)])

                if len(coh1_idx)>0:
                    if not 'coh1_channel_trial_rates' in locals():
                        coh1_channel_trial_rates = np.average(input_z[i][j][coh1_idx],0)
                    else:
                        coh1_channel_trial_rates = np.vstack([coh1_channel_trial_rates,np.average(input_z[i][j][coh1_idx],0)])

    coh1_channel_rates = np.array(np.mean(coh1_channel_trial_rates,0))
    coh0_channel_rates = np.array(np.mean(coh0_channel_trial_rates,0))
    coh1_idx = np.where(coh1_channel_rates>coh0_channel_rates)[0]
    coh0_idx = np.where(coh1_channel_rates<coh0_channel_rates)[0]

    return [coh0_idx,coh1_idx]

def input_layer_over_training_by_coherence(dual_exp_dir=all_spring_dual_dirs,exp_season='spring'):
    # characterize the connectivity from the input layer to recurrent
    # plot over the course of training with shaded error bars
    # compare for rate- and dual-trained

    # ACTUALLY YOU NEED TO DO THIS FOR INDIVIDUAL EXPERIMENTS BECAUSE WE ARE FOLLOWING LABELS NOT ACTUAL COHERENCE

    # from actual experiment now
    for exp_string in dual_exp_dir:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    # get default input tunings from CNN outputs
    [default_coh0_idx, default_coh1_idx] = get_input_tuning_from_CNN()

    # aggregate over all experiments

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")

        coh1_e_exp = []
        coh1_i_exp = []
        coh0_e_exp = []
        coh0_i_exp = []
        epoch_task_loss_exp = []
        epoch_rate_loss_exp = []

        # check if inputs is saved; otherwise use defaults

        filepath = os.path.join(data_dir, xdir, "npz-data", '1-10.npz')
        data = np.load(filepath)
        if 'inputs' in data:
            [coh0_idx, coh1_idx] = get_input_tuning_single_exp(xdir)
        else:
            [coh0_idx, coh1_idx] = [default_coh0_idx, default_coh1_idx]

        # get the truly naive weights
        filepath = os.path.join(data_dir,xdir,"npz-data","input_preweights.npy")
        input_w = np.load(filepath)
        coh1_e_exp.append(np.mean(input_w[coh1_idx,:e_end]))
        coh1_i_exp.append(np.mean(input_w[coh1_idx,e_end:]))
        coh0_e_exp.append(np.mean(input_w[coh0_idx,:e_end]))
        coh0_i_exp.append(np.mean(input_w[coh0_idx,e_end:]))

        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_w = data['tv0.postweights'][0] # just the singular for now; too much data and noise otherwise
            epoch_task_loss_exp.append(np.mean(data['step_task_loss']))
            epoch_rate_loss_exp.append(np.mean(data['step_rate_loss']))
            #for i in range(0,np.shape(input_w)[0]): # 100 trials
            # weights of each type to e units and to i units
            coh1_e_exp.append(np.mean(input_w[coh1_idx,:e_end]))
            coh1_i_exp.append(np.mean(input_w[coh1_idx,e_end:]))
            coh0_e_exp.append(np.mean(input_w[coh0_idx,:e_end]))
            coh0_i_exp.append(np.mean(input_w[coh0_idx,e_end:]))

            # you may want to expand this and grab ALL of the weights

        if not "coh1_e" in locals():
            coh1_e = coh1_e_exp
        else:
            coh1_e = np.vstack([coh1_e, coh1_e_exp])

        if not "coh1_i" in locals():
            coh1_i = coh1_i_exp
        else:
            coh1_i = np.vstack([coh1_i, coh1_i_exp])

        if not "coh0_e" in locals():
            coh0_e = coh0_e_exp
        else:
            coh0_e = np.vstack([coh0_e, coh0_e_exp])

        if not "coh0_i" in locals():
            coh0_i = coh0_i_exp
        else:
            coh0_i = np.vstack([coh0_i, coh0_i_exp])


        if not "epoch_task_loss" in locals():
            epoch_task_loss = epoch_task_loss_exp
        else:
            epoch_task_loss = np.vstack([epoch_task_loss, epoch_task_loss_exp])

        if not "epoch_rate_loss" in locals():
            epoch_rate_loss = epoch_rate_loss_exp
        else:
            epoch_rate_loss = np.vstack([epoch_rate_loss, epoch_rate_loss_exp])

    """
    # CHARACTERIZE AND COMPARE
    # naive and trained means and stds
    # order is avg coherence 1 tuned input to e, avg coherence 0 tuned input to e, avg coherence 1 tuned input to i, avg coherence 0 tuned input to i
    naive_means = [np.mean(coh1_e[:,0],0), np.mean(coh0_e[:,0],0), np.mean(coh1_i[:,0],0), np.mean(coh0_i[:,0],0)]
    naive_stds = [np.std(coh1_e[:,0],0), np.std(coh0_e[:,0],0), np.std(coh1_i[:,0],0), np.std(coh0_i[:,0],0)]

    trained_means = [np.mean(coh1_e[:,100],0), np.mean(coh0_e[:,100],0), np.mean(coh1_i[:,100],0), np.mean(coh0_i[:,100],0)]
    trained_stds = [np.std(coh1_e[:,100],0), np.std(coh0_e[:,100],0), np.std(coh1_i[:,100],0), np.std(coh0_i[:,100],0)]

    # in naive state
    # e recipients of 0 or 1 drive
    [D,p] = scipy.stats.kstest(coh0_e[:,0],coh1_e[:,0])
    # i recipients of 0 or 1 drive
    [D,p] = scipy.stats.kstest(coh0_i[:,0],coh1_i[:,0])

    # in trained state
    # e recipients of 0 or 1 drive
    [D,p] = scipy.stats.kstest(coh0_e[:,100],coh1_e[:,100])
    # i recipients of 0 or 1 drive
    [D,p] = scipy.stats.kstest(coh0_i[:,100],coh1_i[:,100])

    # ratios
    np.mean(coh0_e[:,0])/np.mean(coh0_i[:,0])
    np.mean(coh1_e[:,0])/np.mean(coh1_i[:,0])

    np.mean(coh0_e[:,100])/np.mean(coh0_i[:,100])
    np.mean(coh1_e[:,100])/np.mean(coh1_i[:,100])

    # THIS IS WHAT IS ACTUALLY IN PLOTS:
    np.mean(coh0_e[:,0])/np.mean(coh1_e[:,0])

    """

    fig, ax = plt.subplots(nrows=3, ncols=1)

    coh1_e_mean = np.mean(coh1_e,0)
    coh1_e_std = np.std(coh1_e,0)
    coh0_e_mean = np.mean(coh0_e,0)
    coh0_e_std = np.std(coh0_e,0)

    epochs = np.shape(coh1_e)[1]

    ax[0].plot(np.arange(0,epochs),coh1_e_mean, label='1 mod inputs', color='slateblue')
    ax[0].fill_between(np.arange(0,epochs),coh1_e_mean-coh1_e_std, coh1_e_mean+coh1_e_std, alpha=0.4, facecolor='slateblue')
    ax[0].plot(np.arange(0,epochs),coh0_e_mean, label='0 mod inputs', color='mediumseagreen')
    ax[0].fill_between(np.arange(0,epochs),coh0_e_mean-coh0_e_std, coh0_e_mean+coh0_e_std, alpha=0.4, facecolor='mediumseagreen')
    ax[0].set_title('input weights to excitatory units',fontname='Ubuntu')

    coh1_i_mean = np.mean(coh1_i,0)
    coh1_i_std = np.std(coh1_i,0)
    coh0_i_mean = np.mean(coh0_i,0)
    coh0_i_std = np.std(coh0_i,0)

    ax[1].plot(np.arange(0,epochs),coh1_i_mean, label='1 mod inputs', color='darkorange')
    ax[1].fill_between(np.arange(0,epochs),coh1_i_mean-coh1_i_std, coh1_i_mean+coh1_i_std, alpha=0.4, facecolor='darkorange')
    ax[1].plot(np.arange(0,epochs),coh0_i_mean, label='0 mod inputs', color='orangered')
    ax[1].fill_between(np.arange(0,epochs),coh0_i_mean-coh0_i_std, coh0_i_mean+coh0_i_std, alpha=0.4, facecolor='orangered')
    ax[1].set_title('input weights to inhibitory units',fontname='Ubuntu')


    epochs = np.shape(epoch_task_loss)[1]

    task_mean = np.mean(epoch_task_loss,0)
    task_error = np.std(epoch_task_loss,0)
    ax[2].plot(np.arange(0,epochs),task_mean, label='task loss', color='darkorange')
    ax[2].fill_between(np.arange(0,epochs),task_mean-task_error, task_mean+task_error, alpha=0.4, facecolor='darkorange')

    rate_mean = np.mean(epoch_rate_loss,0)
    rate_error = np.std(epoch_rate_loss,0)
    ax[2].plot(np.arange(0,epochs),rate_mean, label='rate loss', color='orangered')
    ax[2].fill_between(np.arange(0,epochs),rate_mean+rate_error, rate_mean+rate_error, alpha=0.4, facecolor='orangered') #other options include edgecolor='#CC4F1B', facecolor='#FF9848'

    ax[2].set_ylabel('loss',fontname='Ubuntu')
    ax[2].set_title('losses')


    for j in range(0,len(ax)):
        if j < 2:
            ax[j].set_ylabel('average weights',fontname='Ubuntu')
        ax[j].set_xlabel('training epoch',fontname='Ubuntu')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Evolution of input weights over training',fontname='Ubuntu')
    plt.subplots_adjust(wspace=1.0, hspace=1.0)
    plt.draw()

    save_fname = spath+'/corrected_inputs_to_ei_mod_dual.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

    #return [coh1_e,coh0_e,coh1_i,coh0_i]

    """
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)

            data = np.load(filepath)
            spikes = data['spikes']
            true_y = data['true_y']

            # aggregate the mean connectivity strength from the two tuned input populations to e and i units
            # maybe it's too much to do it over more than just the last batch trial for each file
            # that's still 100 datapoints
            for i in range(0,np.shape(true_y)[0]):
                for j in range(0,np.shape(true_y)[1]):

            # ratio of weights; get naive vs. trained distributions and also see how they evolve over training too
            coh0in_to_e/coh0in_to_i
            coh1in_to_e/coh1in_to_i
            # aggregate over all experiments
    """

    # get a number distribution to quantify this, maybe for each experiment
    # the ratio between avg weight from input coh0 and coh1 to e and i recurrent units at the beginning and at the end of training
    # 0_to_e/0_to_i = 1 at beginning
    # 1_to_e/1_to_i = 1 at beginning
    # 0_to_e/0_to_i < 1 at end
    # 1_to_e/1_to_e > 1 at end
    # that's a good start


def characterize_tuned_rec_populations(exp_dirs=save_inz_dirs,exp_season='spring',mix_tuned_indices=False,plot_counts=True):
    # determine tuning of each recurrent unit across each of these experiments
    # according to trials of single coherence level only
    # include save inz as well into these spring experimental categories, okay
    # count up how many e and i units are tuned to each coherence level
    # quantify the extent they are tuned - again their relative rates to coh 0 and coh 1; plot together
    # look at naive state as well
    # [the above points to maybe additional analyses / quantifications to get later]
    # plot the rate distributions of these populations
    # compare between dual, task, rate training
    # as best you can

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    exp_data_dirs = np.unique(exp_data_dirs)

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    coh1_e_ct = []
    coh1_i_ct = []
    coh0_e_ct = []
    coh0_i_ct = []

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        """
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)"""

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))

        # do for both naive and trained (but trained first here)
        spikes = trained_data['spikes']
        true_y = trained_data['true_y']

        # find the relative tuning of all e and i units
        # find which units respond more to input of a certain coh level across batches and trials
        coh0_rec_rates = []
        coh1_rec_rates = []

        for i in range(0,np.shape(true_y)[0]): # batch
            for j in range(0,np.shape(true_y)[1]): # trial
                if true_y[i][j][0]==true_y[i][j][seq_len-1]: # no change trials only
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        ######{{{{   GET INDICES OF COH0 and COH1 TUNED UNITS   }}}}#######
        coh1_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        coh1_e_ct.append(len(coh1_rec_idx[coh1_rec_idx<e_end]))
        coh1_i_ct.append(len(coh1_rec_idx[coh1_rec_idx>=e_end]))

        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        coh0_e_ct.append(len(coh0_rec_idx[coh0_rec_idx<e_end]))
        coh0_i_ct.append(len(coh0_rec_idx[coh0_rec_idx>=e_end]))

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        ######{{{{   GET RATES OF TRAINED-TUNED-INDEXED E AND I UNITS IN TRAINED TRIALS   }}}}#######
        if not 'all_0e_to_0_rates' in locals():
            all_0e_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_0_rates = np.hstack([all_0e_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        if not 'all_0e_to_1_rates' in locals():
            all_0e_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_1_rates = np.hstack([all_0e_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_0_rates' in locals():
            all_1e_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_0_rates = np.hstack([all_1e_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_1_rates' in locals():
            all_1e_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_1_rates = np.hstack([all_1e_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        # now for the i units
        if not 'all_0i_to_0_rates' in locals():
            all_0i_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_0_rates = np.hstack([all_0i_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_0i_to_1_rates' in locals():
            all_0i_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_1_rates = np.hstack([all_0i_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_0_rates' in locals():
            all_1i_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_0_rates = np.hstack([all_1i_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_1_rates' in locals():
            all_1i_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_1_rates = np.hstack([all_1i_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        """
        # get all these e and i units' actual rates in response to trials of each coherence level
        if not 'all_coh0_e_rates' in locals():
            all_coh0_e_rates = coh0_rec_rates[:,:e_end].flatten()
        else:
            all_coh0_e_rates = np.hstack([all_coh0_e_rates,coh0_rec_rates[:,:e_end].flatten()])

        if not 'all_coh0_i_rates' in locals():
            all_coh0_i_rates = coh0_rec_rates[:,e_end:].flatten()
        else:
            all_coh0_i_rates = np.hstack([all_coh0_i_rates,coh0_rec_rates[:,e_end:].flatten()])

        if not 'all_coh1_e_rates' in locals():
            all_coh1_e_rates = coh1_rec_rates[:,:e_end].flatten()
        else:
            all_coh1_e_rates = np.hstack([all_coh1_e_rates,coh1_rec_rates[:,:e_end].flatten()])

        if not 'all_coh1_i_rates' in locals():
            all_coh1_i_rates = coh1_rec_rates[:,e_end:].flatten()
        else:
            all_coh1_i_rates = np.hstack([all_coh1_i_rates,coh1_rec_rates[:,e_end:].flatten()])"""

    trained_ct = [coh1_e_ct,coh1_i_ct,coh0_e_ct,coh0_i_ct]
    return trained_ct

    fig, ax = plt.subplots(nrows=2,ncols=2)
    ax = ax.flatten()

    """
    ax[0].hist(np.array(all_coh0_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh0_e_ct)))+' avg units)')
    ax[0].set_title('coh 0 tuned e units',fontname='Ubuntu')
    ax[1].hist(np.array(all_coh0_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh0_i_ct)))+' avg units)')
    ax[1].set_title('coh 0 tuned i units',fontname='Ubuntu')
    ax[2].hist(np.array(all_coh1_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh1_e_ct)))+' avg units)')
    ax[2].set_title('coh 1 tuned e units',fontname='Ubuntu')
    ax[3].hist(np.array(all_coh1_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='trained ('+str(int(np.mean(coh1_i_ct)))+' avg units)')
    ax[3].set_title('coh 1 tuned i units',fontname='Ubuntu')
    """

    ######{{{{   ADD THE TRAINED E AND I RATES TO SUBPLOTS   }}}}#######

    ######{{{{   SUBPLOT FOR E COHERENCE 0 TRIALS   }}}}#######
    ax[0].hist(all_0e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[0].hist(all_1e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR E COHERENCE 1 TRIALS   }}}}#######
    ax[1].hist(all_0e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[1].hist(all_1e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR I COHERENCE 0 TRIALS   }}}}#######
    ax[2].hist(all_0i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[2].hist(all_1i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')

    ######{{{{   SUBPLOT FOR I COHERENCE 1 TRIALS   }}}}#######
    ax[3].hist(all_0i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned trained')
    ax[3].hist(all_1i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned trained')


    ######{{{{   REPEAT FOR NAIVE   }}}}#######
    ######{{{{   EVENTUALLY FOR THE OTHER PLOT YOU ENVISIONED YOU WILL NEED TO RENAME AND SAVE THE BELOW FOR TRAINED   }}}}#######
    coh1_e_ct = []
    coh1_i_ct = []
    coh0_e_ct = []
    coh0_i_ct = []

    del all_0e_to_0_rates
    del all_0e_to_1_rates
    del all_1e_to_0_rates
    del all_1e_to_1_rates
    del all_0i_to_0_rates
    del all_0i_to_1_rates
    del all_1i_to_0_rates
    del all_1i_to_1_rates

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        """
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)"""

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        naive_data = np.load(os.path.join(np_dir,"1-10.npz"))

        # do for both naive and trained (but trained first here)
        spikes = naive_data['spikes']
        true_y = naive_data['true_y']

        # find the relative tuning of all e and i units
        # find which units respond more to input of a certain coh level across batches and trials
        coh0_rec_rates = []
        coh1_rec_rates = []

        ######{{{{   GET UNITWISE RATES FOR NAIVE TRIALS OF COH1 AND COH0   }}}}#######
        for i in range(0,np.shape(true_y)[0]): # batch
            for j in range(0,np.shape(true_y)[1]): # trial
                if true_y[i][j][0]==true_y[i][j][seq_len-1]: # no change trials only
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                        # these are entirely rates in response to 0 coherence level
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        ######{{{{   FIND THE INDICES OF E AND I UNITS THAT ARE TUNED IN THE NAIVE STATE   }}}}#######
        # specify that these are the naive tuned indices; we do not care about them in the comparison plots for now

        coh1_naive_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        coh0_naive_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]

        coh0_e_ct.append(len(coh0_naive_rec_idx[coh0_naive_rec_idx<e_end]))
        coh0_i_ct.append(len(coh0_naive_rec_idx[coh0_naive_rec_idx>=e_end]))

        coh1_e_ct.append(len(coh1_naive_rec_idx[coh1_naive_rec_idx<e_end]))
        coh1_i_ct.append(len(coh1_naive_rec_idx[coh1_naive_rec_idx>=e_end]))

        ######{{{{   OPTION TO EITHER USE THE TRAINED TUNING INDICES TO INDEX THE NAIVE TRIALS FOR RATE CALCULATIONS OR THE NAIVE TUNING INDICES THEMSELVES   }}}}#######
        if mix_tuned_indices:
            coh0_rec_idx = coh0_naive_rec_idx
            coh1_rec_idx = coh1_naive_rec_idx

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        ######{{{{   GET RATES OF TRAINED-TUNED-INDEXED E AND I UNITS IN NAIVE TRIALS   }}}}#######
        # get all these e and i units' actual rates in response to trials of each coherence level
        # using indices of units that are tuned in their trained states
        # look at their original responses in the naive state
        if not 'all_0e_to_0_rates' in locals():
            all_0e_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_0_rates = np.hstack([all_0e_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        # responses of all coh0-tuned e units to coherence 1
        if not 'all_0e_to_1_rates' in locals():
            all_0e_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()
        else:
            all_0e_to_1_rates = np.hstack([all_0e_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]].flatten()])

        # repeat for all_1e_to_0_rates, all_1e_to_1_rates, all_0i_to_0_rates, all_0i_to_1_rates, all_1i_to_0_rates, all_1i_to_1_rates

        if not 'all_1e_to_0_rates' in locals():
            all_1e_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_0_rates = np.hstack([all_1e_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        if not 'all_1e_to_1_rates' in locals():
            all_1e_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()
        else:
            all_1e_to_1_rates = np.hstack([all_1e_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]].flatten()])

        # now for the i units
        if not 'all_0i_to_0_rates' in locals():
            all_0i_to_0_rates = coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_0_rates = np.hstack([all_0i_to_0_rates,coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_0i_to_1_rates' in locals():
            all_0i_to_1_rates = coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()
        else:
            all_0i_to_1_rates = np.hstack([all_0i_to_1_rates,coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_0_rates' in locals():
            all_1i_to_0_rates = coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_0_rates = np.hstack([all_1i_to_0_rates,coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])

        if not 'all_1i_to_1_rates' in locals():
            all_1i_to_1_rates = coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()
        else:
            all_1i_to_1_rates = np.hstack([all_1i_to_1_rates,coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]].flatten()])


    ######{{{{   ADD THE NAIVE E AND I RATES TO EXISTING SUBPLOTS   }}}}#######

    ######{{{{   SUBPLOT FOR E COHERENCE 0 TRIALS   }}}}#######
    ax[0].hist(all_0e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[0].hist(all_1e_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[0].set_title('E rates to coherence 0 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR E COHERENCE 1 TRIALS   }}}}#######
    ax[1].hist(all_0e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[1].hist(all_1e_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[1].set_title('E rates to coherence 1 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR I COHERENCE 0 TRIALS   }}}}#######
    ax[2].hist(all_0i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[2].hist(all_1i_to_0_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[2].set_title('I rates to coherence 0 trials',fontname='Ubuntu')

    ######{{{{   SUBPLOT FOR I COHERENCE 1 TRIALS   }}}}#######
    ax[3].hist(all_0i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh0-tuned naive')
    ax[3].hist(all_1i_to_1_rates.flatten(),bins=30,density=True,alpha=0.7,label='coh1-tuned naive')
    ax[3].set_title('I rates to coherence 1 trials',fontname='Ubuntu')

        ######{{{{   CHARACTERIZE WITH NUMBERS COMPARING DISTRIBUTIONS   }}}}#######
        # main comparison is: WITHIN TRIALS AND UNITS (E OR I) OF A CERTAIN TYPE
        # how different are NAIVE VS TRAINED
        # how different are COH1-TUNED RATES VS COH0-TUNED RATES

        # plot distributions together for: TRAINED responses of 0-selective and 1-selective e and i units to coherence 0
        # NAIVE
        # in trained state, e and i units should differ in their responses according to their selectivity (0 more than 1 in this case)
        # should use the same units for NAIVE instead of the other way around, but you do want to know how many units are selective vs not

        # follow identity then, units selective in the naive case - their responses in the trained state
        # units selective in the trained state - their responses in the naive state

        # but in the end, maybe we show that the differences aren't so large

        # TRAINED responses of 0-selective and 1-selective e and i units to coherence 1
        # NAIVE
        # in the trained state, e and i units should differ in their responses according to their tuning (1 more than 0 in this case)

        # should we define selectivity better in some way?
        # maybe you'll decide you want to plot different groups together

        # definitely want to characterize how similar and different are the responses of 0-selective vs 1-selective units to a given stimulus

        # separate plot and set of numbers showing just how many e and i units on average (dist) are tuned to 0 or 1 in the naive and the trained states

    """
        if not 'all_coh0_i_rates' in locals():
            all_coh0_i_rates = coh0_rec_rates[:,e_end:].flatten()
        else:
            all_coh0_i_rates = np.hstack([all_coh0_i_rates,coh0_rec_rates[:,e_end:].flatten()])

        if not 'all_coh1_e_rates' in locals():
            all_coh1_e_rates = coh1_rec_rates[:,:e_end].flatten()
        else:
            all_coh1_e_rates = np.hstack([all_coh1_e_rates,coh1_rec_rates[:,:e_end].flatten()])

        if not 'all_coh1_i_rates' in locals():
            all_coh1_i_rates = coh1_rec_rates[:,e_end:].flatten()
        else:
            all_coh1_i_rates = np.hstack([all_coh1_i_rates,coh1_rec_rates[:,e_end:].flatten()])
    """

    naive_ct = [coh1_e_ct,coh1_i_ct,coh0_e_ct,coh0_i_ct]
    # the natural question that arises is: is it the SAME units?
    # are the Most Important ones the same units?

    """
    ax[0].hist(np.array(all_coh0_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh0_e_ct)))+' avg units)')
    # ACTUALLY, we want to plot not all units' rates, but actually separate based on tuning
    # so instead of coh0_rec_rates[:,:e_end], we want coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]]
    # and coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]]
    # to compare the spike rates of coh0 and coh1 tuned recurrent e units in response to coh0 trials
    # within the same plot we also want to see naive vs. trained, or maybe side by side
    # we would expect the two naive distributions to pretty much overlap, even though there is some separability
    # there should be more separability in the trained case

    # and in another plot we want to see the spike rates of coh0 and coh1 tuned recurrent i units in response to coh0 trials
    # coh0 and coh1 tuned recurrent e units to coh1 trials
    # coh0 and coh1 tuned recurrent i units to coh1 trials

    # essentially in the next plot we want to see their weights changing over time; this is rate
    # maybe you can do average rate of the tuned units as just a line thing over time.... yes okay
    # try out a couple different ways of visualization. that's tomorrow.
    # why am I reinventing the wheel? take a lot of what you've done before

    ax[1].hist(np.array(all_coh0_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh0_i_ct)))+' avg units)')
    ax[2].hist(np.array(all_coh1_e_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh1_e_ct)))+' avg units)')
    ax[3].hist(np.array(all_coh1_i_rates).flatten(),density=True,bins=30,alpha=0.6,label='naive ('+str(int(np.mean(coh1_i_ct)))+' avg units)')
    """

    for j in range(0,len(ax)):
        ax[j].set_ylabel('density',fontname='Ubuntu')
        ax[j].set_xlabel('rates (Hz)',fontname='Ubuntu')
        ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    if mix_tuned_indices:
        save_fname = spath+'/characterize_tuning_mixedidx_rate_test.png'
        plt.suptitle('Rates of tuned recurrent units; tuning defined within state',fontname='Ubuntu')
    else:
        save_fname = spath+'/characterize_tuning_trainedidx_rate_test.png'
        plt.suptitle('Rates of tuned recurrent units; tuning defined by trained state',fontname='Ubuntu')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()


    if plot_counts:
        ######{{{{   CREATE NEW PLOT SHOWING COUNTS OF TUNED E AND I UNITS IN NAIVE AND TRAINED STATES   }}}}#######
        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()

        # want to visually compare the avg number of tuned units in naive and trained cases

        ax[0].hist(np.array(trained_ct[2]).flatten(),alpha=0.7,density=True,label='coh0-tuned trained')
        ax[0].hist(np.array(trained_ct[0]).flatten(),alpha=0.7,density=True,label='coh1-tuned trained')
        ax[0].hist(np.array(naive_ct[2]).flatten(),alpha=0.7,density=True,label='coh0-tuned naive')
        ax[0].hist(np.array(naive_ct[0]).flatten(),alpha=0.7,density=True,label='coh1-tuned naive')
        ax[0].set_title('Number of E units that are tuned',fontname='Ubuntu')

        ax[1].hist(np.array(trained_ct[3]).flatten(),alpha=0.7,density=True,label='coh0-tuned trained')
        ax[1].hist(np.array(trained_ct[1]).flatten(),alpha=0.7,density=True,label='coh1-tuned trained')
        ax[1].hist(np.array(naive_ct[3]).flatten(),alpha=0.7,density=True,label='coh0-tuned naive')
        ax[1].hist(np.array(naive_ct[1]).flatten(),alpha=0.7,density=True,label='coh1-tuned naive')
        ax[1].set_title('Number of I units that are tuned',fontname='Ubuntu')

        plt.suptitle('Quantities of tuned recurrent units',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('density',fontname='Ubuntu')
            ax[j].set_xlabel('number of units',fontname='Ubuntu')
            ax[j].legend(fontsize="11",prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        save_fname = spath+'/count_tuning_rate_test.png'
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()


    ######{{{{   FOR LATER: CREATE NEW PLOT SHOWING THE RATE DISTRIBUTIONS OF NAIVE-TUNED-INDEXED E AND I UNITS IN BOTH NAIVE AND TRAINED TRIALS   }}}}#######

    # want a way to also quantify the extent to which units are tuned
    # like the ratio of their mean responses to i vs their mean responses to e
    # in both the trained and naive states for both coherence level trials
    # and for different varieties of training
    # this is the task for tomorrow

    return [trained_ct,naive_ct]


# plot over the course of training how MANY units become tuned

def tuned_rec_layer_over_training(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # plot over the course of training with shaded error bars
    # plot the average weight within and between coherence tuning of recurrent layer units
    # make sure all axes are comparable
    # get the numbers (avg and std weight for all of these connection types? shape tho?) for the weight distributions at the beginning and end of training

    # look at tuning to coherence level
    # look at connections between units in accordance to tuning to coherence level
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/final'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs:
        coh0_ee_ = []
        coh0_ei_ = []
        coh0_ie_ = []
        coh0_ii_ = []

        coh1_ee_ = []
        coh1_ei_ = []
        coh1_ie_ = []
        coh1_ii_ = []

        # coh 0 to 1
        het_ee_ = []
        het_ei_ = []
        het_ie_ = []
        het_ii_ = []

        # coh 1 to 0
        ero_ee_ = []
        ero_ei_ = []
        ero_ie_ = []
        ero_ii_ = []

        print('begin new exp')
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        data = np.load(os.path.join(np_dir,'991-1000.npz'))
        spikes=data['spikes']
        true_y=data['true_y']

        # find which units respond more to input of a certain coh level in the trained state
        coh0_rec_rates = []
        coh1_rec_rates = []

        for i in range(0,np.shape(true_y)[0]):
            for j in range(0,np.shape(true_y)[1]):
                if true_y[i][j][0]==true_y[i][j][seq_len-1]:
                    if true_y[i][j][0]==0:
                        coh0_rec_rates.append(np.mean(spikes[i][j],0))
                    else:
                        coh1_rec_rates.append(np.mean(spikes[i][j],0))

        # find which of the 300 recurrent units respond more on average to one coherence level over the other
        coh1_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        """
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx<e_end]))+' coh1-tuned e units')
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx>=e_end]))+' coh1-tuned i units')
        """
        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        """
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx<e_end]))+' coh0-tuned e units')
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx>=e_end]))+' coh0-tuned i units')
        """

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        # just average weights to begin with?
        coh1_e = np.array(coh1_rec_idx[coh1_rec_idx<e_end])
        coh1_i = np.array(coh1_rec_idx[coh1_rec_idx>=e_end])
        coh0_e = np.array(coh0_rec_idx[coh0_rec_idx<e_end])
        coh0_i = np.array(coh0_rec_idx[coh0_rec_idx>=e_end])


        # collect weights over all of training
        temporal_w = []
        data_files = filenames(num_epochs, epochs_per_file)

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            temp_data = np.load(filepath)
            temporal_w.append(temp_data['tv1.postweights'][99])

        # name them as homo and hetero lol
        # plot weights based on coh tuning over time

        for i in range(0,np.shape(temporal_w)[0]): # again over all training time, but now just one per file (100) instead of craziness (10000)
            coh0_ee_.append(np.mean(temporal_w[i][coh0_e,:][:,coh0_e]))
            coh0_ei_.append(np.mean(temporal_w[i][coh0_e,:][:,coh0_i]))
            coh0_ie_.append(np.mean(temporal_w[i][coh0_i,:][:,coh0_e]))
            coh0_ii_.append(np.mean(temporal_w[i][coh0_i,:][:,coh0_i]))

            coh1_ee_.append(np.mean(temporal_w[i][coh1_e,:][:,coh1_e]))
            coh1_ei_.append(np.mean(temporal_w[i][coh1_e,:][:,coh1_i]))
            coh1_ie_.append(np.mean(temporal_w[i][coh1_i,:][:,coh1_e]))
            coh1_ii_.append(np.mean(temporal_w[i][coh1_i,:][:,coh1_i]))

            het_ee_.append(np.mean(temporal_w[i][coh0_e,:][:,coh1_e]))
            het_ei_.append(np.mean(temporal_w[i][coh0_e,:][:,coh1_i]))
            het_ie_.append(np.mean(temporal_w[i][coh0_i,:][:,coh1_e]))
            het_ii_.append(np.mean(temporal_w[i][coh0_i,:][:,coh1_i]))

            ero_ee_.append(np.mean(temporal_w[i][coh1_e,:][:,coh0_e]))
            ero_ei_.append(np.mean(temporal_w[i][coh1_e,:][:,coh0_i]))
            ero_ie_.append(np.mean(temporal_w[i][coh1_i,:][:,coh0_e]))
            ero_ii_.append(np.mean(temporal_w[i][coh1_i,:][:,coh0_i]))

        if not np.isnan(coh0_ee_).any():
            if not "coh0_ee" in locals():
                coh0_ee = coh0_ee_
            else:
                coh0_ee = np.vstack([coh0_ee, coh0_ee_])

        if not np.isnan(coh0_ei_).any():
            if not "coh0_ei" in locals():
                coh0_ei = coh0_ei_
            else:
                coh0_ei = np.vstack([coh0_ei, coh0_ei_])

        if not np.isnan(coh0_ie_).any():
            if not "coh0_ie" in locals():
                coh0_ie = coh0_ie_
            else:
                coh0_ie = np.vstack([coh0_ie, coh0_ie_])

        if not np.isnan(coh0_ii_).any():
            if not "coh0_ii" in locals():
                coh0_ii = coh0_ii_
            else:
                coh0_ii = np.vstack([coh0_ii, coh0_ii_])

        if not np.isnan(coh1_ee_).any():
            if not "coh1_ee" in locals():
                coh1_ee = coh1_ee_
            else:
                coh1_ee = np.vstack([coh1_ee, coh1_ee_])

        if not np.isnan(coh1_ei_).any():
            if not "coh1_ei" in locals():
                coh1_ei = coh1_ei_
            else:
                coh1_ei = np.vstack([coh1_ei, coh1_ei_])

        if not np.isnan(coh1_ie_).any():
            if not "coh1_ie" in locals():
                coh1_ie = coh1_ie_
            else:
                coh1_ie = np.vstack([coh1_ie, coh1_ie_])

        if not np.isnan(coh1_ii_).any():
            if not "coh1_ii" in locals():
                coh1_ii = coh1_ii_
            else:
                coh1_ii = np.vstack([coh1_ii, coh1_ii_])

        if not np.isnan(het_ee_).any():
            if not "het_ee" in locals():
                het_ee = het_ee_
            else:
                het_ee = np.vstack([het_ee, het_ee_])

        if not np.isnan(het_ei_).any():
            if not "het_ei" in locals():
                het_ei = het_ei_
            else:
                het_ei = np.vstack([het_ei, het_ei_])

        if not np.isnan(het_ie_).any():
            if not "het_ie" in locals():
                het_ie = het_ie_
            else:
                het_ie = np.vstack([het_ie, het_ie_])

        if not np.isnan(het_ii_).any():
            if not "het_ii" in locals():
                het_ii = het_ii_
            else:
                het_ii = np.vstack([het_ii, het_ii_])

        if not np.isnan(ero_ee_).any():
            if not "ero_ee" in locals():
                ero_ee = ero_ee_
            else:
                ero_ee = np.vstack([ero_ee, ero_ee_])

        if not np.isnan(ero_ei_).any():
            if not "ero_ei" in locals():
                ero_ei = ero_ei_
            else:
                ero_ei = np.vstack([ero_ei, ero_ei_])

        if not np.isnan(ero_ie_).any():
            if not "ero_ie" in locals():
                ero_ie = ero_ie_
            else:
                ero_ie = np.vstack([ero_ie, ero_ie_])

        if not np.isnan(ero_ii_).any():
            if not "ero_ii" in locals():
                ero_ii = ero_ii_
            else:
                ero_ii = np.vstack([ero_ii, ero_ii_])

    """
    fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(8,10))

    epochs = np.shape(coh0_ee)[1]

    coh0_ee_mean = np.mean(coh0_ee,0)
    coh0_ee_std = np.std(coh0_ee,0)
    ax[0].plot(np.arange(0,epochs),coh0_ee_mean,color='slateblue',label='ee')
    ax[0].fill_between(np.arange(0,epochs),coh0_ee_mean-coh0_ee_std, coh0_ee_mean+coh0_ee_std, alpha=0.4, facecolor='slateblue')

    coh0_ei_mean = np.mean(coh0_ei,0)
    coh0_ei_std = np.std(coh0_ei,0)
    ax[0].plot(np.arange(0,epochs),coh0_ei_mean,color='mediumseagreen',label='ei')
    ax[0].fill_between(np.arange(0,epochs),coh0_ei_mean-coh0_ei_std, coh0_ei_mean+coh0_ei_std, alpha=0.4, facecolor='mediumseagreen')

    coh0_ie_mean = np.mean(coh0_ie,0)
    coh0_ie_std = np.std(coh0_ie,0)
    ax[0].plot(np.arange(0,epochs),coh0_ie_mean,color='darkorange',label='ie')
    ax[0].fill_between(np.arange(0,epochs),coh0_ie_mean-coh0_ie_std, coh0_ie_mean+coh0_ie_std, alpha=0.4, facecolor='darkorange')

    coh0_ii_mean = np.mean(coh0_ii,0)
    coh0_ii_std = np.std(coh0_ii,0)
    ax[0].plot(np.arange(0,epochs),coh0_ii_mean,color='orangered',label='ii')
    ax[0].fill_between(np.arange(0,epochs),coh0_ii_mean-coh0_ii_std, coh0_ii_mean+coh0_ii_std, alpha=0.4, facecolor='orangered')

    ax[0].set_title('coherence 0 tuned recurrent connections',fontname='Ubuntu')
    ax[0].set_ylabel('average weight',fontname='Ubuntu')


    ax[1].plot(np.arange(0,epochs),np.mean(coh1_ee,0),color='slateblue',label='ee')
    ax[1].fill_between(np.arange(0,epochs),np.mean(coh1_ee,0)-np.std(coh1_ee,0), np.mean(coh1_ee,0)+np.std(coh1_ee,0), alpha=0.4, facecolor='slateblue')

    ax[1].plot(np.arange(0,epochs),np.mean(coh1_ei,0),color='mediumseagreen',label='ei')
    ax[1].fill_between(np.arange(0,epochs),np.mean(coh1_ei,0)-np.std(coh1_ei,0), np.mean(coh1_ei,0)+np.std(coh1_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[1].plot(np.arange(0,epochs),np.mean(coh1_ie,0),color='darkorange',label='ie')
    ax[1].fill_between(np.arange(0,epochs),np.mean(coh1_ie,0)-np.std(coh1_ie,0), np.mean(coh1_ie,0)+np.std(coh1_ie,0), alpha=0.4, facecolor='darkorange')

    ax[1].plot(np.arange(0,epochs),np.mean(coh1_ii,0),color='orangered',label='ii')
    ax[1].fill_between(np.arange(0,epochs),np.mean(coh1_ii,0)-np.std(coh1_ii,0), np.mean(coh1_ii,0)+np.std(coh1_ii,0), alpha=0.4, facecolor='orangered')

    ax[1].set_title('coherence 1 tuned recurrent connections',fontname='Ubuntu')
    ax[1].set_ylabel('average weight',fontname='Ubuntu')


    ax[2].plot(np.arange(0,epochs),np.mean(het_ee,0),color='slateblue',label='ee')
    ax[2].fill_between(np.arange(0,epochs),np.mean(het_ee,0)-np.std(het_ee,0), np.mean(het_ee,0)+np.std(het_ee,0), alpha=0.4, facecolor='slateblue')

    ax[2].plot(np.arange(0,epochs),np.mean(het_ei,0),color='mediumseagreen',label='ei')
    ax[2].fill_between(np.arange(0,epochs),np.mean(het_ei,0)-np.std(het_ei,0), np.mean(het_ei,0)+np.std(het_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[2].plot(np.arange(0,epochs),np.mean(het_ie,0),color='darkorange',label='ie')
    ax[2].fill_between(np.arange(0,epochs),np.mean(het_ie,0)-np.std(het_ie,0), np.mean(het_ie,0)+np.std(het_ie,0), alpha=0.4, facecolor='darkorange')

    ax[2].plot(np.arange(0,epochs),np.mean(het_ii,0),color='orangered',label='ii')
    ax[2].fill_between(np.arange(0,epochs),np.mean(het_ii,0)-np.std(het_ii,0), np.mean(het_ii,0)+np.std(het_ii,0), alpha=0.4, facecolor='orangered')

    ax[2].set_title('coherence 0 to coherence 1 tuned recurrent connections',fontname='Ubuntu')
    ax[2].set_ylabel('average weight',fontname='Ubuntu')

    ax[3].plot(np.arange(0,epochs),np.mean(ero_ee,0),color='slateblue',label='ee')
    ax[3].fill_between(np.arange(0,epochs),np.mean(ero_ee,0)-np.std(ero_ee,0), np.mean(ero_ee,0)+np.std(ero_ee,0), alpha=0.4, facecolor='slateblue')

    ax[3].plot(np.arange(0,epochs),np.mean(ero_ei,0),color='mediumseagreen',label='ei')
    ax[3].fill_between(np.arange(0,epochs),np.mean(ero_ei,0)-np.std(ero_ei,0), np.mean(ero_ei,0)+np.std(ero_ei,0), alpha=0.4, facecolor='mediumseagreen')

    ax[3].plot(np.arange(0,epochs),np.mean(ero_ie,0),color='darkorange',label='ie')
    ax[3].fill_between(np.arange(0,epochs),np.mean(ero_ie,0)-np.std(ero_ie,0), np.mean(ero_ie,0)+np.std(ero_ie,0), alpha=0.4, facecolor='darkorange')

    ax[3].plot(np.arange(0,epochs),np.mean(ero_ii,0),color='orangered',label='ii')
    ax[3].fill_between(np.arange(0,epochs),np.mean(ero_ii,0)-np.std(ero_ii,0), np.mean(ero_ii,0)+np.std(ero_ii,0), alpha=0.4, facecolor='orangered')

    ax[3].set_title('coherence 1 to coherence 0 tuned recurrent connections',fontname='Ubuntu')
    ax[3].set_ylabel('average weight',fontname='Ubuntu')

    for j in range(0,len(ax)):
        ax[j].set_xlim(left=-5,right=105)
        ax[j].set_ylim(bottom=-1.5,top=0.25)
        ax[j].set_xlabel('training epoch')
        ax[j].legend(prop={"family":"Ubuntu"})
        for tick in ax[j].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[j].get_yticklabels():
            tick.set_fontname("Ubuntu")

    plt.suptitle('Recurrent Connectivity by Coherence Tuning: Rate Trained',fontname='Ubuntu')
    save_fname = spath+'/rec_weights_by_tuning_over_ratetraining.png'
    plt.subplots_adjust(hspace=0.8,wspace=0.8)
    plt.draw()
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()
    """

    return [coh0_ee,coh0_ei,coh0_ie,coh0_ii,coh1_ee,coh1_ei,coh1_ie,coh1_ii,het_ee,het_ei,het_ie,het_ii,ero_ee,ero_ei,ero_ie,ero_ii]

    # COMPARE

    """
    naive_means = []
    naive_stds = []
    trained_means = []
    trained_stds = []
    for arr in arrs:
        naive_means.append(np.mean(arr[:,0]))
        naive_stds.append(np.std(arr[:,0]))
        trained_means.append(np.mean(arr[:,99]))
        trained_stds.append(np.std(arr[:,99]))

    # now look at just trained
    within_coh = [coh0_ee,coh0_ei,coh0_ie,coh0_ii,coh1_ee,coh1_ei,coh1_ie,coh1_ii]
    between_coh = [het_ee,het_ei,het_ie,het_ii,ero_ee,ero_ei,ero_ie,ero_ii]
    within_Ds = []
    within_ps = []
    btwn_Ds = []
    btwn_ps = []
    for i in range(0,len(within_coh)/2):
        [D,p] = scipy.stats.kstest(within_coh[i][:,99],within_coh[i+4][:,99])
        within_Ds.append(D)
        within_ps.append(p)
        [D,p] = scipy.stats.kstest(between_coh[i][:,99],between_coh[i+4][:,99])
        btwn_Ds.append(D)
        btwn_ps.append(p)

within_ii = np.hstack([coh0_ii[:,99],coh1_ii[:,99]])
within_ie = np.hstack([coh0_ie[:,99],coh1_ie[:,99]])
btwn_ii = np.hstack([het_ii[:,99],ero_ii[:,99]])
btwn_ie = np.hstack([het_ie[:,99],ero_ie[:,99]])

within_ee = np.hstack([coh0_ee[:,99],coh1_ee[:,99]])
within_ei = np.hstack([coh0_ei[:,99],coh1_ei[:,99]])
btwn_ee = np.hstack([het_ee[:,99],ero_ee[:,99]])
btwn_ei = np.hstack([het_ei[:,99],ero_ei[:,99]])

[D,p] = scipy.stats.kstest(within_ii,btwn_ii)
[D,p] = scipy.stats.kstest(within_ii,within_ie)
[D,p] = scipy.stats.kstest(within_ie,btwn_ie)
[D,p] = scipy.stats.kstest(btwn_ii,btwn_ie)

    # quantify with naive vs trained ratios

    # across coherence i / within coherence i naive
    np.mean([het_ii[:,0],ero_ii[:,0],het_ie[:,0],ero_ie[:,0]])/np.mean([coh0_ii[:,0],coh0_ie[:,0],coh1_ii[:,0],coh1_ie[:,0]])
    #1.196 dual trained CORRECTED TO 1.215
    #1.0576923076923077 rate trained CORRECTED TO 1.035 (need update text)

    # across coherence i / within coherence i trained
    np.mean([het_ii[:,99],ero_ii[:,99],het_ie[:,99],ero_ie[:,99]])/np.mean([coh0_ii[:,99],coh0_ie[:,99],coh1_ii[:,99],coh1_ie[:,99]])
    #1.904 dual trained CORRECTED TO 1.979
    #1.005921052631579 rate trained CORRECTED TO 1.027 (need update text)

    # across coherence e / within coherence e naive
    np.mean([het_ee[:,0],het_ei[:,0],ero_ee[:,0],ero_ei[:,0]])/np.mean([coh0_ee[:,0],coh0_ei[:,0],coh1_ee[:,0],coh1_ei[:,0]])
    #0.928
    previously, rate-trained 0.9956
    # across coherence e / within coherence e trained
    np.mean([het_ee[:,99],het_ei[:,99],ero_ee[:,99],ero_ei[:,99]])/np.mean([coh0_ee[:,99],coh0_ei[:,99],coh1_ee[:,99],coh1_ei[:,99]])
    #0.4995
    previously, rate_trained 1.051

    """


"""
# below this line are NOT priorities for now

#def input_amp_or_supp_based_on_training(dual_exp_dir,task_exp_dir,rate_exp_dir):
# this is the one where we were looking at single trials
# you require a way to characterize

def labeled_lines(exp_dir=spec_input_dirs,exp_season='winter'):
    # demonstrate the relationship between sum of input weights and sum of output weights
    # do so across all experiments of this type
    # plot some sort of line? or something? to show the relationship?
"""
