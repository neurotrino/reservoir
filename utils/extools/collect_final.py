"""final plot generation for series of completed experiments; synthesis of things in analyze_final"""

# ---- external imports -------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import scipy
import sys
import seaborn as sns
import networkx as nx

from scipy.sparse import load_npz

# ---- internal imports -------------------------------------------------------
sys.path.append("../")
sys.path.append("../../")

from utils.misc import filenames
from utils.misc import generic_filenames
from utils.misc import get_experiments
from utils.extools.analyze_structure import get_degrees
from utils.extools.fn_analysis import reciprocity
from utils.extools.fn_analysis import reciprocity_ei
from utils.extools.fn_analysis import calc_density
from utils.extools.fn_analysis import out_degree
from utils.extools.MI import simplest_confMI
from utils.extools.MI import simplest_asym_confMI
from utils.extools.analyze_structure import _nets_from_weights
from utils.extools.analyze_dynamics import fastbin
from utils.extools.analyze_dynamics import trial_recruitment_graphs
from utils.extools.analyze_dynamics import asym_trial_recruitment_graphs
from utils.extools.analyze_dynamics import threshold_fnet
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

#ALL DUAL TRAINED TO BEGIN WITH:
spec_output_dirs = ["run-batch30-specout-onlinerate0.1-savey","run-batch30-dualloss-silence","run-batch30-dualloss-swaplabels"]
spec_input_dirs = ["run-batch30-dualloss-specinput0.3-rewire"]
spec_nointoout_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
save_inz_dirs = ["run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz","run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
save_inz_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz"]
spec_nointoout_dirs_rate = ["run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire-inputx5-saveinz","run-batch30-rateloss-specinput0.2-nointoout-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-rateloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]
spec_nointoout_dirs_task = ["run-batch30-taskloss-specinput0.2-nointoout-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire","run-batch30-taskloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire-inputx5"]

def dists_of_all_weights(dual_exp_dir=spec_nointoout_dirs,task_exp_dir=spec_nointoout_dirs_task,rate_exp_dir=spec_nointoout_dirs_rate,exp_season='spring'):
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

def dists_of_all_rates(exp_dir=spec_input_dirs,exp_season='winter'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

def dists_of_all_synch(exp_dir=spec_input_dirs,exp_season='winter'):
    # aggregate over all experiments of this type
    # do so separately for coh0 and coh1 only trials!
    # plot distributions in naive state
    # plot distributions in trained state
    # do for rate trained as well
    # make everything on comparable axes
    # return means and stds, save in some file

def dists_of_input_rates(exp_dir=save_inz_dirs,exp_season='spring'):
    # also bring in the rate trained ones too. just anything that contains saveinz; also the original CNN outputs too
    # plot the distribution of rates of each channel in response to the two coherence levels
    # do so for all 16 channels together (and measure, compare their distributions)
    # do so for each channel separately and describe just how close their firing rates are to each coherence level

def input_layer_over_training_by_coherence(dual_exp_dir=save_inz_dirs,rate_exp_dir=save_inz_dirs_rate,exp_season='spring'):
    # characterize the connectivity from the input layer to recurrent
    # plot over the course of training with shaded error bars
    # for not-save-inz experiments, get the information about input channels' coherence tunings from the original CNN output file
    # get a number distribution to quantify this, maybe the 

def characterize_tuned_rec_populations(dual_exp_dir=spec_nointoout_dirs,task_exp_dir=spec_nointoout_dirs_task,rate_exp_dir=spec_nointoout_dirs_rate,exp_season='spring'):
    # determine tuning of each recurrent unit across each of these experiments
    # count up how many are tuned to each coherence level (according to trials of single coherence level)
    # include save inz as well into these spring experimental categories, okay
    # plot the distributions o

    # compare between dual, task, rate training
    # as best you can


# below this line are not priorities for now

#def input_amp_or_supp_based_on_training(dual_exp_dir,task_exp_dir,rate_exp_dir):

def labeled_lines(exp_dir=spec_input_dirs,exp_season='winter'):
    # demonstrate the relationship between sum of input weights and sum of output weights
    # do so across all experiments of this type
    # plot some sort of line? or something? to show the relationship?
