"""Dynamic (spike) analysis for series of completed experiments"""

# external ----
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../')
sys.path.append('../../')

# internal ----
from utils.misc import filenames
from utils.misc import get_experiments

data_dir = '/data/experiments/'

#def plot_rates_over_time():
    # plot for each experiment, one rate value per batch update
    # rates averaged over entire runs and 30 trials for each update
    # rates of e units only
    # rates of i units only
    # we should save the precise input / trial structure / input spikes yikes 
