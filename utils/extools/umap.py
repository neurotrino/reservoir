"""UMAP Embedding Analysis"""

# Use scikit-learn virtualenv
# source sklearn-venv/bin/activate

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import umap

save_name='recruit_bin10_full'
coh_lvl = 'coh0'
recruit_path = '/data/results/experiment1/recruitment_graphs_bin10_full/'
data_dir = "/data/experiments/"
experiment_string = "run-batch30-specout-onlinerate0.1-savey"
task_experiment_string = 'run-batch30-onlytaskloss'
num_epochs = 1000
epochs_per_file = 10
e_end = 241
i_end = 300
savepath = "/data/results/experiment1/"

#def map_syn_graphs(save_name):
# use pandas dataset?
