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
