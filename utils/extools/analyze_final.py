"""(simple) analyses and final plot generation for series of completed experiments"""

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

# Paul Tol's colorblind-friendly palette for scientific visualization
COLOR_PALETTE = [
    "#332288",  # indigo
    "#117733",  # green
    "#44AA99",  # teal
    "#88CCEE",  # cyan
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#AA4499",  # purple
    "#882255",  # wine
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


def moving_average(spikes,bin):
    # spikes shaped [units x time]
    units = np.shape(spikes)[0]
    dur = np.shape(spikes)[1]
    dur_binned = dur-(bin-1)
    moving_avg = np.zeros([units,dur_binned])
    for t in range(0,dur_binned):
        moving_avg[:,t] = np.mean(spikes[:,t:t+bin],1)
    return moving_avg # still in the shape of [units] in first dimension


def describe_ei_by_tuning(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # look at tuning to coherence level
    # look at connections between units in accordance to tuning to coherence level
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs:
        print('begin new exp')
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
        trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))
        data=trained_data

        # go thru final epoch trials
        true_y = data['true_y']
        spikes = data['spikes']
        w = data['tv1.postweights']

        # collect weights over all of training
        temporal_w = []
        data_files = filenames(num_epochs, epochs_per_file)

        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            temp_data = np.load(filepath)
            temporal_w.append(temp_data['tv1.postweights'])

        # find which units respond more to input of a certain coh level across batches and trials
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
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx<e_end]))+' coh1-tuned e units')
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx>=e_end]))+' coh1-tuned i units')
        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx<e_end]))+' coh0-tuned e units')
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx>=e_end]))+' coh0-tuned i units')

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        # just average weights to begin with?
        coh1_e = np.array(coh1_rec_idx[coh1_rec_idx<e_end])
        coh1_i = np.array(coh1_rec_idx[coh1_rec_idx>=e_end])
        coh0_e = np.array(coh0_rec_idx[coh0_rec_idx<e_end])
        coh0_i = np.array(coh0_rec_idx[coh0_rec_idx>=e_end])

        # name them as homo and hetero lol

        ho_ee=np.mean(w[:,:,:][:,coh1_e,:][:,:,coh1_e],0)
        ho_ei=np.mean(w[:,:,:][:,coh1_e,:][:,:,coh1_i],0)
        ho_ie=np.mean(w[:,:,:][:,coh1_i,:][:,:,coh1_e],0)
        ho_ii=np.mean(w[:,:,:][:,coh1_i,:][:,:,coh1_i],0)

        het_ee=np.mean(w[:,:,:][:,coh1_e,:][:,:,coh0_e],0)
        het_ei=np.mean(w[:,:,:][:,coh1_e,:][:,:,coh0_i],0)
        het_ie=np.mean(w[:,:,:][:,coh1_i,:][:,:,coh0_e],0)
        het_ii=np.mean(w[:,:,:][:,coh1_i,:][:,:,coh0_i],0)

        ero_ee = np.mean(w[:,:,:][:,coh0_e,:][:,:,coh1_e],0)
        ero_ei = np.mean(w[:,:,:][:,coh0_e,:][:,:,coh1_i],0)
        ero_ie = np.mean(w[:,:,:][:,coh0_i,:][:,:,coh1_e],0)
        ero_ii = np.mean(w[:,:,:][:,coh0_i,:][:,:,coh1_i],0)

        mo_ee = np.mean(w[:,:,:][:,coh0_e,:][:,:,coh0_e],0)
        mo_ei = np.mean(w[:,:,:][:,coh0_e,:][:,:,coh0_i],0)
        mo_ie = np.mean(w[:,:,:][:,coh0_i,:][:,:,coh0_e],0)
        mo_ii = np.mean(w[:,:,:][:,coh0_i,:][:,:,coh0_i],0)

        fig, ax = plt.subplots(nrows=2,ncols=2)
        ax = ax.flatten()

        ax[0].hist(ho_ee[ho_ee!=0].flatten(),alpha=0.5,density=True,bins=30,label='ee',color='dodgerblue')
        ax[0].hist(ho_ei[ho_ei!=0].flatten(),alpha=0.5,density=True,bins=30,label='ei',color='mediumseagreen')
        ax[0].hist(ho_ie[ho_ie!=0].flatten(),alpha=0.5,density=True,bins=30,label='ie',color='darkorange')
        ax[0].hist(ho_ii[ho_ii!=0].flatten(),alpha=0.5,density=True,bins=30,label='ii',color='orangered')
        ax[0].set_title('coherence 1 tuned to coherence 1 tuned',fontname='Ubuntu')

        ax[1].hist(het_ee[het_ee!=0].flatten(),alpha=0.5,density=True,bins=30,label='ee',color='dodgerblue')
        ax[1].hist(het_ei[het_ei!=0].flatten(),alpha=0.5,density=True,bins=30,label='ei',color='mediumseagreen')
        ax[1].hist(het_ie[het_ie!=0].flatten(),alpha=0.5,density=True,bins=30,label='ie',color='darkorange')
        ax[1].hist(het_ii[het_ii!=0].flatten(),alpha=0.5,density=True,bins=30,label='ii',color='orangered')
        ax[1].set_title('coherence 1 tuned to coherence 0 tuned',fontname='Ubuntu')

        ax[2].hist(ero_ee[ero_ee!=0].flatten(),alpha=0.5,density=True,bins=30,label='ee',color='dodgerblue')
        ax[2].hist(ero_ei[ero_ei!=0].flatten(),alpha=0.5,density=True,bins=30,label='ei',color='mediumseagreen')
        ax[2].hist(ero_ie[ero_ie!=0].flatten(),alpha=0.5,density=True,bins=30,label='ie',color='darkorange')
        ax[2].hist(ero_ii[ero_ii!=0].flatten(),alpha=0.5,density=True,bins=30,label='ii',color='orangered')
        ax[2].set_title('coherence 0 tuned to coherence 1 tuned',fontname='Ubuntu')

        ax[3].hist(mo_ee[mo_ee!=0].flatten(),alpha=0.5,density=True,bins=30,label='ee',color='dodgerblue')
        ax[3].hist(mo_ei[mo_ei!=0].flatten(),alpha=0.5,density=True,bins=30,label='ei',color='mediumseagreen')
        ax[3].hist(mo_ie[mo_ie!=0].flatten(),alpha=0.5,density=True,bins=30,label='ie',color='darkorange')
        ax[3].hist(mo_ii[mo_ii!=0].flatten(),alpha=0.5,density=True,bins=30,label='ii',color='orangered')
        ax[3].set_title('coherence 0 tuned to coherence 0 tuned',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_ylabel('density',fontname='Ubuntu')
            ax[j].set_xlabel('weights',fontname='Ubuntu')
            ax[j].legend(prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('average trained recurrent weights by tuning',fontname='Ubuntu')

        save_fname = spath+'/'+exp_path+'_weights_by_tuning_quad.png'
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()

        # clustering part
        fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(5,6))

        # collect over time
        e_clustering = []
        i_clustering = []
        for i in range(0,np.shape(temporal_w)[0]):
            for j in range(0,np.shape(temporal_w)[1]):

                # look also at clustering within ee and ii
                _, Ge, Gi = _nets_from_weights(temporal_w[i][j])

                clustering = nx.clustering(Ge, nodes=Ge.nodes, weight="weight")
                clustering = list(clustering.items())
                e_clustering.append(np.mean(np.array(clustering)[:, 1]))

                clustering = nx.clustering(Gi, nodes=Gi.nodes, weight="weight")
                clustering = list(clustering.items())
                i_clustering.append(np.mean(np.array(clustering)[:, 1]))

        ax[0].plot(e_clustering,color='dodgerblue',label='ee')
        ax[0].plot(i_clustering,color='orangered',label='ii')
        ax[0].set_title('recurrent clustering over time',fontname='Ubuntu')
        ax[0].set_ylabel('clustering',fontname='Ubuntu')

        # plot weights based on coh tuning over time
        coh0_ee = []
        coh0_ei = []
        coh0_ie = []
        coh0_ii = []
        coh1_ee = []
        coh1_ei = []
        coh1_ie = []
        coh1_ii = []

        for i in range(0,np.shape(temporal_w)[0]): # again over all training time
            for j in range(0,np.shape(temporal_w)[1]):
                coh0_ee.append(np.mean(temporal_w[i,j,coh0_e,coh0_e]))
                coh0_ei.append(np.mean(temporal_w[i,j,coh0_e,coh0_i]))
                coh0_ie.append(np.mean(temporal_w[i,j,coh0_i,coh0_e]))
                coh0_ii.append(np.mean(temporal_w[i,j,coh0_i,coh0_i]))
                coh1_ee.append(np.mean(temporal_w[i,j,coh1_e,coh1_e]))
                coh1_ei.append(np.mean(temporal_w[i,j,coh1_e,coh1_i]))
                coh1_ie.append(np.mean(temporal_w[i,j,coh1_i,coh1_e]))
                coh1_ii.append(np.mean(temporal_w[i,j,coh1_i,coh1_i]))

        ax[1].plot(coh0_ee,color='slateblue',label='ee')
        ax[1].plot(coh0_ei,color='mediumseagreen',label='ei')
        ax[1].plot(coh0_ie,color='darkorange',label='ie')
        ax[1].plot(coh0_ii,color='orangered',label='ii')
        ax[1].set_title('coherence 0 tuned recurrent connections',fontname='Ubuntu')
        ax[1].set_ylabel('average weight',fontname='Ubuntu')

        ax[2].plot(coh1_ee,color='slateblue',label='ee')
        ax[2].plot(coh1_ei,color='mediumseagreen',label='ei')
        ax[2].plot(coh1_ie,color='darkorange',label='ie')
        ax[2].plot(coh1_ii,color='orangered',label='ii')
        ax[2].set_title('coherence 1 tuned recurrent connections',fontname='Ubuntu')
        ax[2].set_ylabel('average weight',fontname='Ubuntu')

        for j in range(0,len(ax)):
            ax[j].set_xlabel('training time')
            ax[j].legend(prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('recurrent connectivity by coherence tuning',fontname='Ubuntu')
        save_fname = spath+'/'+exp_path+'_weights_by_tuning_over_training.png'
        plt.subplots_adjust(hspace=0.8,wspace=0.8)
        plt.draw()
        plt.savefig(save_fname,dpi=300)
        # Teardown
        plt.clf()
        plt.close()


        """
        # look at clustering for ee and ii units that had the largest weight changes from naive to trained
        naive_w = naive_data['tv1.postweights'][0]
        trained_w = w[99]
        w_diff = np.abs(trained_w-naive_w)
        thr = np.quantile(w_diff, 0.25)
        greatest_delta_ws = np.where(w_diff[w_diff>thr])[0]

        _, Ge, Gi = _nets_from_weights(greatest_delta_ws)"""



def describe_tuning(exp_dirs=spec_output_dirs,exp_season='fall'):
    # plot the coherence-tuning properties of the recurrent units
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs:
        print('begin new exp')
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        naive_data = np.load(os.path.join(np_dir,"41-50.npz"))
        trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))
        data=trained_data

        # go thru final epoch trials
        true_y = data['true_y']
        spikes = data['spikes']

        # find which units respond more to input of a certain coh level across batches and trials
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
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx<e_end]))+' coh1-tuned e units')
        print('there are '+str(len(coh1_rec_idx[coh1_rec_idx>=e_end]))+' coh1-tuned i units')
        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx<e_end]))+' coh0-tuned e units')
        print('there are '+str(len(coh0_rec_idx[coh0_rec_idx>=e_end]))+' coh0-tuned i units')

        coh0_rec_rates = np.array(coh0_rec_rates)
        coh1_rec_rates = np.array(coh1_rec_rates)

        fig, ax = plt.subplots(nrows=2,ncols=1)

        # for each unit, plot their average rate to one vs the other
        ax[0].hist((coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]]).flatten(),alpha=0.5,color='dodgerblue',bins=30,density=True,label='coh1-driven e')
        ax[0].hist((coh0_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]]).flatten(),alpha=0.5,color='darkorange',bins=30,density=True,label='coh1-driven i')
        ax[0].hist((coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]]).flatten(),alpha=0.5,color='mediumseagreen',bins=30,density=True,label='coh0-driven e')
        ax[0].hist((coh0_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]]).flatten(),alpha=0.5,color='orangered',bins=30,density=True,label='coh0-driven i')
        ax[0].set_title('rates on coherence 0 trials',fontname='Ubuntu')
        ax[0].set_xlabel('average rate',fontname='Ubuntu')
        ax[0].set_ylabel('density',fontname='Ubuntu')

        ax[1].hist((coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx<e_end]]).flatten(),alpha=0.5,color='dodgerblue',bins=30,density=True,label='coh1-driven e')
        ax[1].hist((coh1_rec_rates[:,coh1_rec_idx[coh1_rec_idx>=e_end]]).flatten(),alpha=0.5,color='darkorange',bins=30,density=True,label='coh1-driven i')
        ax[1].hist((coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx<e_end]]).flatten(),alpha=0.5,color='mediumseagreen',bins=30,density=True,label='coh0-driven e')
        ax[1].hist((coh1_rec_rates[:,coh0_rec_idx[coh0_rec_idx>=e_end]]).flatten(),alpha=0.5,color='orangered',bins=30,density=True,label='coh0-driven i')
        ax[1].set_title('rates on coherence 1 trials',fontname='Ubuntu')
        ax[1].set_xlabel('average rate',fontname='Ubuntu')
        ax[1].set_ylabel('density',fontname='Ubuntu')

        ax[1].set_xlim([0,np.max(coh1_rec_rates)])
        ax[0].set_xlim([0,np.max(coh1_rec_rates)])

        for j in range(0,len(ax)):
            ax[j].legend(prop={"family":"Ubuntu"})
            for tick in ax[j].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[j].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Trained firing rates to coherence level',fontname='Ubuntu')

        save_fname = spath+'/'+exp_path+'_trained_rates_by_coherence.png'
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        plt.draw()
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


def single_trial_delay_corresp(exp_dirs=save_inz_dirs_rate,exp_season='spring',rand_exp_idx=1):

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # check if folder exists, otherwise create it for saving files
    spath = '/data/results/experiment1/set_plots/'+exp_season+'/trained_trials'
    if not os.path.isdir(spath):
        os.makedirs(spath)

    for xdir in exp_data_dirs:
        #xdir = 'run-batch30-dualloss-specinput0.2-nointoout-noinoutrewire-inputx5-swaplabels-saveinz [2023-04-14 05.09.58]'
        print('begin new exp')
        exp_path = xdir[-9:-1]

        # arbitrary selection for now
        #exp_path = '21.06.01'
        #xdir = 'run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire [2023-03-20 21.06.01]'

        np_dir = os.path.join(data_dir,xdir,"npz-data")
        naive_data = np.load(os.path.join(np_dir,"41-50.npz"))
        trained_data = np.load(os.path.join(np_dir,"991-1000.npz"))
        data=trained_data

        # go thru final epoch trials
        true_y = data['true_y'][99]
        pred_y = data['pred_y'][99]
        spikes = data['spikes'][99]
        w = data['tv1.postweights'][99]
        in_w = data['tv0.postweights'][99]
        in_spikes = data['inputs'][99]

        coh0_rates = []
        coh1_rates = []
        coh0_rec_rates = []
        coh1_rec_rates = []
        for i in range(0,len(true_y)):
            if true_y[i][0]==true_y[i][seq_len-1]:
                if true_y[i][0]==0:
                    coh0_rates.append(np.mean(in_spikes[i],0))
                    coh0_rec_rates.append(np.mean(spikes[i],0))
                else:
                    coh1_rates.append(np.mean(in_spikes[i],0))
                    coh1_rec_rates.append(np.mean(spikes[i],0))
        # find which of the 16 input channels respond more to one coherence level over the other
        coh1_idx = np.where(np.mean(coh1_rates,0)>np.mean(coh0_rates,0))[0]
        coh0_idx = np.where(np.mean(coh1_rates,0)<np.mean(coh0_rates,0))[0]

        # find which of the 300 recurrent units respond more to one coherence level over the other
        coh1_rec_idx = np.where(np.mean(coh1_rec_rates,0)>np.mean(coh0_rec_rates,0))[0]
        coh0_rec_idx = np.where(np.mean(coh1_rec_rates,0)<np.mean(coh0_rec_rates,0))[0]

        # go thru all trials in this batch
        for i in range(0,len(true_y)):
            # determine if there was a coherence change in this trial
            if true_y[i][0] != true_y[i][seq_len-1]:
                # if change in coherence, plot true_y, pred_y together
                # rates of the 16 input channels together-ish
                # rates of the two populations of input channels
                # rates of e units and i units
                # rates of e to e, ei, ie, ii
                # rates e that receive mostly coh 0 vs coh 1 drive
                # rates i that receive mostly coh 0 vs coh 1 drive

                # alright, alright
                diffs = np.diff(true_y[i],axis=0)
                # t_change is the first timestep of the new coherence level
                t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
                # find average pred before and after
                pre_avg = np.average(pred_y[i][:t_change])
                post_avg = np.average(pred_y[i][t_change:])
                # determine the duration after coherence change until we first pass (pos or neg direction) the after-change average
                if pre_avg < post_avg:
                    # if we are increasing coherence level, crossing is when we go above the 75th percentile of post-change preds
                    if np.shape(np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0])[0]>0:
                        delay_dur = np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0][0]
                    else:
                        delay_dur = np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0][0]
                elif pre_avg > post_avg:
                    # if we are decreasing coherence level, crossing is when we fall below the 25th percentile of post-change preds
                    if np.shape(np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0])[0]>0:
                        delay_dur = np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0][0]
                    else:
                        delay_dur = np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0]

                save_fname = spath+'/'+exp_path+'_bin20_rate_trial'+str(i)+'.png'
                fig, ax = plt.subplots(nrows=6,ncols=1,figsize=(8,11))

                ax[0].plot(pred_y[i],color='dodgerblue',alpha=0.5,label='prediction')
                ax[0].plot(true_y[i],color='mediumblue',alpha=0.5,label='true y')
                ax[0].vlines(t_change,ymin=np.min(pred_y[i]),ymax=np.max(pred_y[i]),color='red',label='t change')
                ax[0].vlines(t_change+delay_dur,ymin=np.min(pred_y[i]),ymax=np.max(pred_y[i]),color='darkorange',label='t delay')
                ax[0].set_ylabel('output',fontname='Ubuntu')
                ax[0].set_title('output',fontname='Ubuntu')

                # plot heatmap of the input spikes to different populations
                sns.heatmap(np.transpose(in_spikes[i]),cmap='crest',cbar=False,xticklabels=False,yticklabels=False,ax=ax[1])
                ax[1].set_ylabel('spike rate',fontname='Ubuntu')
                ax[1].set_title('input channel spikes',fontname='Ubuntu')

                """
                # plot average input rates across the two populations and overall
                #ax[2].plot(np.mean(in_spikes[i],1),color='dodgerblue',label='all inputs')
                # find the two input populations
                ax[2].plot(np.mean(in_spikes[i][:,coh0_idx],1),alpha=0.5,color='dodgerblue',label='coh 0 driven')
                ax[2].plot(np.mean(in_spikes[i][:,coh1_idx],1),alpha=0.5,color='yellowgreen',label='coh 1 driven')
                ax[2].vlines(t_change,ymin=np.min(np.mean(in_spikes[i][:,coh1_idx],1)),ymax=np.max(np.mean(in_spikes[i][:,coh1_idx],1)),color='red',label='t change')
                ax[2].vlines(t_change+delay_dur,ymin=np.min(np.mean(in_spikes[i][:,coh1_idx],1)),ymax=np.max(np.mean(in_spikes[i][:,coh1_idx],1)),color='darkorange',label='t delay')
                ax[2].set_ylabel('spike rate',fontname='Ubuntu')
                ax[2].set_title('input rates by group',fontname='Ubuntu')"""

                # plot the difference between average input rates of the two coherence-driven populations
                # find the two input populations' difference
                coh0_avg_rate = np.mean(moving_average(np.transpose(in_spikes[i][:,coh0_idx]),bin=20),0)
                coh1_avg_rate = np.mean(moving_average(np.transpose(in_spikes[i][:,coh1_idx]),bin=20),0)
                ax[2].plot(coh1_avg_rate-coh0_avg_rate,color='dodgerblue',label='coh1-coh0')
                #ax[2].vlines(t_change,ymin=np.min(coh1_avg_rate-coh0_avg_rate),ymax=np.max(coh1_avg_rate-coh0_avg_rate),color='red',label='t change')
                #ax[2].vlines(t_change+delay_dur,ymin=np.min(coh1_avg_rate-coh0_avg_rate),ymax=np.max(coh1_avg_rate-coh0_avg_rate),color='darkorange',label='t delay')
                ax[2].set_ylabel('spike rate difference',fontname='Ubuntu')
                ax[2].set_title('average difference in input group rates',fontname='Ubuntu')

                # plot the difference between coh0-driven e and coh1-driven e rates
                # difference between coh0-driven i and coh0-driven i rates
                # right below is drive based on sum of weights
                #coh0_rec = np.where(np.sum(in_w[coh0_idx,:],0)>np.sum(in_w[coh1_idx,:],0))[0]
                #coh1_rec = np.where(np.sum(in_w[coh1_idx,:],0)>np.sum(in_w[coh0_idx,:],0))[0]

                coh0_rec_e_rate = np.mean(moving_average(np.transpose(spikes[i][:,coh0_rec_idx[coh0_rec_idx<e_end]]),bin=20),0)
                coh0_rec_i_rate = np.mean(moving_average(np.transpose(spikes[i][:,coh0_rec_idx[coh0_rec_idx>=e_end]]),bin=20),0)
                coh1_rec_e_rate = np.mean(moving_average(np.transpose(spikes[i][:,coh1_rec_idx[coh1_rec_idx<e_end]]),bin=20),0)
                coh1_rec_i_rate = np.mean(moving_average(np.transpose(spikes[i][:,coh1_rec_idx[coh1_rec_idx>=e_end]]),bin=20),0)
                ax[3].plot(coh1_rec_e_rate-coh0_rec_e_rate,color='slateblue',label='e coh1-coh0')
                ax[3].plot(coh1_rec_i_rate-coh0_rec_i_rate,color='mediumseagreen',label='i coh1-coh0')
                ax[3].set_ylabel('spike rate difference',fontname='Ubuntu')
                ax[3].set_title('average difference between e and i recurrent rates by dominant input group',fontname='Ubuntu')

                input_avg_rates = moving_average(np.transpose(in_spikes[i][:,coh0_idx]),bin=20)
                for j in range(0,np.shape(input_avg_rates)[0]):
                    ax[4].plot(input_avg_rates[j])
                ax[4].set_ylabel('moving spike rate',fontname='Ubuntu')
                ax[4].set_title('coh0 driven input channels',fontname='Ubuntu')

                input_avg_rates = moving_average(np.transpose(in_spikes[i][:,coh1_idx]),bin=20)
                for j in range(0,np.shape(input_avg_rates)[0]):
                    ax[5].plot(input_avg_rates[j])
                ax[5].set_ylabel('moving spike rate',fontname='Ubuntu')
                ax[5].set_title('coh1 driven input channels',fontname='Ubuntu')

                #e_diff = np.mean(spikes[i][:,coh1_rec[coh1_rec<e_end]],1)-np.mean(spikes[i][:,coh0_rec[coh0_rec<e_end]],1)
                #i_diff = np.mean(spikes[i][:,coh1_rec[coh1_rec>=e_end]],1)-np.mean(spikes[i][:,coh0_rec[coh0_rec>=e_end]],1)
                #ax[3].plot(e_diff,color='dodgerblue',label='e coh1-coh0')
                #ax[3].plot(i_diff,color='mediumseagreen',label='i coh1-coh0')
                #ax[3].vlines(t_change,ymin=np.min(i_diff),ymax=np.max(i_diff),color='red',label='t change')
                #ax[3].vlines(t_change+delay_dur,ymin=np.min(i_diff),ymax=np.max(i_diff),color='darkorange',label='t delay')
                #ax[3].set_ylabel('spike rate difference',fontname='Ubuntu')
                #ax[3].set_title('average difference between e and i recurrent rates by dominant input group')

                # generate functional network and recruitment graphs for all timesteps of this trial
                #binned_z = fastbin(z=np.transpose(spikes[i]), bin_sz=20, num_units=300) # sharing 20 ms bins for everything for now
                #fn = simplest_confMI(binned_z,correct_signs=True)
                #rns = trial_recruitment_graphs(w, fn, binned_z, threshold=1)

                #rns_ee = rns[:,:e_end,:e_end]
                #rns_ei = rns[:,:e_end,e_end:]
                #rns_ie = rns[:,e_end:,:e_end]
                #rns_ii = rns[:,e_end:,e_end:]

                # plot the average ee ei ie ii functional weights over time
                #if len(rns_ee[rns_ee!=0])>0:
                    #ax[4].plot(np.mean(rns_ee[:][rns_ee[:]!=0],1),alpha=0.6,color='slateblue',label='ee')
                #if len(rns_ei[rns_ei!=0])>0:
                    #ax[4].plot(np.mean(rns_ei[:][rns_ei[:]!=0],1),alpha=0.6,color='dodgerblue',label='ei')
                #if len(rns_ie[rns_ie!=0])>0:
                    #ax[4].plot(np.mean(rns_ie[:][rns_ie[:]!=0],1),alpha=0.6,color='mediumseagreen',label='ie')
                #if len(rns_ii[rns_ii!=0])>0:
                    #ax[4].plot(np.mean(rns_ii[:][rns_ii[:]!=0],1),alpha=0.6,color='yellowgreen',label='ii')

                # plotting like nonzero above only works if have at least one nonzero in each bin
                #ax[4].plot(np.mean(rns_ee,(1,2)),alpha=0.7,color='slateblue',label='ee')
                #ax[4].plot(np.mean(rns_ei,(1,2)),alpha=0.7,color='dodgerblue',label='ei')
                #ax[4].plot(np.mean(rns_ie,(1,2)),alpha=0.7,color='mediumseagreen',label='ie')
                #ax[4].plot(np.mean(rns_ii,(1,2)),alpha=0.7,color='yellowgreen',label='ii')
                #ax[4].set_ylabel('weight',fontname='Ubuntu')
                #ax[4].vlines(int(t_change/20),ymin=np.min(rns),ymax=np.max(rns),color='red',label='t change')
                #ax[4].vlines(int((t_change+delay_dur)/20),ymin=np.min(rns),ymax=np.max(rns),color='darkorange',label='t delay')
                #ax[4].set_title('average recurrent recruitment weights',fontname='Ubuntu')
                # plot the average density of those over time as well
                # plot the ee clustering and ii clustering over time as well
                # maybe just start with the first
                # plot a vline according to corrected bin

                #binned_inz = fastbin(np.transpose(in_spikes[i]), 20, 16)
                #in_fn = simplest_asym_confMI(binned_inz, binned_z, correct_signs=False)
                #in_rns = asym_trial_recruitment_graphs(in_w, in_fn, binned_inz, binned_z, threshold=1)
                #in_rns_e = in_rns[:,:,:e_end]
                #in_rns_i = in_rns[:,:,e_end:]
                #ax[5].plot(np.mean(in_rns_e,(1,2)),alpha=0.7,color='dodgerblue',label='in to e')
                #ax[5].plot(np.mean(in_rns_i,(1,2)),alpha=0.7,color='mediumseagreen',label='in to i')
                #ax[5].set_ylabel('weight',fontname='Ubuntu')
                #ax[5].set_title('average input recruitment weights',fontname='Ubuntu')

                #densities = np.zeros([4,np.shape(rns)[0]])
                #for j in range(0,np.shape(rns)[0]):
                    #densities[0,j] = calc_density(rns_ee[j])
                    #densities[1,j] = calc_density(rns_ei[j])
                    #densities[2,j] = calc_density(rns_ie[j])
                    #densities[3,j] = calc_density(rns_ii[j])
                #colors=['slateblue','dodgerblue','mediumseagreen','yellowgreen']
                #labels=['ee','ei','ie','ii']
                #for j in range(0,4):
                    #ax[5].plot(densities[j,:],alpha=0.7,color=colors[j],label=labels[j])
                #ax[5].set_ylabel('density',fontname='Ubuntu')
                #ax[5].set_title('recruitment densities',fontname='Ubuntu')

                # plot the rates of recurrent e units that project more to e vs those that project more to i
                # get the recurrent units that project more to e vs i and vice versa
                #more_to_e = np.where(np.sum(np.abs(w[:,:e_end]),1)>np.sum(np.abs(w[:,e_end:])))[0]
                #more_to_i = np.where(np.sum(np.abs(w[:,:e_end]),1)<np.sum(np.abs(w[:,e_end:])))[0]

                # plot the rates of recurrent e units that project more to i vs recurrent i units that project more to e
                #ax[4].plot(np.mean(spikes[i][:,more_to_e[more_to_e<e_end]],1),alpha=0.5,color='slateblue',label='e driving e')
                #ax[4].plot(np.mean(spikes[i][:,more_to_e[more_to_e>=e_end]],1),alpha=0.5,color='dodgerblue',label='e driving i')
                #ax[4].plot(np.mean(spikes[i][:,more_to_i[more_to_i<e_end]],1),alpha=0.5,color='mediumseagreen',label='i driving e')
                #ax[4].plot(np.mean(spikes[i][:,more_to_i[more_to_i>=e_end]],1),alpha=0.5,color='yellowgreen',label='i driving i')
                #ax[4].set_ylabel('spike rate',fontname='Ubuntu')
                #ax[4].set_title('average e and i recurrent rates by dominant recurrent group')

                # plot their differences
                #ax[5].plot(np.mean(spikes[i][:,:e_end],1)-np.mean(spikes[i][:,e_end:]),color='slateblue',label='e drive vs i drive')
                #ax[5].plot(np.mean(spikes[i][:,more_to_e[more_to_e<e_end]],1)-np.mean(spikes[i][:,more_to_e[more_to_e>=e_end]],1),color='dodgerblue',label='e drive to e vs i')
                #ax[5].plot(np.mean(spikes[i][:,more_to_i[more_to_i<e_end]],1)-np.mean(spikes[i][:,more_to_i[more_to_i>=e_end]],1),color='mediumseagreen',label='i drive to e vs i')
                #ax[5].plot(np.mean(spikes[i][:,more_to_e[more_to_e>=e_end]],1)-np.mean(spikes[i][:,more_to_i[more_to_i<e_end]],1),color='yellowgreen',label='e drive to i vs i drive to e')
                #ax[5].set_ylabel('spike rate difference',fontname='Ubuntu')
                #ax[5].set_title('average difference between recurrent rates by dominant recurrent group')


                #ax[3].plot(np.mean(spikes[i][:,coh0_rec[coh0_rec>=e_end]],1),alpha=0.5,color='slateblue',label='coh 0 driven e')
                #ax[3].plot(np.mean(spikes[i][:,coh1_rec[coh1_rec>=e_end]],1),alpha=0.5,color='dodgerblue',label='coh 1 driven e')
                #ax[3].plot(np.mean(spikes[i][:,coh0_rec[coh0_rec<e_end]],1),alpha=0.5,color='mediumseagreen',label='coh 0 driven i')
                #ax[3].plot(np.mean(spikes[i][:,coh1_rec[coh1_rec<e_end]],1),alpha=0.5,color='yellowgreen',label='coh 1 driven i')
                #ax[3].vlines(t_change,ymin=np.min(np.mean(spikes[i][:,coh1_rec],1)),ymax=np.max(np.mean(spikes[i][:,coh1_rec],1)),color='red',label='t change')
                #ax[3].vlines(t_change+delay_dur,ymin=np.min(np.mean(spikes[i][:,coh1_rec],1)),ymax=np.max(np.mean(spikes[i][:,coh1_rec],1)),color='darkorange',label='t delay')
                #ax[3].set_ylabel('spike rate',fontname='Ubuntu')
                #ax[3].set_title('E and I rates by input group',fontname='Ubuntu')

                # plot average input rates across the two populations and overall
                #ax[2].plot(np.mean(in_spikes[i],1),color='dodgerblue',label='all inputs')
                # find the two input populations
                #ax[2].plot(np.mean(in_spikes[i][:,coh0_idx],1),alpha=0.5,color='mediumseagreen',label='coh 0 driven')
                #ax[2].plot(np.mean(in_spikes[i][:,coh1_idx],1),alpha=0.5,color='yellowgreen',label='coh 1 driven')
                #ax[2].vlines(t_change,ymin=np.min(np.mean(in_spikes[i][:,coh1_idx],1)),ymax=np.max(np.mean(in_spikes[i][:,coh1_idx],1)),color='red',label='t change')
                #ax[2].vlines(t_change+delay_dur,ymin=np.min(np.mean(in_spikes[i][:,coh1_idx],1)),ymax=np.max(np.mean(in_spikes[i][:,coh1_idx],1)),color='darkorange',label='t delay')
                #ax[2].set_ylabel('spike rate',fontname='Ubuntu')
                #ax[2].set_title('input rates by group',fontname='Ubuntu')

                #ax[3].plot(np.mean(spikes[i][:,:e_end],1),alpha=0.5,color='dodgerblue',label='excit')
                #ax[3].plot(np.mean(spikes[i][:,e_end:],1),alpha=0.5,color='darkorange',label='inhib')
                #ax[3].vlines(t_change,ymin=np.min(np.mean(spikes[i][:,:e_end],1)),ymax=np.max(np.mean(spikes[i][:,:e_end],1)),color='red',label='t change')
                #ax[3].vlines(t_change+delay_dur,ymin=np.min(np.mean(spikes[i][:,:e_end],1)),ymax=np.max(np.mean(spikes[i][:,:e_end],1)),color='darkorange',label='t delay')
                #ax[3].set_ylabel('spike rate',fontname='Ubuntu')
                #ax[3].set_title('E and I rates',fontname='Ubuntu')

                # plot rates of e units that mostly receive pop 1 vs pop 2
                # plot rates of i units that mostly receive pop 1 vs pop 2
                # find the units that mostly receive input from the two populations
                #coh0_rec = np.where(np.sum(in_w[coh0_idx,:],0)>np.sum(in_w[coh1_idx,:],0))[0]
                #coh1_rec = np.where(np.sum(in_w[coh1_idx,:],0)>np.sum(in_w[coh0_idx,:],0))[0]
                #ax[4].plot(np.mean(spikes[i][:,coh0_rec],1),alpha=0.5,color='mediumseagreen',label='coh 0 driven')
                #ax[4].plot(np.mean(spikes[i][:,coh1_rec],1),alpha=0.5,color='yellowgreen',label='coh 1 driven')
                #ax[4].vlines(t_change,ymin=np.min(np.mean(spikes[i][:,coh1_rec],1)),ymax=np.max(np.mean(spikes[i][:,coh1_rec],1)),color='red',label='t change')
                #ax[4].vlines(t_change+delay_dur,ymin=np.min(np.mean(spikes[i][:,coh1_rec],1)),ymax=np.max(np.mean(spikes[i][:,coh1_rec],1)),color='darkorange',label='t delay')
                #ax[4].set_ylabel('spike rate',fontname='Ubuntu')
                #ax[4].set_title('E and I rates by input group',fontname='Ubuntu')

                for j in range(0,len(ax)):
                    ax[j].set_xlabel('time (ms)',fontname='Ubuntu')
                    if j!=1 and j!=4 and j!=5:
                        ax[j].legend(prop={"family":"Ubuntu"})
                        for tick in ax[j].get_xticklabels():
                            tick.set_fontname("Ubuntu")
                        for tick in ax[j].get_yticklabels():
                            tick.set_fontname("Ubuntu")

                plt.suptitle('measures for trial '+str(i))
                plt.draw()
                plt.subplots_adjust(wspace=1.0, hspace=1.0)
                plt.draw()
                plt.savefig(save_fname,dpi=300)

                # Teardown
                plt.clf()
                plt.close()


def single_fn_delay_recruit(rn_bin=10,exp_dirs=spec_input_dirs,exp_season='spring',rand_exp_idx=5):
    # generate a single functional network across all trials for a particular batch update (last) of a dual-trained network
    # or honestly maybe just constrained to a couple change trials for now

    """
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])
    """

    # arbitrarily pick one experiment for now
    #xdir = exp_data_dirs[rand_exp_idx]
    #exp_path = xdir[-9:-1]
    exp_path = '21.06.01'
    xdir = 'run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire [2023-03-20 21.06.01]'

    # check if folder exists, otherwise create it for saving files
    if not os.path.isdir(os.path.join(savepath, exp_season+'_fns', exp_path, 'trained')):
        os.makedirs(os.path.join(savepath, exp_season+'_fns', exp_path, 'trained'))

    np_dir = os.path.join(data_dir, xdir, "npz-data")
    #naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
    trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

    # go through final epoch trials
    true_y = trained_data['true_y'][99]
    pred_y = trained_data['pred_y'][99]
    spikes = trained_data['spikes'][99]
    w = trained_data['tv1.postweights'][99]

    fns = []

    # go thru all trials in this batch
    for i in range(0,len(true_y)):
        # determine if there was a coherence change in this trial
        if true_y[i][0] != true_y[i][seq_len-1]:
            # generate a FN for this entire trial

            binned_z = fastbin(np.transpose(spikes[i]), rn_bin, 300) # sharing 20 ms bins for everything for now

            fn = simplest_confMI(binned_z,correct_signs=True)

            fns.append(fn)

            # find the time of change and delay in report
            diffs = np.diff(true_y[i],axis=0)
            # t_change is the first timestep of the new coherence level
            t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
            # find average pred before and after
            pre_avg = np.average(pred_y[i][:t_change])
            post_avg = np.average(pred_y[i][t_change:])
            # determine the duration after coherence change until we first pass (pos or neg direction) the after-change average
            if pre_avg < post_avg:
                # if we are increasing coherence level, crossing is when we go above the 75th percentile of post-change preds
                delay_dur = np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0][0]
            elif pre_avg > post_avg:
                # if we are decreasing coherence level, crossing is when we fall below the 25th percentile of post-change preds
                delay_dur = np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0][0]

            # take the period of time 250 ms before and 250 ms for recruitment graphs
            # generate 20-ms (rn_bin) recruitment graphs from binned spikes
            rn_binned_z = fastbin(np.transpose(spikes[i][t_change-250:t_change+delay_dur+250][:]), rn_bin, 300)

            rns = trial_recruitment_graphs(w, fn, rn_binned_z, threshold=1)
            # ragged array; each is sized timesteps x 300 x 300

            # save all recruitment networks for this trial for later UMAP analyses
            np.savez_compressed(
                savepath+exp_season+'_fns/'+exp_path+'/trained/trial_'+str(i)+'_rns',
                **{
                    "rns": rns
                }
            )

    # save all functional networks
    np.savez_compressed(
        savepath+exp_season+'_fns/'+exp_path+'/trained/fns',
        **{
            "fns": fns
        }
    )

def single_batch_recruit_coh_compare(rn_dir='/data/results/experiment1/spring_fns/21.06.01/trained/',rn_bin=10,exp_season='spring'):
    # load in fns for a particular final batch of an experiment
    # generate rns for coh0-only and coh1-only trials and see how we are doing
    # in terms of comparison of ee, ei, ie, and ii connections
    # maybe densities and strengths are a good place to begin
    # only make comparisons within a given coherence type

    data = np.load(rn_dir+'fns.npz',allow_pickle=True)
    fns = data['fns'] # one fn for each of < 30 trials within this final trained batch
    conn_types = ['e->e','e->i','i->e','i->i']

    exp_path = '21.06.01'
    xdir = '/data/experiments/run-batch30-dualloss-specinput0.2-nointoout-twopopsbyrate-noinoutrewire [2023-03-20 21.06.01]'
    data = np.load(xdir+'/npz-data/991-1000.npz')
    spikes = data['spikes'][99]
    true_y = data['true_y'][99]
    w = data['tv1.postweights'][99]

    # check if there is a change

    # 4 subplots for each coherence type
    # plot average strengths why not
    # are we even moving over time?
    # no, just plot the distributions

    fns = []
    cohs = []

    for i in range(0,len(spikes)): # for each of 30 trials within this batch
        if true_y[i][0]==true_y[i][-1]: # check if consistent coherence level

            fig, ax = plt.subplots(nrows=2,ncols=2)

            # generate fn from whole trial
            binned_z = fastbin(np.transpose(spikes[i]), rn_bin, 300) # sharing 20 ms bins for everything for now
            fn = simplest_confMI(binned_z,correct_signs=True)
            fns.append(fn)

            trial_dur = len(true_y[i])
            avg_weights = np.zeros([4,trial_dur])

            # generate recruitment graphs over all time points
            rns = trial_recruitment_graphs(w, fn, binned_z, threshold=1)

            if true_y[i][0]>0:
                coh_lvl = 'coh1'
                cohs.append(1)
            else:
                coh_lvl = 'coh0'
                cohs.append(0)

            # save rns
            np.savez_compressed(
                savepath+exp_season+'_fns/'+exp_path+'/trained/trial_'+str(i)+'_nochange_'+coh_lvl+'_rns',
                **{
                    "rns": rns
                }
            )

            rn_ee = rns[:,:241,:241]
            rn_ei = rns[:,:241,241:]
            rn_ie = rns[:,241:,:241]
            rn_ii = rns[:,241:,241:]

            ax[0,0].hist(rn_ee[rn_ee!=0],bins=30,density=True)
            ax[0,1].hist(rn_ei[rn_ei!=0],bins=30,density=True)
            ax[1,0].hist(rn_ie[rn_ie!=0],bins=30,density=True)
            ax[1,1].hist(rn_ii[rn_ii!=0],bins=30,density=True)

            ax = ax.flatten()
            for j in range(0,len(ax)):
                ax[j].set_xlabel('weights',fontname='Ubuntu')
                ax[j].set_ylabel('density',fontname='Ubuntu')
                ax[j].set_title(conn_types[j],fontname='Ubuntu')

            plt.suptitle('average recruitment graph weights for '+coh_lvl)
            plt.draw()
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            plt.draw()
            save_fname = rn_dir+'trial'+str(i)+'_nochange_'+coh_lvl+'_rn_weights.png'
            plt.savefig(save_fname,dpi=300)

            # Teardown
            plt.clf()
            plt.close()

            # to be really thorough I should do this for plots in the naive state
            # whatever we determine to be naive, or for equivalent experiments trained only on the rate
            # gossh that's a good deal to look at

    # save all fns
    np.savez_compressed(
        savepath+exp_season+'_fns/'+exp_path+'/trained/nochange_fns',
        **{
            "fns": fns,
            "cohs": cohs
        }
    )

    # plot all the functional nets as a
    for i in range(len(fns)):
        coh0_ws = []
        coh0_dens = []
        coh1_ws = []
        coh1_dens = []
        if cohs[i]==1:
            coh1_ws.append(np.mean(fns[i][fns[i]!=0]))
        else:
            coh0_ws.append(np.mean(fns[i][fns[i]!=0]))

    # i haven't done the simple thing of plotting activity rates of e and i units again, have i?


def input_fns(exp_dirs=save_inz_dirs,fn_dir='/data/results/experiment1/spring_fns/',fn_bin=10,exp_season='spring',threshold=0.1):
    # generate input-to-recurrent functional networks
    # begin with trained; also do for naive
    # plot the distribution of functional weights for the final (or first) batch's set of 30 trials
    # focus on no-coherence-change trials
    # later on generate input fns for the delay period

    # get all experiment folders within this season
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string, final_npz='991-1000.npz')
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string, final_npz='991-1000.npz')])

    fns_coh0_naive = []
    fns_coh1_naive = []
    fns_coh0_trained = []
    fns_coh1_trained = []

    # load in data for each experiment
    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        in_spikes = naive_data['inputs'][0]
        spikes = naive_data['spikes'][0]
        true_y = naive_data['true_y'][0]

        for i in range(0,len(true_y)): # for each trial
            if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change in this trial
                binned_inz = fastbin(np.transpose(in_spikes[i]), fn_bin, 16)
                binned_z = fastbin(np.transpose(spikes[i]), fn_bin, 300) # sharing 10 ms bins for everything for now
                # would be fun to visualize as heatmap later
                fn = simplest_asym_confMI(binned_inz, binned_z, correct_signs=False)
                if threshold is not None: # threshold values in fn to top quartile, decile, etc.
                    fn = threshold_fnet(fn,threshold,copy=False)
                if true_y[i][0]==0:
                    # coherence 0
                    fns_coh0_naive.append(fn)
                else:
                    # coherence 1
                    fns_coh1_naive.append(fn)

        # repeat for trained
        in_spikes = trained_data['inputs'][99]
        spikes = trained_data['spikes'][99]
        true_y = trained_data['true_y'][99]

        for i in range(0,len(true_y)): # for each trial
            if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change in this trial
                binned_inz = fastbin(np.transpose(in_spikes[i]), fn_bin, 16)
                binned_z = fastbin(np.transpose(spikes[i]), fn_bin, 300) # sharing 10 ms bins for everything for now
                # would be fun to visualize as heatmap later
                fn = simplest_asym_confMI(binned_inz, binned_z, correct_signs=False)
                if threshold is not None: # threshold values in fn to top quartile, decile, etc.
                    fn = threshold_fnet(fn,threshold,copy=False)
                if true_y[i][0]==0:
                    # coherence 0
                    fns_coh0_trained.append(fn)
                else:
                    # coherence 1
                    fns_coh1_trained.append(fn)

    # save FNs
    np.savez_compressed(
        fn_dir+'coherence_separate_input_fns_decile',
        **{
            "fns_coh0_naive": fns_coh0_naive,
            "fns_coh1_naive": fns_coh1_naive,
            "fns_coh0_trained": fns_coh0_trained,
            "fns_coh1_trained": fns_coh1_trained
        }
    )

    # start with naive
    for i in range(0,len(fns_coh0_naive)): # for each trial
        if not 'channel_fns_coh0_naive' in locals():
            channel_fns_coh0_naive = fns_coh0_naive[i]
        else:
            channel_fns_coh0_naive = np.hstack([channel_fns_coh0_naive,fns_coh0_naive[i]]) # stack with first dimension as n_input (16) always

    for i in range(0,len(fns_coh1_naive)): # for each trial
        if not 'channel_fns_coh1_naive' in locals():
            channel_fns_coh1_naive = fns_coh0_naive[i]
        else:
            channel_fns_coh1_naive = np.hstack([channel_fns_coh1_naive,fns_coh1_naive[i]])

    # plot distributions of functional weights for all 16 input channels
    fig, ax = plt.subplots(nrows=2,ncols=2)

    sns.heatmap(np.mean(fns_coh0_naive,0), ax=ax[0,0])
    #ax[0,0].hist(np.transpose(channel_fns_coh0_naive),bins=20,histtype='step', density=True, stacked=True)
    ax[0,0].set_title('naive coherence 0', fontname="Ubuntu")
    #ax[1,0].hist(np.transpose(channel_fns_coh1_naive),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh1_naive,0), ax=ax[1,0])
    ax[1,0].set_title('naive coherence 1', fontname="Ubuntu")

    # now do trained
    for i in range(0,len(fns_coh0_trained)): # for each trial
        if not 'channel_fns_coh0_trained' in locals():
            channel_fns_coh0_trained = fns_coh0_trained[i]
        else:
            channel_fns_coh0_trained = np.hstack([channel_fns_coh0_trained,fns_coh0_trained[i]]) # stack with first dimension as n_input (16) always

    for i in range(0,len(fns_coh1_trained)): # for each trial
        if not 'channel_fns_coh1_trained' in locals():
            channel_fns_coh1_trained = fns_coh0_trained[i]
        else:
            channel_fns_coh1_trained = np.hstack([channel_fns_coh1_trained,fns_coh1_trained[i]])

    #ax[0,1].hist(np.transpose(channel_fns_coh0_trained),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh0_trained,0), ax=ax[0,1])
    ax[0,1].set_title('trained coherence 0', fontname="Ubuntu")
    #ax[1,1].hist(np.transpose(channel_fns_coh1_trained),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh1_trained,0), ax=ax[1,1])
    ax[1,1].set_title('trained coherence 1', fontname="Ubuntu")


    ax = ax.flatten()
    for i in range(0,len(ax)):
        ax[i].set_xlabel('recurrent units', fontname="Ubuntu")
        ax[i].set_ylabel('input channels', fontname='Ubuntu')
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        #ax[i].set_xlim([0,0.11])

    plt.suptitle("Functional weights of 16 input channels", fontname="Ubuntu")

    # Draw and save
    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    save_fname = savepath+'/spring_fns/spring_input_fn_decile_heatmaps.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

    """# now that fns have been aggregated for all final-batch trials of all experiments,
    # plot quad of their weight distributions
    # convert to numpy arrays
    fns_coh0_ee = np.array(fns_coh0)[:,:e_end,:e_end].flatten()
    fns_coh0_ei = np.array(fns_coh0)[:,:e_end,e_end:].flatten()
    fns_coh0_ie = np.array(fns_coh0)[:,e_end:,:e_end].flatten()
    fns_coh0_ii = np.array(fns_coh0)[:,e_end:,e_end:].flatten()
    fns_coh1_ee = np.array(fns_coh1)[:,:e_end,:e_end].flatten()
    fns_coh1_ei = np.array(fns_coh1)[:,:e_end,e_end:].flatten()
    fns_coh1_ie = np.array(fns_coh1)[:,e_end:,:e_end].flatten()
    fns_coh1_ii = np.array(fns_coh1)[:,e_end:,e_end:].flatten()

    # plot quad for just coherence 0 to begin with
    fig, ax = plt.subplots(nrows=1,ncols=2)
    ax[0].hist(fns_coh0_ee[fns_coh0_ee!=0],bins=30,alpha=0.5,density=True,color='dodgerblue')
    ax[0].hist(fns_coh0_ei[fns_coh0_ei!=0],bins=30,alpha=0.5,density=True,color='seagreen')
    ax[0].hist(fns_coh0_ie[fns_coh0_ie!=0],bins=30,alpha=0.5,density=True,color='darkorange')
    ax[0].hist(fns_coh0_ii[fns_coh0_ii!=0],bins=30,alpha=0.5,density=True,color='orangered')
    ax[0].legend(['ee','ei','ie','ii'])
    ax[0].set_title('Coherence label 0',fontname='Ubuntu')

    ax[1].hist(fns_coh1_ee[fns_coh1_ee!=0],bins=30,alpha=0.5,color='dodgerblue',density=True)
    ax[1].hist(fns_coh1_ei[fns_coh1_ei!=0],bins=30,alpha=0.5,color='seagreen',density=True)
    ax[1].hist(fns_coh1_ie[fns_coh1_ie!=0],bins=30,alpha=0.5,color='darkorange',density=True)
    ax[1].hist(fns_coh1_ii[fns_coh1_ii!=0],bins=30,alpha=0.5,color='orangered',density=True)
    ax[1].legend(['ee','ei','ie','ii'])
    ax[1].set_title('Coherence label 1',fontname='Ubuntu')

    plt.suptitle('Trained input-to-main functional weights',fontname='Ubuntu')

    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # go through and set all axes
    for i in range(0,len(ax)):
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        ax[i].set_xlabel('functional weight values',fontname='Ubuntu')
        ax[i].set_ylabel('density',fontname='Ubuntu')

    plt.draw()

    save_fname = fn_dir+'coherence_separate_trained_input_fn_weights.png'
    plt.savefig(save_fname,dpi=300)"""

def rec_fns_based_on_input_fns(exp_dirs=spec_nointoout_dirs_rate,fn_dir='/data/results/experiment1/spring_fns/',fn_bin=10,exp_season='spring',threshold=0.1):

    # get all experiment folders within this season
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string, final_npz='991-1000.npz')
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string, final_npz='991-1000.npz')])

    fns_coh0_naive = []
    fns_coh1_naive = []
    fns_coh0_trained = []
    fns_coh1_trained = []

    # load in data for each experiment
    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        spikes = naive_data['spikes'][0]
        true_y = naive_data['true_y'][0]

        for i in range(0,len(true_y)): # for each trial
            if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change in this trial
                #binned_inz = fastbin(np.transpose(in_spikes[i]), fn_bin, 16)
                binned_z = fastbin(np.transpose(spikes[i]), fn_bin, 300) # sharing 10 ms bins for everything for now
                # would be fun to visualize as heatmap later
                fn = simplest_confMI(binned_z, correct_signs=True)
                if threshold is not None: # threshold values in fn to top quartile, decile, etc.
                    fn = threshold_fnet(fn,threshold,copy=False)
                if true_y[i][0]==0:
                    # coherence 0
                    fns_coh0_naive.append(fn)
                else:
                    # coherence 1
                    fns_coh1_naive.append(fn)

        # repeat for trained
        spikes = trained_data['spikes'][99]
        true_y = trained_data['true_y'][99]

        for i in range(0,len(true_y)): # for each trial
            if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change in this trial
                #binned_inz = fastbin(np.transpose(in_spikes[i]), fn_bin, 16)
                binned_z = fastbin(np.transpose(spikes[i]), fn_bin, 300) # sharing 10 ms bins for everything for now
                # would be fun to visualize as heatmap later
                fn = simplest_confMI(binned_z, correct_signs=True)
                if threshold is not None: # threshold values in fn to top quartile, decile, etc.
                    fn = threshold_fnet(fn,threshold,copy=False)
                if true_y[i][0]==0:
                    # coherence 0
                    fns_coh0_trained.append(fn)
                else:
                    # coherence 1
                    fns_coh1_trained.append(fn)

    # save FNs
    np.savez_compressed(
        fn_dir+'coherence_separate_rate_fns_decile',
        **{
            "fns_coh0_naive": fns_coh0_naive,
            "fns_coh1_naive": fns_coh1_naive,
            "fns_coh0_trained": fns_coh0_trained,
            "fns_coh1_trained": fns_coh1_trained
        }
    )

    # start with naive
    for i in range(0,len(fns_coh0_naive)): # for each trial
        if not 'channel_fns_coh0_naive' in locals():
            channel_fns_coh0_naive = fns_coh0_naive[i]
        else:
            channel_fns_coh0_naive = np.hstack([channel_fns_coh0_naive,fns_coh0_naive[i]]) # stack with first dimension as n_input (16) always

    for i in range(0,len(fns_coh1_naive)): # for each trial
        if not 'channel_fns_coh1_naive' in locals():
            channel_fns_coh1_naive = fns_coh0_naive[i]
        else:
            channel_fns_coh1_naive = np.hstack([channel_fns_coh1_naive,fns_coh1_naive[i]])

    # plot distributions of functional weights for all 16 input channels
    fig, ax = plt.subplots(nrows=2,ncols=2)

    sns.heatmap(np.mean(fns_coh0_naive,0), ax=ax[0,0])
    #ax[0,0].hist(np.transpose(channel_fns_coh0_naive),bins=20,histtype='step', density=True, stacked=True)
    ax[0,0].set_title('naive coherence 0', fontname="Ubuntu")
    #ax[1,0].hist(np.transpose(channel_fns_coh1_naive),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh1_naive,0), ax=ax[1,0])
    ax[1,0].set_title('naive coherence 1', fontname="Ubuntu")

    # now do trained
    for i in range(0,len(fns_coh0_trained)): # for each trial
        if not 'channel_fns_coh0_trained' in locals():
            channel_fns_coh0_trained = fns_coh0_trained[i]
        else:
            channel_fns_coh0_trained = np.hstack([channel_fns_coh0_trained,fns_coh0_trained[i]]) # stack with first dimension as n_input (16) always

    for i in range(0,len(fns_coh1_trained)): # for each trial
        if not 'channel_fns_coh1_trained' in locals():
            channel_fns_coh1_trained = fns_coh0_trained[i]
        else:
            channel_fns_coh1_trained = np.hstack([channel_fns_coh1_trained,fns_coh1_trained[i]])

    #ax[0,1].hist(np.transpose(channel_fns_coh0_trained),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh0_trained,0), ax=ax[0,1])
    ax[0,1].set_title('trained coherence 0', fontname="Ubuntu")
    #ax[1,1].hist(np.transpose(channel_fns_coh1_trained),bins=20,histtype='step', density=True, stacked=True)
    sns.heatmap(np.mean(fns_coh1_trained,0), ax=ax[1,1])
    ax[1,1].set_title('trained coherence 1', fontname="Ubuntu")


    ax = ax.flatten()
    for i in range(0,len(ax)):
        ax[i].set_xlabel('recurrent units', fontname="Ubuntu")
        ax[i].set_ylabel('recurrent units', fontname='Ubuntu')
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        #ax[i].set_xlim([0,0.11])

    plt.suptitle("Functional weights of recurrent layer", fontname="Ubuntu")

    # Draw and save
    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    save_fname = savepath+'/spring_fns/spring_rate_fn_decile_heatmaps.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

def within_coh_comparisons(exp_dirs=spec_nointoout_dirs_rate,fn_dir='/data/results/experiment1/spring_fns/',fn_bin=10,exp_season='spring'):
    # generate FNs based on only coh 0 and coh 1 responses in single trials of the final trained batch
    # compare ee ei ie ii weights within the functional network for each coherence level only

    # get all experiment folders within this season
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    fns_coh0 = []
    fns_coh1 = []

    # load in trained data for each experiment
    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        #naive_data = np.load(os.path.join(np_dir,"1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        spikes = trained_data['spikes'][0]
        true_y = trained_data['true_y'][0]

        # go thru all trials within this batch
        # determine if there was a coherence change or not

        for i in range(0,len(true_y)):
            if true_y[i][0]==true_y[i][seq_len-1]: # no coherence change
                binned_z = fastbin(np.transpose(spikes[i]), fn_bin, 300) # sharing 20 ms bins for everything for now
                fn = simplest_confMI(binned_z,correct_signs=True)
                if true_y[i][0]==0:
                    # coherence 0
                    fns_coh0.append(fn)
                else:
                    # coherence 1
                    fns_coh1.append(fn)

    # save FNs
    np.savez_compressed(
        fn_dir+'coherence_separate_trained_rate_fns',
        **{
            "fns_coh0": fns_coh0,
            "fns_coh1": fns_coh1
        }
    )

    # now that fns have been aggregated for all final-batch trials of all experiments,
    # plot quad of their weight distributions
    # convert to numpy arrays
    fns_coh0_ee = np.array(fns_coh0)[:,:e_end,:e_end].flatten()
    fns_coh0_ei = np.array(fns_coh0)[:,:e_end,e_end:].flatten()
    fns_coh0_ie = np.array(fns_coh0)[:,e_end:,:e_end].flatten()
    fns_coh0_ii = np.array(fns_coh0)[:,e_end:,e_end:].flatten()
    fns_coh1_ee = np.array(fns_coh1)[:,:e_end,:e_end].flatten()
    fns_coh1_ei = np.array(fns_coh1)[:,:e_end,e_end:].flatten()
    fns_coh1_ie = np.array(fns_coh1)[:,e_end:,:e_end].flatten()
    fns_coh1_ii = np.array(fns_coh1)[:,e_end:,e_end:].flatten()

    # plot quad for just coherence 0 to begin with
    fig, ax = plt.subplots(nrows=1,ncols=2)
    ax[0].hist(fns_coh0_ee[fns_coh0_ee!=0],bins=30,alpha=0.5,density=True,color='dodgerblue')
    ax[0].hist(fns_coh0_ei[fns_coh0_ei!=0],bins=30,alpha=0.5,density=True,color='seagreen')
    ax[0].hist(fns_coh0_ie[fns_coh0_ie!=0],bins=30,alpha=0.5,density=True,color='darkorange')
    ax[0].hist(fns_coh0_ii[fns_coh0_ii!=0],bins=30,alpha=0.5,density=True,color='orangered')
    ax[0].legend(['ee','ei','ie','ii'])
    ax[0].set_title('Coherence label 0',fontname='Ubuntu')

    ax[1].hist(fns_coh1_ee[fns_coh1_ee!=0],bins=30,alpha=0.5,color='dodgerblue',density=True)
    ax[1].hist(fns_coh1_ei[fns_coh1_ei!=0],bins=30,alpha=0.5,color='seagreen',density=True)
    ax[1].hist(fns_coh1_ie[fns_coh1_ie!=0],bins=30,alpha=0.5,color='darkorange',density=True)
    ax[1].hist(fns_coh1_ii[fns_coh1_ii!=0],bins=30,alpha=0.5,color='orangered',density=True)
    ax[1].legend(['ee','ei','ie','ii'])
    ax[1].set_title('Coherence label 1',fontname='Ubuntu')

    plt.suptitle('Trained functional weights',fontname='Ubuntu')

    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # go through and set all axes
    for i in range(0,len(ax)):
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        ax[i].set_xlabel('functional weight values',fontname='Ubuntu')
        ax[i].set_ylabel('density',fontname='Ubuntu')

    plt.draw()

    save_fname = fn_dir+'coherence_separate_trained_rate_fn_weights.png'
    plt.savefig(save_fname,dpi=300)


def plot_delay_rn_measures(rn_dir='/data/results/experiment1/spring_fns/21.06.01/trained/'):
    # by measures, we mostly mean density and recurrence for now
    # load in rns
    rn_files = os.listdir(rn_dir)
    conn_types = ['e->e','e->i','i->e','i->i']
    for fname in rn_files:
        if 'rns.npz' in fname:
            trial_idx = int(fname.split("_")[1])
            fig, ax = plt.subplots(nrows=2,ncols=2)

            data = np.load(rn_dir+fname,allow_pickle=True)
            rns = data['rns']
            # plot separately for all 4 types of connections
            rn_ee = rns[:,:241,:241]
            rn_ei = rns[:,:241,241:]
            rn_ie = rns[:,241:,:241]
            rn_ii = rns[:,241:,241:]

            timesteps = np.shape(rn_ee)[0]
            density = np.zeros([4,timesteps])
            #recurrence = np.zeros([4,timesteps])
            for i in range(0,timesteps-1):
                density[:,i] = [calc_density(rn_ee[i]),calc_density(rn_ei[i]),calc_density(rn_ie[i]),calc_density(rn_ii[i])]
                #recurrence[:,timesteps] = [,,,]

            ax = ax.flatten()
            for i in range(0,len(ax)):
                ax[i].plot(density[i])
                ax[i].set_xlabel('time starting 250ms before change',fontname='Ubuntu')
                ax[i].set_ylabel('recruitment graph density',fontname='Ubuntu')
                ax[i].set_title(conn_types[i])
                for tick in ax[i].get_xticklabels():
                    tick.set_fontname('Ubuntu')
                for tick in ax[i].get_yticklabels():
                    tick.set_fontname('Ubuntu')

            plt.suptitle('recruitment graph density, trial '+str(trial_idx),fontname='Ubuntu')
            # draw and save plot
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            plt.draw()
            save_fname = rn_dir+'trial'+str(trial_idx)+'_rn_density.png'
            plt.savefig(save_fname,dpi=300)

            # Teardown
            plt.clf()
            plt.close()

def plot_single_batch_delays(fpath,spath):
    trained_data = np.load(fpath)
    true_y = trained_data['true_y'][99]
    pred_y = trained_data['pred_y'][99]
    delay_durs = []
    change_times = []
    change_ys = []
    change_preds = []
    trial_idx = []

    exp_str = fpath.split("/")[5]

    for i in range(0,len(true_y)):
        # check to see if there is a change in this trial
        if true_y[i][0] != true_y[i][seq_len-1]:
            trial_idx.append(i)
            change_ys.append(true_y[i])
            change_preds.append(pred_y[i])
            # find time of change
            diffs = np.diff(true_y[i],axis=0)
            # t_change is the first timestep of the new coherence level
            t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
            change_times.append(t_change)
            # find average pred before and after
            pre_avg = np.average(pred_y[i][:t_change])
            post_avg = np.average(pred_y[i][t_change:])
            # determine the duration after coherence change until we first pass (pos or neg direction) the after-change average
            if pre_avg < post_avg:
                # if we are increasing coherence level, crossing is when we go above the 75th percentile of post-change preds
                t_crossing = np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0][0]
            elif pre_avg > post_avg:
                # if we are decreasing coherence level, crossing is when we fall below the 25th percentile of post-change preds
                t_crossing = np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0][0]
            # append
            delay_durs.append(t_crossing)

    # now plot all trials separately if they don't exist yet
    for i in range(0,len(delay_durs)):
        save_fname = spath+'trial'+str(trial_idx[i])+'_delays.png'
        if not os.path.isfile(save_fname):
            plt.plot(change_preds[i],color='dodgerblue',alpha=0.5)
            plt.plot(change_ys[i],color='mediumblue')
            plt.vlines(change_times[i],ymin=np.max(change_preds[i]),ymax=np.min(change_preds[i]),color='red')
            plt.vlines(change_times[i]+delay_durs[i],ymin=np.max(change_preds[i]),ymax=np.min(change_preds[i]),color='darkorange')
            plt.xlabel('time (ms)',fontname='Ubuntu')
            plt.ylabel('output',fontname='Ubuntu')

            plt.title('trial '+str(trial_idx[i]),fontname='Ubuntu')
            plt.xticks(fontname='Ubuntu')
            plt.yticks(fontname='Ubuntu')
            plt.legend(['pred y','true y','time of change','delay'],prop={"family":"Ubuntu"})

            plt.draw()
            plt.savefig(save_fname,dpi=300)

            # Teardown
            plt.clf()
            plt.close()

    return [change_times,delay_durs]


def determine_delays(exp_dirs=spec_input_dirs,exp_season='winter'):
    # principled way to decide what constitutes the duration of a delay between coherence changes
    # plot to see delay markers on a few trials
    # get the experiments you want to begin with
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    all_delays = []

    # go through the experiments and files
    for xdir in exp_data_dirs:
        exp_path = xdir[-9:-1]

        np_dir = os.path.join(data_dir, xdir, "npz-data")
        #naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        # go through final epoch trials
        true_y = trained_data['true_y'][99]
        pred_y = trained_data['pred_y'][99]
        delay_durs = []
        change_times = []
        change_ys = []
        change_preds = []

        for i in range(0,len(true_y)):
            # check to see if there is a change in this trial
            if true_y[i][0] != true_y[i][seq_len-1]:
                change_ys.append(true_y[i])
                change_preds.append(pred_y[i])
                # find time of change
                diffs = np.diff(true_y[i],axis=0)
                # t_change is the first timestep of the new coherence level
                t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
                change_times.append(t_change)
                # find average pred before and after
                pre_avg = np.average(pred_y[i][:t_change])
                post_avg = np.average(pred_y[i][t_change:])
                # determine the duration after coherence change until we first pass (pos or neg direction) the after-change average
                if pre_avg < post_avg:
                    # if we are increasing coherence level, crossing is when we go above the 75th percentile of post-change preds
                    t_crossing = np.where(pred_y[i][t_change:]>np.quantile(pred_y[i][t_change:],0.75))[0][0]
                elif pre_avg > post_avg:
                    # if we are decreasing coherence level, crossing is when we fall below the 25th percentile of post-change preds
                    t_crossing = np.where(pred_y[i][t_change:]<np.quantile(pred_y[i][t_change:],0.25))[0][0]
                # append
                delay_durs.append(t_crossing)
                all_delays.append(t_crossing)

        # take average duration as The Delay
        delay = np.average(delay_durs)

        # select a few trials and plot randomly with The Delay to visually inspect
        fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(8,10))
        ax = ax.flatten()
        trials = np.random.choice(np.arange(0,len(delay_durs)),size=[4],replace=False)
        for i in range(0,len(trials)):
            ax[i].plot(change_preds[trials[i]],color='dodgerblue',alpha=0.4)
            ax[i].plot(change_ys[trials[i]],color='mediumblue')
            ax[i].vlines(change_times[trials[i]],ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='red')
            ax[i].vlines(change_times[trials[i]]+delay,ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='orangered')
            ax[i].vlines(change_times[trials[i]]+delay_durs[trials[i]],ymin=np.max(change_preds[trials[i]]),ymax=np.min(change_preds[trials[i]]),color='darkorange')
            ax[i].set_xlabel('time (ms)',fontname='Ubuntu')
            ax[i].set_ylabel('output',fontname='Ubuntu')

            ax[i].set_title('trial '+str(trials[i]),fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname('Ubuntu')
            for tick in ax[i].get_yticklabels():
                tick.set_fontname('Ubuntu')

        ax[3].legend(['pred y','true y','time of change','avg delay','trial delay'],prop={"family":"Ubuntu"})

        plt.suptitle('sample trained trials with change delays',fontname='Ubuntu')
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        plt.draw()

        save_fname = savepath+'/set_plots/'+exp_season+'/'+str(exp_path)+'_delay_trials_test.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

    # plot the distribution of delays
    plt.hist(all_delays,bins=30)
    plt.xlabel('delay duration (ms)',fontname='Ubuntu')
    plt.ylabel('count',fontname='Ubuntu')
    plt.title('all trained delay durations for '+exp_season,fontname='Ubuntu')
    """
    for tick in ax.get_xticklabels():
        tick.set_fontname('Ubuntu')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Ubuntu')"""
    plt.draw()
    save_fname = savepath+'/set_plots/'+exp_season+'/delay_dist_test.png'
    plt.savefig(save_fname,dpi=300)
    # Teardown
    plt.clf()
    plt.close()

#def delay_MI_gen():
    # determine delays of all experiments
    # only look at the delay period of trained experiments
    # generate based on each batch of 30 trials
    # first look at just the final batch, anyway
    # generate separately based on pre-delay (coh 0 or 1 separate), delay (to coh 0 or to coh 1 separate), post-delay (coh 0 or 1 separate)
    # generate ummmmmmmmm recruitment graphs for that time span?
    # also allowed to generate general MI graphs and then only dynamically examine recruitment graphs?


#def plot_in_out_rates(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # plot the rates for input-receiving and output-giving units for coherence 0 and


def plot_all_rates(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # plot separately for coherence 0 and 1 trials
    # honestly don't even worry about the changes for now
    # that is for tmr

    fig, ax = plt.subplots(nrows=2,ncols=2)

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # go through all dirs and grab the weight distributions of the first and last epochs
    data_files = filenames(num_epochs, epochs_per_file)

    coh0_e_rates = []
    coh0_i_rates = []
    coh1_e_rates = []
    coh1_i_rates = []

    # the units that receive input
    coh0_e_in_rates = []
    coh1_e_in_rates = []
    coh0_i_in_rates = []
    coh1_i_in_rates = []
    # the units that send output
    coh0_e_out_rates = []
    coh1_e_out_rates = []
    coh0_i_out_rates = []
    coh1_i_out_rates = []

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))

        spikes = naive_data['spikes'][0]
        true_y = naive_data['true_y'][0]
        input = naive_data['tv0.postweights'][0]
        output = naive_data['tv2.postweights'][0]

        # find units that receive input and those that project to output
        in_idx = []
        out_idx = []
        # for each recurrent unit
        for i in range(0,np.shape(input)[1]):
            if len(np.nonzero(input[:,i]))>0:
                in_idx.append(i)
            if len(np.nonzero(output[i,:]))>0:
                out_idx.append(i)

        in_idx = np.array(in_idx)
        out_idx = np.array(out_idx)

        for i in range(0,len(true_y)):
            if true_y[i][0]==true_y[i][seq_len-1]:
                if true_y[i][0]==1:
                    # reminder that spikes are shaped [batch (100), trial (30), time (4080), neuron (300)]
                    coh1_e_rates.append(np.mean(spikes[i][:,:e_end],axis=0)) # average across time for each neuron, so axis 0
                    coh1_i_rates.append(np.mean(spikes[i][:,:e_end],axis=0))
                    coh1_e_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx<=e_end]],axis=0))
                    coh1_i_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx>e_end]],axis=0))
                    coh1_e_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx<=e_end]],axis=0))
                    coh1_i_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx>e_end]],axis=0))
                else:
                    coh0_e_rates.append(np.mean(spikes[i][:,e_end:],axis=0))
                    coh0_i_rates.append(np.mean(spikes[i][:,e_end:],axis=0))
                    coh0_e_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx<=e_end]],axis=0))
                    coh0_i_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx>e_end]],axis=0))
                    coh0_e_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx<=e_end]],axis=0))
                    coh0_i_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx>e_end]],axis=0))
            """
            else:
                # find time of coherence change
                diffs = np.diff(true_y[i],axis=0)
                # t_change is the first timestep of the new coherence level
                t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
                # find average rates before and after
                if true_y[i][0]==1:
                    coh1_e_rates.append(np.mean(spikes[i][:t_change,:e_end],axis=0))
                    coh0_e_rates.append(np.mean(spikes[i][t_change:,:e_end],axis=0))
                    coh1_i_rates.append(np.mean(spikes[i][:t_change,e_end:],axis=0))
                    coh0_i_rates.append(np.mean(spikes[i][t_change:,e_end:],axis=0))
                else:
                    coh0_e_rates.append(np.mean(spikes[i][:t_change,:e_end],axis=0))
                    coh1_e_rates.append(np.mean(spikes[i][t_change:,:e_end],axis=0))
                    coh0_i_rates.append(np.mean(spikes[i][:t_change,e_end:],axis=0))
                    coh1_i_rates.append(np.mean(spikes[i][t_change:,e_end:],axis=0))
                """
    # plot for naive
    ax[0,0].hist(np.array(coh0_e_rates).flatten(),bins=30,alpha=0.4,density=True,color='dodgerblue',label='all, naive')
    #ax[0,0].hist(np.array(coh0_e_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='mediumblue',label='in-receiving, naive')
    ax[0,0].hist(np.array(coh0_e_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='teal',label='out-sending, naive')
    ax[0,0].set_title('coherence 0 excitatory')
    ax[0,1].hist(np.array(coh0_i_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkorange',label='all, naive')
    #ax[0,1].hist(np.array(coh0_i_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='orangered',label='in-receiving, naive')
    ax[0,1].hist(np.array(coh0_i_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='greenyellow',label='out-sending, naive')
    ax[0,1].set_title('coherence 0 inhibitory')
    ax[1,0].hist(np.array(coh1_e_rates).flatten(),bins=30,alpha=0.4,density=True,color='dodgerblue',label='all, naive')
    #ax[1,0].hist(np.array(coh1_e_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='mediumblue',label='in-receiving, naive')
    ax[1,0].hist(np.array(coh1_e_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='teal',label='out-sending, naive')
    ax[1,0].set_title('coherence 1 excitatory')
    ax[1,1].hist(np.array(coh1_i_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkorange',label='all, naive')
    #ax[1,1].hist(np.array(coh1_i_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='orangered',label='in-receiving, naive')
    ax[1,1].hist(np.array(coh1_i_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='greenyellow',label='out-sending, naive')
    ax[1,1].set_title('coherence 1 inhibitory')

    # repeat for trained
    coh0_e_rates = []
    coh0_i_rates = []
    coh1_e_rates = []
    coh1_i_rates = []
    # the units that receive input
    coh0_e_in_rates = []
    coh1_e_in_rates = []
    coh0_i_in_rates = []
    coh1_i_in_rates = []
    # the units that send output
    coh0_e_out_rates = []
    coh1_e_out_rates = []
    coh0_i_out_rates = []
    coh1_i_out_rates = []

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        spikes = trained_data['spikes'][99]
        true_y = trained_data['true_y'][99]
        input = trained_data['tv0.postweights'][99]
        output = trained_data['tv2.postweights'][99]

        in_idx = []
        out_idx = []
        # for each recurrent unit
        for i in range(0,np.shape(input)[1]):
            if len(np.nonzero(input[:,i]))>0:
                in_idx.append(i)
            if len(np.nonzero(output[i,:]))>0:
                out_idx.append(i)

        in_idx = np.array(in_idx)
        out_idx = np.array(out_idx)

        for i in range(0,len(true_y)):
            if true_y[i][0]==true_y[i][seq_len-1]:
                if true_y[i][0]==1:
                    coh1_e_rates.append(np.average(spikes[i][:,:e_end]))
                    coh1_i_rates.append(np.average(spikes[i][:,:e_end]))
                    coh1_e_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx<=e_end]],axis=0))
                    coh1_i_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx>e_end]],axis=0))
                    coh1_e_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx<=e_end]],axis=0))
                    coh1_i_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx>e_end]],axis=0))
                else:
                    coh0_e_rates.append(np.average(spikes[i][:,e_end:]))
                    coh0_i_rates.append(np.average(spikes[i][:,e_end:]))
                    coh0_e_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx<=e_end]],axis=0))
                    coh0_i_in_rates.append(np.mean(spikes[i][:,in_idx[in_idx>e_end]],axis=0))
                    coh0_e_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx<=e_end]],axis=0))
                    coh0_i_out_rates.append(np.mean(spikes[i][:,out_idx[out_idx>e_end]],axis=0))
            """
            else:
                # find time of coherence change
                diffs = np.diff(true_y[i],axis=0)
                # t_change is the first timestep of the new coherence level
                t_change = np.where(np.diff(true_y[i],axis=0)!=0)[0][0]+1
                # find average rates before and after
                if true_y[i][0]==1:
                    coh1_e_rates.append(np.average(spikes[i][:t_change,:e_end]))
                    coh0_e_rates.append(np.average(spikes[i][t_change:,:e_end]))
                    coh1_i_rates.append(np.average(spikes[i][:t_change,e_end:]))
                    coh0_i_rates.append(np.average(spikes[i][t_change:,e_end:]))
                else:
                    coh0_e_rates.append(np.average(spikes[i][:t_change,:e_end]))
                    coh1_e_rates.append(np.average(spikes[i][t_change:,:e_end]))
                    coh0_i_rates.append(np.average(spikes[i][:t_change,e_end:]))
                    coh1_i_rates.append(np.average(spikes[i][t_change:,e_end:]))
                """

    # plot all together
    ax[0,0].hist(np.array(coh0_e_rates).flatten(),bins=30,alpha=0.4,density=True,color='purple',label='all, trained')
    #ax[0,0].hist(np.array(coh0_e_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkslateblue',label='in-receiving, trained')
    ax[0,0].hist(np.array(coh0_e_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='indigo',label='out-sending, trained')
    ax[0,0].set_title('coherence 0 excitatory')
    ax[0,1].hist(np.array(coh0_i_rates).flatten(),bins=30,alpha=0.4,density=True,color='mediumvioletred',label='all, trained')
    #ax[0,1].hist(np.array(coh0_i_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkviolet',label='in-receiving, trained')
    ax[0,1].hist(np.array(coh0_i_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='crimson',label='out-sending, trained')
    ax[0,1].set_title('coherence 0 inhibitory')
    ax[1,0].hist(np.array(coh1_e_rates).flatten(),bins=30,alpha=0.4,density=True,color='purple',label='all, trained')
    #ax[1,0].hist(np.array(coh1_e_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkslateblue',label='in-receiving, trained')
    ax[1,0].hist(np.array(coh1_e_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='indigo',label='out-sending, trained')
    ax[1,0].set_title('coherence 1 excitatory')
    ax[1,0].legend()
    ax[1,1].hist(np.array(coh1_i_rates).flatten(),bins=30,alpha=0.4,density=True,color='mediumvioletred',label='all, trained')
    #ax[1,1].hist(np.array(coh1_i_in_rates).flatten(),bins=30,alpha=0.4,density=True,color='darkviolet',label='in-receiving, trained')
    ax[1,1].hist(np.array(coh1_i_out_rates).flatten(),bins=30,alpha=0.4,density=True,color='crimson',label='out-sending, trained')
    ax[1,1].set_title('coherence 1 inhibitory')
    ax[1,1].legend()

    plt.suptitle('rates of all recurrent units',fontname='Ubuntu')

    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # go through and set all axes
    ax = ax.flatten()
    for i in range(0,len(ax)):
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        ax[i].set_xlabel('unitwise rate (Hz)',fontname='Ubuntu')
        ax[i].set_ylabel('density',fontname='Ubuntu')

    plt.draw()

    save_fname = savepath+'/set_plots/'+exp_season+'_inoutrec_rates_test.png'
    plt.savefig(save_fname,dpi=300)


# well, now you need to go and fix the input weights


def plot_weight_delta_dists(exp_dirs=spec_nointoout_dirs_rate,exp_season='spring'): # just for dual-training for now
    fig, ax = plt.subplots(nrows=3,ncols=1)

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])
    # go through all dirs and grab the weight distributions of the first and last epochs
    data_files = filenames(num_epochs, epochs_per_file) # useful for plotting evolution over the entire course of training
    in_naive = []
    in_trained = []
    rec_naive = []
    rec_trained = []
    out_naive = []
    out_trained = []
    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        in_naive.append(naive_data['tv0.postweights'][0])
        in_trained.append(trained_data['tv0.postweights'][99])

        rec_naive.append(naive_data['tv1.postweights'][0])
        rec_trained.append(trained_data['tv1.postweights'][99])

        out_naive.append(naive_data['tv2.postweights'][0])
        out_trained.append(trained_data['tv2.postweights'][99])

    # convert to numpy arrays
    in_naive = np.array(in_naive)
    in_trained = np.array(in_trained)
    rec_naive = np.array(rec_naive)
    rec_trained = np.array(rec_trained)
    out_naive = np.array(out_naive)
    out_trained = np.array(out_trained)

    # plot ee, ei, ie and ii separately, and only nonzero weight values

    in_naive = in_naive.flatten()
    in_trained = in_trained.flatten()
    in_delta = in_trained-in_naive
    ax[0].hist(in_delta[in_delta!=0],alpha=0.7,bins=30,density=True,color='dodgerblue')
    #ax[0].hist(in_trained-in_naive[in_trained-in_naive!=0],bins=30,density=True,color='darkorange')
    ax[0].legend(['e edges','i edges'])
    ax[0].set_title('input weights',fontname='Ubuntu')

    # plot layers separately
    rec_naive_ee = rec_naive[:,:e_end,:e_end].flatten()
    rec_naive_ei = rec_naive[:,:e_end,e_end:].flatten()
    rec_naive_ie = rec_naive[:,e_end:,:e_end].flatten()
    rec_naive_ii = rec_naive[:,e_end:,e_end:].flatten()
    rec_trained_ee = rec_trained[:,:e_end,:e_end].flatten()
    rec_trained_ei = rec_trained[:,:e_end,e_end:].flatten()
    rec_trained_ie = rec_trained[:,e_end:,:e_end].flatten()
    rec_trained_ii = rec_trained[:,e_end:,e_end:].flatten()

    ee_delta = rec_trained_ee-rec_naive_ee
    ei_delta = rec_trained_ei-rec_naive_ei
    ie_delta = rec_trained_ie-rec_naive_ie
    ii_delta = rec_trained_ii-rec_naive_ii

    ax[1].hist(ee_delta[ee_delta!=0],bins=30,alpha=0.7,color='dodgerblue',density=True)
    ax[1].hist(ei_delta[ei_delta!=0],bins=30,alpha=0.7,color='seagreen',density=True)
    ax[1].hist(ie_delta[ie_delta!=0],bins=30,alpha=0.7,color='darkorange',density=True)
    ax[1].hist(ii_delta[ii_delta!=0],bins=30,alpha=0.7,color='orangered',density=True)
    ax[1].legend(['ee','ei','ie','ii'])
    ax[1].set_title('recurrent weights',fontname='Ubuntu')

    out_naive_e = out_naive[:,0:e_end].flatten()
    out_trained_e = out_trained[:,0:e_end].flatten()
    out_naive_i = out_naive[:,e_end:i_end].flatten()
    out_trained_i = out_trained[:,e_end:i_end].flatten()
    eo_delta = out_trained_e-out_naive_e
    io_delta = out_trained_i-out_naive_i
    ax[2].hist(eo_delta[eo_delta!=0],alpha=0.7,bins=30,color='dodgerblue',density=True)
    ax[2].hist(io_delta[io_delta!=0],alpha=0.7,bins=30,color='darkorange',density=True)
    ax[2].set_title('output weights',fontname='Ubuntu')
    ax[2].legend(['e edges','i edges'])

    plt.suptitle('delta of weights after rate training',fontname='Ubuntu')

    # go through and set all axes
    ax = ax.flatten()
    for i in range(0,len(ax)):
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        ax[i].set_xlabel('nonzero weight change trained - naive',fontname='Ubuntu')
        ax[i].set_ylabel('density',fontname='Ubuntu')

    plt.draw()
    plt.subplots_adjust(wspace=0.7,hspace=1.0)

    save_fname = savepath+'/set_plots/'+exp_season+'_quad_rate_weight_deltas_test.png'
    plt.savefig(save_fname,dpi=300)


def plot_all_weight_dists(exp_dirs=spec_nointoout_dirs_rate,exp_season='spring'): # just for dual-training for now
    fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(8,8))

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])
    # go through all dirs and grab the weight distributions of the first and last epochs
    data_files = filenames(num_epochs, epochs_per_file) # useful for plotting evolution over the entire course of training
    in_naive = []
    in_trained = []
    rec_naive = []
    rec_trained = []
    out_naive = []
    out_trained = []
    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        naive_data = np.load(os.path.join(np_dir, "1-10.npz"))
        trained_data = np.load(os.path.join(np_dir, "991-1000.npz"))

        in_naive.append(naive_data['tv0.postweights'][0])
        in_trained.append(trained_data['tv0.postweights'][99])

        rec_naive.append(naive_data['tv1.postweights'][0])
        rec_trained.append(trained_data['tv1.postweights'][99])

        out_naive.append(naive_data['tv2.postweights'][0])
        out_trained.append(trained_data['tv2.postweights'][99])

    # convert to numpy arrays
    in_naive = np.array(in_naive)
    in_trained = np.array(in_trained)
    rec_naive = np.array(rec_naive)
    rec_trained = np.array(rec_trained)
    out_naive = np.array(out_naive)
    out_trained = np.array(out_trained)

    # plot ee, ei, ie and ii separately, and only nonzero weight values

    in_naive = in_naive.flatten()
    ax[0,0].hist(in_naive[in_naive>0],bins=30,density=True,color='dodgerblue')
    #ax[0,0].hist(in_naive[in_naive<0],bins=30,density=True,color='darkorange')
    #ax[0,0].legend(['e edges','i edges'])
    ax[0,0].set_title('naive input weights',fontname='Ubuntu')

    in_trained = in_trained.flatten()
    ax[0,1].hist(in_trained[in_trained>0],bins=30,color='dodgerblue',density=True)
    ax[0,1].hist(in_trained[in_trained<0],bins=30,color='darkorange',density=True)
    #ax[0,1].legend(['e edges','i edges'])
    ax[0,1].set_title('trained input weights',fontname='Ubuntu')

    # plot layers separately
    rec_naive_ee = rec_naive[:,:e_end,:e_end].flatten()
    rec_naive_ei = rec_naive[:,:e_end,e_end:].flatten()
    rec_naive_ie = rec_naive[:,e_end:,:e_end].flatten()
    rec_naive_ii = rec_naive[:,e_end:,e_end:].flatten()
    ax[1,0].hist(rec_naive_ee[rec_naive_ee>0],bins=30,alpha=0.7,color='dodgerblue',density=True)
    ax[1,0].hist(rec_naive_ei[rec_naive_ei>0],bins=30,alpha=0.7,color='seagreen',density=True)
    ax[1,0].hist(rec_naive_ie[rec_naive_ie<0],bins=30,alpha=0.7,color='darkorange',density=True)
    ax[1,0].hist(rec_naive_ii[rec_naive_ii<0],bins=30,alpha=0.7,color='orangered',density=True)
    ax[1,0].legend(['ee','ei','ie','ii'])
    ax[1,0].set_title('naive recurrent weights',fontname='Ubuntu')

    rec_trained_ee = rec_trained[:,:e_end,:e_end].flatten()
    rec_trained_ei = rec_trained[:,:e_end,e_end:].flatten()
    rec_trained_ie = rec_trained[:,e_end:,:e_end].flatten()
    rec_trained_ii = rec_trained[:,e_end:,e_end:].flatten()
    ax[1,1].hist(rec_trained_ee[rec_trained_ee>0],bins=30,alpha=0.7,color='dodgerblue',density=True)
    ax[1,1].hist(rec_trained_ei[rec_trained_ei>0],bins=30,alpha=0.7,color='seagreen',density=True)
    ax[1,1].hist(rec_trained_ie[rec_trained_ie<0],bins=30,alpha=0.7,color='darkorange',density=True)
    ax[1,1].hist(rec_trained_ii[rec_trained_ii<0],bins=30,alpha=0.7,color='orangered',density=True)
    #ax[1,0].legend(['ee','ei','ie','ii'])
    ax[1,1].set_title('trained recurrent weights',fontname='Ubuntu')

    out_naive_e = out_naive[:,0:e_end].flatten()
    ax[2,0].hist(out_naive_e[out_naive_e>0],bins=30,color='dodgerblue',density=True)
    out_naive_i = out_naive[:,e_end:i_end].flatten()
    ax[2,0].hist(out_naive_i[out_naive_i<0],bins=30,color='darkorange',density=True)
    #ax[2,0].legend(['e edges','i edges'])
    ax[2,0].set_title('naive output weights',fontname='Ubuntu')

    out_trained_e = out_trained[:,0:e_end].flatten()
    ax[2,1].hist(out_trained_e[out_trained_e>0],bins=30,color='dodgerblue',density=True)
    out_trained_i = out_trained[:,e_end:i_end].flatten()
    ax[2,1].hist(out_trained_i[out_trained_i<0],bins=30,color='darkorange',density=True)
    ax[2,1].legend(['e edges','i edges'])
    ax[2,1].set_title('trained output weights',fontname='Ubuntu')

    plt.suptitle('rate trained with no direct in-to-out units',fontname='Ubuntu')

    plt.subplots_adjust(wspace=0.4, hspace=0.7)

    # go through and set all axes
    ax = ax.flatten()
    for i in range(0,len(ax)):
        for tick in ax[i].get_xticklabels():
            tick.set_fontname("Ubuntu")
        for tick in ax[i].get_yticklabels():
            tick.set_fontname("Ubuntu")
        ax[i].set_xlabel('weight values',fontname='Ubuntu')
        ax[i].set_ylabel('density',fontname='Ubuntu')

    plt.draw()

    save_fname = savepath+'/set_plots/'+exp_season+'_quad_rate_weights_test.png'
    plt.savefig(save_fname,dpi=300)


def plot_input_receiving_rates(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # plot, over all of training time, the evolution of the average firing rates of each of the subpopulations that receive direct input channel connections
    # separately for the two coherence levels
    # and maybe also separately for e and i units
    # do so only for no-coherence-change trials
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    # eventually plot something that is specific to each experiment
    # but otherwise 16 separate input channels populations, rates over all of the trial for a coherence level, over all of training
    # so ultimately x axis training epoch, y axis rate, 16 separate channels, maybe e and i separate and coherence level separate
    # go thru a single experiment to verify that your array shapes are working correctly haha

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        # sized 16 x batch
        e_coh0_rates = np.zeros([16,10000])
        e_coh1_rates = np.zeros([16,10000])
        i_coh0_rates = np.zeros([16,10000])
        i_coh1_rates = np.zeros([16,10000])

        # loop through all experiments
        for filename in data_files:
            start_idx = (int(filename.split('-')[0])-1)*10
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            in_w = data['tv0.postweights']
            #w = data['tv1.postweights']
            true_y = data['true_y']
            spikes = data['spikes']
            # go through all 100 batches
            for i in range(0,np.shape(in_w)[0]):
                # find which units receive each channel's input
                input_pop_indices = []
                for input_idx in range(0,np.shape(in_w)[1]): # for 16 input channels, get indices of their projection populations
                    single_channel_indices = np.argwhere(in_w[i][input_idx]!=0).flatten()
                    input_pop_indices.append(single_channel_indices)

                # go through all 30 trials
                all_rates = []
                cohs = []
                for j in range(0,np.shape(true_y)[1]):
                    # check if no change trial
                    if true_y[i][j][0]==true_y[i][j][seq_len-1]:
                        if true_y[i][j][0]==0: # if coherence 0 trial
                            cohs.append(0)
                        else:
                            cohs.append(1)
                        all_rates.append(np.average(spikes[i][j],0)) # avg firing rates for all 300 units

                all_rates=np.array(all_rates)
                cohs=np.array(cohs)
                coh0_idx = np.where(cohs==0)[0]
                coh1_idx = np.where(cohs==1)[0]

                # go through all 16 input channels
                for input_idx in range(0,np.shape(in_w)[1]):
                    interm = all_rates[coh0_idx] # all trials with coherence 0
                    e_coh0_rates[input_idx,i+start_idx] = np.mean(interm[:,input_pop_indices[input_idx][input_pop_indices[input_idx]<e_end]])
                    i_coh0_rates[input_idx,i+start_idx] = np.mean(interm[:,input_pop_indices[input_idx][input_pop_indices[input_idx]>=e_end]])
                    interm = all_rates[coh1_idx] # all trials with coherence 1
                    e_coh1_rates[input_idx,i+start_idx] = np.mean(interm[:,input_pop_indices[input_idx][input_pop_indices[input_idx]<e_end]])
                    i_coh1_rates[input_idx,i+start_idx] = np.mean(interm[:,input_pop_indices[input_idx][input_pop_indices[input_idx]>=e_end]])

        # plot for each experiment
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].plot(np.transpose(e_coh0_rates))
        ax[0,0].set_title('coherence 0 excitatory',fontname='Ubuntu')
        ax[0,1].plot(np.transpose(e_coh1_rates))
        ax[0,1].set_title('coherence 1 excitatory',fontname='Ubuntu')
        ax[1,0].plot(np.transpose(i_coh0_rates))
        ax[1,0].set_title('coherence 0 inhibitory',fontname='Ubuntu')
        ax[1,1].plot(np.transpose(i_coh1_rates))
        ax[1,1].set_title('coherence 1 inhibitory',fontname='Ubuntu')

        ax = ax.flatten()
        for i in range(0,len(ax)):
            ax[i].set_xlabel('training batch',fontname='Ubuntu')
            ax[i].set_ylabel('average rate',fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Average rates of 16 input-receiving populations')

        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        save_fname = savepath+'/set_plots/'+exp_season+'/'+exp_path+'_input_receiving_rates.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


# plot rates by input receiving group for delay period [1 to 0] and [0 to 1] (so a 4 x 4 with e and i units separate)
# plot rates by input receiving group for failure to 1 when 0 and to 0 when 1
# plot rates by input receiving group for correct 0 and correct 1

# naive states as well?

# generate functional graphs for the above as well?

def plot_group_input_receiving_rates(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # plot, over all of training time, the evolution of the average firing rates of each of the subpopulations that receive direct input channel connections
    # separately for the two coherence levels
    # and maybe also separately for e and i units
    # do so only for no-coherence-change trials

    # determine which coherence level the input units prefer based on original CNN output file
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

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        e_coh0_rates = np.zeros([2,10000]) # rates in response to coh0 input, first row is units that receive more coh0-dom inputs, second is units that receive more coh1-dom inputs
        e_coh1_rates = np.zeros([2,10000])
        i_coh0_rates = np.zeros([2,10000])
        i_coh1_rates = np.zeros([2,10000])

        # find which units received more of each channel duo's input weights
        filepath = os.path.join(data_dir,xdir,'npz-data','991-1000.npz')
        data = np.load(filepath)
        in_w = data['tv0.postweights'][99]
        coh0_units = np.where(in_w[coh0_idx,:]>in_w[coh1_idx,:])[1]
        coh1_units = np.where(in_w[coh1_idx,:]>in_w[coh0_idx,:])[1]

        # loop through all experiments
        for filename in data_files:
            start_idx = (int(filename.split('-')[0])-1)*10
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            in_w = data['tv0.postweights']
            true_y = data['true_y']
            spikes = data['spikes']
            for i in range(0,np.shape(in_w)[0]):
                # go through all 30 trials
                all_rates = []
                cohs = []
                for j in range(0,np.shape(true_y)[1]):
                    # check if no change trial
                    if true_y[i][j][0]==true_y[i][j][seq_len-1]:
                        if true_y[i][j][0]==0: # if coherence 0 trial
                            cohs.append(0)
                        else:
                            cohs.append(1)
                        all_rates.append(np.average(spikes[i][j],0)) # avg firing rates for all 300 units

                all_rates=np.array(all_rates)
                cohs=np.array(cohs)
                coh0_trials = np.where(cohs==0)[0]
                coh1_trials = np.where(cohs==1)[0]

                # get all types of unit groups' responses to coherence 0 trials
                interm = all_rates[coh0_trials]
                # e units that receive more input from coh0-driven channels
                e_coh0_rates[0,i+start_idx] = np.mean(interm[:,coh0_units[coh0_units<e_end]])
                # e units that receive more input from coh1-driven channels
                e_coh0_rates[1,i+start_idx] = np.mean(interm[:,coh1_units[coh1_units<e_end]])
                # i units that receive more input from coh0-driven channels
                i_coh0_rates[0,i+start_idx] = np.mean(interm[:,coh0_units[coh0_units>=e_end]])
                # i units that receive more input from coh1-driven channels
                i_coh0_rates[1,i+start_idx] = np.mean(interm[:,coh1_units[coh1_units>=e_end]])

                # now responses to all coherence 1 trials
                interm = all_rates[coh1_trials]
                e_coh1_rates[0,i+start_idx] = np.mean(interm[:,coh0_units[coh0_units<e_end]])
                # e units that receive more input from coh1-driven channels
                e_coh1_rates[1,i+start_idx] = np.mean(interm[:,coh1_units[coh1_units<e_end]])
                # i units that receive more input from coh0-driven channels
                i_coh1_rates[0,i+start_idx] = np.mean(interm[:,coh0_units[coh0_units>=e_end]])
                # i units that receive more input from coh1-driven channels
                i_coh1_rates[1,i+start_idx] = np.mean(interm[:,coh1_units[coh1_units>=e_end]])

        # plot
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].plot(np.transpose(e_coh0_rates))
        ax[0,0].plot(np.transpose(i_coh0_rates))
        ax[0,0].set_title('coherence 0 trials',fontname='Ubuntu')
        ax[0,0].legend(['0-driven e','1-driven e','0-driven i','1-driven i'])

        ax[0,1].plot(np.transpose(e_coh1_rates))
        ax[0,1].plot(np.transpose(i_coh1_rates))
        ax[0,1].set_title('coherence 1 trials',fontname='Ubuntu')
        ax[0,1].legend(['0-driven e','1-driven e','0-driven i','1-driven i'])

        ax[1,0].plot(np.mean(e_coh0_rates,0))
        ax[1,0].plot(np.mean(i_coh0_rates,0))
        ax[1,0].set_title('coherence 0 trials',fontname='Ubuntu')
        ax[1,0].legend(['all e','all i'])

        ax[1,1].plot(np.mean(e_coh1_rates,0))
        ax[1,1].plot(np.mean(i_coh1_rates,0))
        ax[1,1].set_title('coherence 1 trials',fontname='Ubuntu')
        ax[1,1].legend(['all e','all i'])

        ax = ax.flatten()
        for i in range(0,len(ax)):
            ax[i].set_xlabel('training epoch',fontname='Ubuntu')
            ax[i].set_ylabel('average rate',fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Average rates of recurrent subpopulations')

        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        save_fname = savepath+'/set_plots/'+exp_season+'/'+exp_path+'_input_receiving_rates_grouped.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

def plot_input_grouped_rec_weights(exp_dirs=spec_nointoout_dirs,exp_season='spring'):

    # determine which coherence level the input units prefer based on original CNN output file
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

    data_files = filenames(num_epochs, epochs_per_file)

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # eventually plot something that is specific to each experiment
    # but otherwise 16 separate input channels populations, rates over all of the trial for a coherence level, over all of training
    # so ultimately x axis training epoch, y axis rate, 16 separate channels, maybe e and i separate and coherence level separate
    # go thru a single experiment to verify that your array shapes are working correctly haha

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        filepath = os.path.join(np_dir, '991-1000.npz')
        data = np.load(filepath)
        w = data['tv1.postweights'][99]
        in_w = data['tv0.postweights'][99]

        # find recurrent units that receive input primarily from one of the populations
        coh0_units = np.where(np.sum(in_w[coh0_idx,:],0)>np.sum(in_w[coh1_idx,:],0))[0]
        coh1_units = np.where(np.sum(in_w[coh1_idx,:],0)>np.sum(in_w[coh0_idx,:],0))[0]

        coh0_w = np.zeros([4,10000]) # average outgoing weights of units that receive more coh0-dom inputs
        coh1_w = np.zeros([4,10000]) # average outgoing weights of units that receive more coh1-dom inputs

        coh0_e = coh0_units[coh0_units<e_end]
        coh0_i = coh0_units[coh0_units>=e_end]
        coh1_e = coh1_units[coh1_units<e_end]
        coh1_i = coh1_units[coh1_units>=e_end]

        for filename in data_files:
            start_idx = (int(filename.split('-')[0])-1)*10
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            in_w = data['tv0.postweights']
            w = data['tv1.postweights']
            for i in range(0,np.shape(w)[0]): # for each batch
                coh0_w[0,i+start_idx] = np.mean(w[i][coh0_e,:][:,coh0_e])
                coh0_w[1,i+start_idx] = np.mean(w[i][coh0_e,:][:,coh0_i])
                coh0_w[2,i+start_idx] = np.mean(w[i][coh0_i,:][:,coh0_e])
                coh0_w[3,i+start_idx] = np.mean(w[i][coh0_i,:][:,coh0_i])

                coh1_w[0,i+start_idx] = np.mean(w[i][coh1_e,:][:,coh1_e])
                coh1_w[1,i+start_idx] = np.mean(w[i][coh1_e,:][:,coh1_i])
                coh1_w[2,i+start_idx] = np.mean(w[i][coh1_i,:][:,coh1_e])
                coh1_w[3,i+start_idx] = np.mean(w[i][coh1_i,:][:,coh1_i])

        # plot
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(coh0_w[0,:],color='dodgerblue')
        ax[0].plot(coh0_w[1,:],color='seagreen')
        ax[0].plot(coh0_w[2,:],color='darkorange')
        ax[0].plot(coh0_w[3,:],color='orangered')
        ax[0].set_title('0-driven units',fontname='Ubuntu')

        ax[1].plot(coh1_w[0,:],color='dodgerblue')
        ax[1].plot(coh1_w[1,:],color='seagreen')
        ax[1].plot(coh1_w[2,:],color='darkorange')
        ax[1].plot(coh1_w[3,:],color='orangered')
        ax[1].set_title('1-driven units',fontname='Ubuntu')

        """ax[0].hist(w[coh0_e,:][:,coh0_e].flatten(),bins=22,density=True)
        ax[0].hist(w[coh0_e,:][:,coh0_i].flatten(),bins=22,density=True)
        ax[0].hist(w[coh0_i,:][:,coh0_e].flatten(),bins=22,density=True)
        ax[0].hist(w[coh0_i,:][:,coh0_i].flatten(),bins=22,density=True)
        ax[0].set_title('0-driven units',fontname='Ubuntu')

        ax[1].hist(w[coh1_e,:][:,coh1_e].flatten(),bins=22,density=True)
        ax[1].hist(w[coh1_e,:][:,coh1_i].flatten(),bins=22,density=True)
        ax[1].hist(w[coh1_i,:][:,coh1_e].flatten(),bins=22,density=True)
        ax[1].hist(w[coh1_i,:][:,coh1_i].flatten(),bins=22,density=True)
        ax[1].set_title('1-driven units',fontname='Ubuntu')"""

        for i in range(0,len(ax)):
            ax[i].set_xlabel('training epochs',fontname='Ubuntu')
            ax[i].set_ylabel('average weights',fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")
            ax[i].legend(['ee','ei','ie','ii'],prop={"family":"Ubuntu"})

        plt.suptitle('Recurrent weights')

        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        save_fname = savepath+'/set_plots/'+exp_season+'/'+exp_path+'_input_receiving_weights_grouped_overtime.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


# measure within-input-channel-receiving clustering vs outside / all

def plot_output_sending_rates(exp_dirs=spec_nointoout_dirs,exp_season='spring'):
    # same as above function, but for the subpopulation of units that actually project to output
    # plot relative to rates of the non-projecting population
    # also separately for two coherence levels and for e and i units
    # do so only for no-coherence-change trials
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    # eventually plot something that is specific to each experiment
    # but otherwise 16 separate input channels populations, rates over all of the trial for a coherence level, over all of training
    # so ultimately x axis training epoch, y axis rate, 16 separate channels, maybe e and i separate and coherence level separate
    # go thru a single experiment to verify that your array shapes are working correctly haha

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        # sized 16 x batch
        e_coh0_rates = np.zeros([2,10000])
        e_coh1_rates = np.zeros([2,10000])
        i_coh0_rates = np.zeros([2,10000])
        i_coh1_rates = np.zeros([2,10000])

        # loop through all experiments
        for filename in data_files:
            start_idx = (int(filename.split('-')[0])-1)*10
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            out_w = data['tv2.postweights']
            true_y = data['true_y']
            spikes = data['spikes']
            # go through all 100 batches
            for i in range(0,np.shape(out_w)[0]):
                # find which units project to output, and which do not
                out_pop_indices = np.where(out_w[i]!=0)[0]
                not_pop_indices = np.where(out_w[i]==0)[0]

                # go through all 30 trials
                all_rates = []
                cohs = []
                for j in range(0,np.shape(true_y)[1]):
                    # check if no change trial
                    if true_y[i][j][0]==true_y[i][j][seq_len-1]:
                        if true_y[i][j][0]==0: # if coherence 0 trial
                            cohs.append(0)
                        else:
                            cohs.append(1)
                        all_rates.append(np.average(spikes[i][j],0)) # avg firing rates for all 300 units

                all_rates=np.array(all_rates)
                cohs=np.array(cohs)
                coh0_idx = np.where(cohs==0)[0]
                coh1_idx = np.where(cohs==1)[0]

                # go through output projecting and not projecting units
                interm = all_rates[coh0_idx] # all trials with coherence 0
                e_coh0_rates[0,i+start_idx] = np.mean(interm[:,out_pop_indices[out_pop_indices<e_end]])
                i_coh0_rates[0,i+start_idx] = np.mean(interm[:,out_pop_indices[out_pop_indices>=e_end]])
                e_coh0_rates[1,i+start_idx] = np.mean(interm[:,not_pop_indices[not_pop_indices<e_end]])
                i_coh0_rates[1,i+start_idx] = np.mean(interm[:,not_pop_indices[not_pop_indices>=e_end]])
                interm = all_rates[coh1_idx] # all trials with coherence 1
                e_coh1_rates[0,i+start_idx] = np.mean(interm[:,out_pop_indices[out_pop_indices<e_end]])
                i_coh1_rates[0,i+start_idx] = np.mean(interm[:,out_pop_indices[out_pop_indices>=e_end]])
                e_coh1_rates[1,i+start_idx] = np.mean(interm[:,not_pop_indices[not_pop_indices<e_end]])
                i_coh1_rates[1,i+start_idx] = np.mean(interm[:,not_pop_indices[not_pop_indices>=e_end]])

        # plot for each experiment
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0,0].plot(np.transpose(e_coh0_rates))
        ax[0,0].set_title('coherence 0 excitatory',fontname='Ubuntu')
        ax[0,1].plot(np.transpose(e_coh1_rates))
        ax[0,1].set_title('coherence 1 excitatory',fontname='Ubuntu')
        ax[1,0].plot(np.transpose(i_coh0_rates))
        ax[1,0].set_title('coherence 0 inhibitory',fontname='Ubuntu')
        ax[1,1].plot(np.transpose(i_coh1_rates))
        ax[1,1].set_title('coherence 1 inhibitory',fontname='Ubuntu')

        ax = ax.flatten()
        for i in range(0,len(ax)):
            ax[i].set_xlabel('training batch',fontname='Ubuntu')
            ax[i].set_ylabel('average rate',fontname='Ubuntu')
            ax[i].legend(['output projecting','not projecting'],prop={"family":"Ubuntu"})
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        plt.suptitle('Average rates of output/not-projecting populations')

        # Draw and save
        plt.draw()
        plt.subplots_adjust(wspace=0.4, hspace=0.7)
        save_fname = savepath+'/set_plots/'+exp_season+'/'+exp_path+'_output_sending_rates.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

def input_channel_indiv_weight_changes(exp_dirs=save_inz_dirs):
    # plot the average input connection strength from two populatons of
    # input channels (according to average rate) for the two coherence levels
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        epoch_loss = []
        rate_loss = []

        # loop through all training time for this experiment
        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_w = data['tv0.postweights'][0]
            loss = data['step_task_loss']
            rate_losses = data['step_rate_loss']
            #for i in range(0,np.shape(input_w)[0]): # 100 trials
            # weights of each type to e units and to i units
            # oh, maybe I should do this without zEros. well, later.

            if not 'input_to_e' in locals():
                input_to_e = np.mean(input_w[:,:e_end],1)
            else:
                input_to_e = np.vstack([input_to_e,np.mean(input_w[:,:e_end],1)])

            if not 'input_to_i' in locals():
                input_to_i = np.mean(input_w[:,e_end:],1)
            else:
                input_to_i = np.vstack([input_to_i,np.mean(input_w[:,e_end:],1)])

            epoch_loss.append(np.mean(loss))
            rate_loss.append(np.mean(rate_losses))

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(input_to_e)
        ax[0].set_ylim([0,0.9])
        ax[0].set_title('input weights to excitatory units',fontname='Ubuntu')
        ax[1].plot(input_to_i)
        ax[1].set_ylim([0,0.9])
        ax[1].set_title('input weights to inhibitory units',fontname='Ubuntu')
        ax[2].plot(epoch_loss)
        ax[2].plot(rate_loss)
        ax[2].legend(['task loss','rate loss'],prop={"family":"Ubuntu"})
        ax[2].set_title('losses')

        for i in range(0,len(ax)):
            ax[i].set_xlabel('training epoch',fontname='Ubuntu')
            ax[i].set_ylabel('average weights',fontname='Ubuntu')
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        ax[2].set_ylabel('task loss')

        plt.suptitle('Evolution of input weights over training')
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        plt.draw()

        save_fname = savepath+'/set_plots/spring/'+str(exp_path)+'_input_weights_over_time.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()

        del input_to_e
        del input_to_i


def input_channel_ratewise_weight_changes_fromCNN(exp_dirs=spec_input_dirs,exp_season='winter'):
    # determine which coherence level the input units prefer based on original CNN output file
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

    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        coh1_e = []
        coh1_i = []
        coh0_e = []
        coh0_i = []
        epoch_task_loss = []
        epoch_rate_loss = []

        # get the truly naive weights
        filepath = os.path.join(data_dir,xdir,"npz-data","input_preweights.npy")
        input_w = np.load(filepath)
        coh1_e.append(np.mean(input_w[coh1_idx,:e_end]))
        coh1_i.append(np.mean(input_w[coh1_idx,e_end:]))
        coh0_e.append(np.mean(input_w[coh0_idx,:e_end]))
        coh0_i.append(np.mean(input_w[coh0_idx,e_end:]))

        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_w = data['tv0.postweights'][0]
            epoch_task_loss.append(np.mean(data['step_task_loss']))
            epoch_rate_loss.append(np.mean(data['step_rate_loss']))
            #for i in range(0,np.shape(input_w)[0]): # 100 trials
            # weights of each type to e units and to i units
            coh1_e.append(np.mean(input_w[coh1_idx,:e_end]))
            coh1_i.append(np.mean(input_w[coh1_idx,e_end:]))
            coh0_e.append(np.mean(input_w[coh0_idx,:e_end]))
            coh0_i.append(np.mean(input_w[coh0_idx,e_end:]))

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(coh1_e)
        ax[0].plot(coh0_e)
        ax[0].set_title('input weights to excitatory units',fontname='Ubuntu')
        ax[1].plot(coh1_i)
        ax[1].plot(coh0_i)
        ax[1].set_title('input weights to inhibitory units',fontname='Ubuntu')

        for i in range(0,len(ax)):
            ax[i].set_xlabel('training epoch',fontname='Ubuntu')
            ax[i].set_ylabel('average weights',fontname='Ubuntu')
            ax[i].legend(['coh 1 preferring inputs','coh 0 preferring inputs'],prop={"family":"Ubuntu"})
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        ax[2].plot(epoch_task_loss)
        ax[2].plot(epoch_rate_loss)
        ax[2].set_ylabel('loss',fontname='Ubuntu')
        ax[2].legend(['task loss','rate loss'],prop={"family":"Ubuntu"})

        plt.suptitle('Evolution of input weights over training')
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        plt.draw()

        save_fname = savepath+'/set_plots/'+exp_season+'/'+str(exp_path)+'_inputs_to_ei.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


def input_channel_ratewise_weight_changes(exp_dirs=save_inz_dirs,exp_season='spring'):
    # plot the average input connection strength from two populatons of
    # input channels (according to average rate) for the two coherence levels
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string)
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string)])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file)

    for xdir in exp_data_dirs: # loop through experiments
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        # loop through all training time for this experiment
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

        coh1_e = []
        coh1_i = []
        coh0_e = []
        coh0_i = []
        epoch_task_loss = []
        epoch_rate_loss = []

        # get the truly naive weights
        filepath = os.path.join(data_dir,xdir,"npz-data","input_preweights.npy")
        input_w = np.load(filepath)
        coh1_e.append(np.mean(input_w[coh1_idx,:e_end]))
        coh1_i.append(np.mean(input_w[coh1_idx,e_end:]))
        coh0_e.append(np.mean(input_w[coh0_idx,:e_end]))
        coh0_i.append(np.mean(input_w[coh0_idx,e_end:]))

        # now do weights over time
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_w = data['tv0.postweights'][0]
            epoch_task_loss.append(np.mean(data['step_task_loss']))
            epoch_rate_loss.append(np.mean(data['step_rate_loss']))
            #for i in range(0,np.shape(input_w)[0]): # 100 trials
            # weights of each type to e units and to i units
            coh1_e.append(np.mean(input_w[coh1_idx,:e_end]))
            coh1_i.append(np.mean(input_w[coh1_idx,e_end:]))
            coh0_e.append(np.mean(input_w[coh0_idx,:e_end]))
            coh0_i.append(np.mean(input_w[coh0_idx,e_end:]))

        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].plot(coh1_e)
        ax[0].plot(coh0_e)
        ax[0].set_title('input weights to excitatory units',fontname='Ubuntu')
        ax[1].plot(coh1_i)
        ax[1].plot(coh0_i)
        ax[1].set_title('input weights to inhibitory units',fontname='Ubuntu')

        for i in range(0,len(ax)):
            ax[i].set_xlabel('training epoch',fontname='Ubuntu')
            ax[i].set_ylabel('average weights',fontname='Ubuntu')
            ax[i].legend(['coh 1 preferring inputs','coh 0 preferring inputs'],prop={"family":"Ubuntu"})
            for tick in ax[i].get_xticklabels():
                tick.set_fontname("Ubuntu")
            for tick in ax[i].get_yticklabels():
                tick.set_fontname("Ubuntu")

        ax[2].plot(epoch_task_loss)
        ax[2].plot(epoch_rate_loss)
        ax[2].set_ylabel('loss',fontname='Ubuntu')
        ax[2].legend(['task loss','rate loss'],prop={"family":"Ubuntu"})

        plt.suptitle('Evolution of input weights over training')
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        plt.draw()

        save_fname = savepath+'/set_plots/'+exp_season+'/'+str(exp_path)+'_inputs_to_ei.png'
        plt.savefig(save_fname,dpi=300)

        # Teardown
        plt.clf()
        plt.close()


def plot_input_channel_rates(from_CNN=False,exp_dirs=save_inz_dirs):
    # NEED TO BE CAREFUL BASED ON MIXING OF COHERENCE SWAP / UNSWAP LABELS
    # from_CNN means the original output rates from the CNN that are used to generate Poisson spikes actually
    """
    if from_CNN:
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

        #coh0_rates = np.average(coh0_channel_trial_rates,0)
        #coh1_rates = np.average(coh1_channel_trial_rates,0)
    """
    #if not from_CNN:
    for exp_string in exp_dirs:
        if not 'exp_data_dirs' in locals():
            exp_data_dirs = get_experiments(data_dir, exp_string, final_npz='591-600.npz')
        else:
            exp_data_dirs = np.hstack([exp_data_dirs,get_experiments(data_dir, exp_string, final_npz='591-600.npz')])

    # aggregate across all experiments and all trials
    data_files = filenames(num_epochs, epochs_per_file, final_npz='591-600.npz')

    for xdir in exp_data_dirs:
        np_dir = os.path.join(data_dir, xdir, "npz-data")
        exp_path = xdir[-9:-1]

        # loop through all experiments
        for filename in data_files:
            filepath = os.path.join(data_dir, xdir, "npz-data", filename)
            data = np.load(filepath)
            input_z = data['inputs']
            # shaped [100 batches x 30 trials x 4080 timesteps x 16 units]
            true_y = data['true_y']
            # shaped [100 batches x 30 trials x 4080 timesteps]
            for i in range(0,np.shape(true_y)[0]):
                # for each of 100 batches
                for j in range(0,np.shape(true_y)[1]):
                    coh0_idx = np.where(true_y[i][j]==0)[0]
                    coh1_idx = np.where(true_y[i][j]==1)[0]
                    # take average rates across that trial's timepoints for the same coherence level and append
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

    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].hist(coh0_channel_trial_rates,bins=30,histtype='step', density=True, stacked=True)
    ax[0].set_title('coherence 0', fontname="Ubuntu")
    ax[0].set_xlabel('spike rate (Hz)', fontname="Ubuntu")
    ax[0].set_ylabel('density', fontname="Ubuntu")
    ax[0].set_ylim([0,6])
    ax[1].hist(coh1_channel_trial_rates,bins=30,histtype='step', density=True, stacked=True)
    ax[1].set_title('coherence 1', fontname="Ubuntu")
    ax[1].set_xlabel('spike rate (Hz)', fontname="Ubuntu")
    ax[1].set_ylabel('density', fontname="Ubuntu")
    ax[1].set_ylim([0,6])
    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Ubuntu")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Ubuntu")
    #ax[1,0].hist(late_in[i,:],bins=50,histtype='step')
    #ax[1,1].hist(trained_in[i,:],bins=50,histtype='step')

    plt.suptitle("Rates of 16 input channels based on experiment", fontname="Ubuntu")

    # Draw and save
    plt.draw()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    save_fname = savepath+'/set_plots/input_rates_exp_final.png'
    plt.savefig(save_fname,dpi=300)

    # Teardown
    plt.clf()
    plt.close()

    #return [coh0_rates,coh1_rates]
