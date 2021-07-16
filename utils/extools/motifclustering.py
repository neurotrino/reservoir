# process logger npz files and calculate triplet motif clustering
# first on synaptic graph across epochs,
# then ultimately on inferred functional and recruitment graphs

import sys
#sys.path.append("snn-infrastructure/")
from motifclustering import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import distplot as distplot
import h5py
import os
"""
data=np.load('1-5.npz')
for k in data.files:
    print(k)
"""
#experiment = 'ccd_with_dot_trained_cnn_alif_3'
#experiment = 'ccd_with_dot_trained_cnn_alif_2'
experiment = 'ccd_with_dot_trained_cnn_lif'

def test_local():
	sims = 10
	mu = -0.64
	sigma = 0.51
	dist = [mu, sigma]
	epochs_per_npz = 10
	props = []
	cc = []
	dens = []
	input_dens = []
	epoch_end = [9,19,29,39,49,59,69,79,89,99]
	#epochwise_w_rec = []
	#epochwise_w_in = []
	epoch_groups = ['1-10','191-200']
	nfiles = len(epoch_groups)

	for i in range(nfiles):
		fname = 'ccd_data/alif/npz-data/' + epoch_groups[i] + '.npz'
		data = np.load(fname)
		for j in range(len(epoch_end)):
			w_rec = data['tv1.postweights'][:,:,epoch_end[j]]
			# calculate propensity for this epoch
			propensities = w_motif_propensity(w_rec,sims,dist)
			props.append(propensities)
			# calculate clustering coefficients for this epoch
			coefs = motifs_cc(w_rec)
			cc.append(coefs)

			# record recurrent weight density
			total_ct = np.size(w_rec)
			zero_ct = w_rec[w_rec==0].shape[0]
			dens.append((total_ct - zero_ct)/float(total_ct))
			#epochwise_w_rec.append(w_rec)

	# create plot of beginning and ending weight distributions
	start_fname = 'ccd_data/alif/npz-data/' + epoch_groups[0] + '.npz'
	start_data = np.load(fname)
	start_w_rec = data['tv1.postweights'][:,:,0]
	end_fname = 'ccd_data/alif/npz-data/' + epoch_groups[len(epoch_groups)-1] + '.npz'
	end_data = np.load(fname)
	end_w_rec = data['tv1.postweights'][:,:,99]

	fig, ax = plt.subplots(4, figsize=(6, 8))
	fig.suptitle("ALIF SNN")

	begin_w_dist = start_w_rec.flatten()
	end_w_dist = end_w_rec.flatten()
	sns.kdeplot(begin_w_dist, bw = 0.02, ax = ax[0], label='initial')
	sns.kdeplot(end_w_dist, bw=0.02, ax = ax[0], label = 'end')
	#ax[0].hist(start_w_rec.flatten(), bins=50, fc = '#1f77b4', label="initial")
	#ax[0].hist(end_w_rec.flatten(), bins=50, fc = '#ff7f0e', label="end")
	ax[0].set_xlabel("weights")
	ax[0].set_ylabel("counts")
	ax[0].legend(prop={'size': 7})
	ax[0].set_title("recurrent weight distribution")

	"""
	# plot input weight distribution
	begin_w_in = start_w_in.flatten()
	end_w_in = start_w_in.flatten()
	ax[1].set_xlabel("weights")
	ax[1].set_ylabel("counts")
	sns.kdeplot(begin_w_in, bw = 0.02, ax = ax[1], label='initial')
	sns.kdeplot(end_w_in, bw=0.02, ax = ax[1], label = 'end')
	ax[1].legend(prop={'size': 5})
	ax[1].set_title("input weight distribution")
	"""

	# create plot of density over time
	ax[1].plot(dens)
	ax[1].set_title('proportion of total recurrent connectivity')
	ax[1].set_xlabel("epoch")
	ax[1].set_ylabel("density")

	# process clustering and create plot of clustering over time
	mean_props = []
	mean_cc = []
	for i in range(nfiles):
		mean_props.append(np.nanmean(props[i],axis=1))
		mean_cc.append(np.nanmean(cc[i],axis=1))
	ax[2].plot(mean_cc)
	ax[2].set_xlabel("epoch")
	ax[2].set_ylabel("clustering coefficient")
	ax[2].set_title("clustering coefficient")
	ax[2].legend(['fanin','fanout','middleman'], prop={'size': 7})

	ax[3].plot(mean_props)
	ax[3].set_xlabel("epoch")
	ax[3].set_ylabel("propensity")
	ax[3].set_title("clustering propensity")
	ax[3].legend(['fanin','fanout','middleman'], prop={'size': 7})

	# draw and save plot
	fig.subplots_adjust(hspace=1)
	plt.draw()
	plt.savefig("ccd_data/alif/output_figure.png", dpi=300)

	return props


def plot_data_over_epochs(experiment): # March 23, 2021: creating plots for Graz meeting
    sims = 10
    mu = -0.64
    sigma = 0.51
    dist = [mu, sigma]
    props = []
    cc = []
    dens = []
    input_dens = []
    loss = []
    epoch_end = [9,19,29,39,49,59,69,79,89,99]
    epoch_groups = ['1-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120','121-130','131-140','141-150']#,'151-160','161-170','171-180','181-190','191-200']
    nfiles = len(epoch_groups)
    for i in range(nfiles):
        fname = '../experiments/' + experiment + '/npz-data/' + epoch_groups[i] + '.npz'
        data = np.load(fname)
        for j in range(len(epoch_end)):
            w_rec = data['tv1.postweights'][epoch_end[j],:,:]
            # calculate propensity for this epoch
            propensities = w_motif_propensity(w_rec,sims,dist)
            props.append(propensities)
            # calculate clustering coefficients for this epoch
            coefs = motifs_cc(w_rec)
            cc.append(coefs)

            # record recurrent weight density
            total_ct = np.size(w_rec)
            zero_ct = w_rec[w_rec==0].shape[0]
            dens.append((total_ct - zero_ct)/float(total_ct))
            #epochwise_w_rec.append(w_rec)

        # record loss
        for j in range(len(data['epoch_loss'])):
            loss.append(data['epoch_loss'][j])

    # create plot of beginning and ending weight distributions
    start_fname = '../experiments/' + experiment + '/npz-data/' + epoch_groups[0] + '.npz'
    start_data = np.load(start_fname)
    start_w_rec = data['tv1.postweights'][0,:,:]
    end_fname = '../experiments/' + experiment + '/npz-data/' + epoch_groups[len(epoch_groups)-1] + '.npz'
    end_data = np.load(end_fname)
    end_w_rec = data['tv1.postweights'][99,:,:]

    fig, ax = plt.subplots(5, figsize=(6, 8))
    fig.suptitle("ALIF SNN")

    begin_w_dist = start_w_rec.flatten()
    end_w_dist = end_w_rec.flatten()
    sns.kdeplot(begin_w_dist, bw = 0.02, ax = ax[0], label='initial')
    sns.kdeplot(end_w_dist, bw=0.02, ax = ax[0], label = 'end')
    #ax[0].hist(start_w_rec.flatten(), bins=50, fc = '#1f77b4', label="initial")
    #ax[0].hist(end_w_rec.flatten(), bins=50, fc = '#ff7f0e', label="end")
    ax[0].set_xlabel("weights")
    ax[0].set_ylabel("counts")
    ax[0].legend(prop={'size': 7})
    ax[0].set_title("recurrent weight distribution")

    # create plot of loss over time
    ax[1].plot(loss)
    ax[1].set_title('total loss per epoch')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('MSE loss')

    ax[2].plot(dens)
    ax[2].set_title('proportion of total recurrent connectivity')
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("density")

    # process clustering and create plot of clustering over time
    mean_props = []
    mean_cc = []
    for i in range(len(props)):
        mean_props.append(np.nanmean(props[i],axis=1))
        mean_cc.append(np.nanmean(cc[i],axis=1))

    ax[3].plot(mean_cc)
    ax[3].set_xlabel("epoch")
    ax[3].set_ylabel("clustering coefficient")
    ax[3].set_title("clustering coefficient")
    ax[3].legend(['fanin','fanout','middleman'], prop={'size': 7})

    ax[4].plot(mean_props)
    ax[4].set_xlabel("epoch")
    ax[4].set_ylabel("propensity")
    ax[4].set_title("clustering propensity")
    ax[4].legend(['fanin','fanout','middleman'], prop={'size': 7})

    # draw and save plot
    fig.subplots_adjust(hspace=1)
    plt.draw()
    plt.savefig('../experiments/' + experiment + "/analysis/output_figure_to_160.png", dpi=300)



def motif_propensity_over_epochs(): # edge-normalized, controls for density in rewiring conditions
	n_epochs = 30
	sims = 10
	mu = -0.64
	sigma = 0.51
	dist = [mu, sigma]
	motifs = []
	for epoch in range(n_epochs):
		rec_w = process_hdf5(epoch)
		propensities = w_motif_propensity(rec_w,sims,dist)
		motifs.append(propensities)
	# save file with motif values
	motifs = np.array(motifs)
	#np.save("minimal_sparsity_data/v2/gen_data/set1_motifs.npy",motifs)
	# create plot of motifs over time
	return motifs

def motif_cc_over_epochs():
	n_epochs = 30
	motifs = []
	for epoch in range(n_epochs):
		rec_w = process_hdf5(epoch)
		ccs = motifs_cc(rec_w)
		motifs.append(ccs)
	# save file with motif values
	motifs = np.array(motifs)
	# create plot of motifs over time
	return motifs

def w_motif_propensity(rec_w,sims,dist):
	cc = motifs_cc(rec_w)
	middle_mean = 0
	fanin_mean = 0
	fanout_mean = 0
	for i in range(sims):
		sample_mat = np.copy(rec_w)
		# find all nonzero edges in the rec_w
		indices = np.where(rec_w > 0)
		for j in indices[0]:
			# randomly populate each of the non-zero synapses with a new value from the same distribution
			sample_mat[indices[0][j],indices[1][j]] = np.random.lognormal(dist[0], dist[1])
		samp_cc = motifs_cc(sample_mat)
		fanin_mean = fanin_mean + samp_cc[0]
		fanout_mean = fanout_mean + samp_cc[1]
		middle_mean = middle_mean + samp_cc[2]

	# after the loop completes, we have a vector of summed CCs for each of the units in rec_w
	# we average each vector's values by the number of sims we ran, and then normalize the actual CC of the corresponding units by those values.
	fanin = cc[0]/(fanin_mean/sims)
	fanout = cc[1]/(fanout_mean/sims)
	middle = cc[2]/(middle_mean/sims)
	propensity = [fanin, fanout, middle]
	return propensity

def motifs_cc(rec_w):
	W = np.transpose(rec_w) # Wij convention in Fagiolo is the transpose of ours
	temp = np.cbrt(W)

	dim = rec_w.shape[1]
	d_in = np.zeros(dim)
	d_out = np.zeros(dim)
	d_bi = np.zeros(dim)
	for i in range(dim):
		in_nodes = np.where(W[:,i]!=0)
		out_nodes = np.where(W[i,:]!=0)
		bi_nodes = np.intersect1d(in_nodes, out_nodes)
		d_in[i] = np.size(in_nodes)
		d_out[i] = np.size(out_nodes)
		d_bi[i] = np.size(bi_nodes)

	d_tot = d_in + d_out #element-wise addition, multiplication, subtraction
	denom = np.multiply(d_in, d_out) - d_bi

	fanin_CC  = np.divide(np.diag(np.matmul(np.transpose(temp),np.square(temp))), np.multiply(d_in, d_in-1))
	fanout_CC  = np.divide(np.diag(np.matmul(np.square(temp),np.transpose(temp))), np.multiply(d_out, d_out-1))
	middleman_CC = np.divide(np.diag(np.matmul(np.matmul(temp,np.transpose(temp)),temp)), denom)
	CC = [fanin_CC, fanout_CC, middleman_CC]
	return CC

def process_hdf5(epoch):
	hf = h5py.File("minimal_sparsity_data/v2/logs/set4/end_epoch_" + str(epoch) + ".hdf5",'r')
	n1 = hf.get('rnn')
	n2 = n1.get('rnn')
	lif_cell = n2.get('alif_ei')
	in_w = lif_cell.get('input_weights:0')
	in_w = np.array(in_w)
	rec_w = lif_cell.get('recurrent_weights:0')
	rec_w = np.array(rec_w)
	return rec_w
