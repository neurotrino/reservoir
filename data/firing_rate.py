from tensorflow import keras as keras
import tensorflow as tf
import os
import argparse
import numpy as np
import scipy.sparse

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

#MODEL_PATH = cwd + 'models/'
MODEL_PATH = '/home/macleanlab/front-end-CNN/models/'
#IMG_PATH = cwd + 'plots/'
#MOVIE_PATH = '/home/macleanlab/mufeng/tfrecord_data_processing/'
MOVIE_PATH = '/home/macleanlab/stim/processed/' # where all the tfrecords are located
# in total there are processed_data_299.tfrecord files
"""
parser = argparse.ArgumentParser(description='Args for generating the spikes')
parser.add_argument('model_name', default='ch_model8', type=str, help='Name of the model')
parser.add_argument('num_dot_movies', default=60, type=int, help='Number of dot movies to check') # all of them??
# readme says "The maximum number of dot coh change movies is 60", though likely there are more tfrecords on .226
parser.add_argument('make_positive', default='abs', type=str, help='How do you want to make firing rates positive, rec or abs')
parser.add_argument('--num_natural_movies', default=20, type=int, help='Number of natural movies to check')
args = parser.parse_args()
model_name = args.model_name
num_dot_movies = args.num_dot_movies
num_natural_movies = args.num_natural_movies
makepositive = args.make_positive
"""

FRAMES_PER_TRIAL = 240
FRAMERATE = 60
MS_PER_TRIAL = (FRAMES_PER_TRIAL // FRAMERATE) * 1000

model_name = 'ch_model8'
num_dot_movies = 60
make_positive = 'abs'

model = tf.keras.models.load_model(MODEL_PATH+model_name)

def create_dataset(paths, max_seq_len=4800, encoding='png', pool=None):
    """
    Read in .tfrecord datasets of the drifting dots movies
    """
    feature_description = {
        'frames': tf.io.VarLenFeature(tf.string),
        'change_label': tf.io.VarLenFeature(tf.int64),
        'coherence_label': tf.io.VarLenFeature(tf.int64),
        'direction_label': tf.io.VarLenFeature(tf.int64),
        'dominant_direction': tf.io.VarLenFeature(tf.int64),
        'trial_coherence': tf.io.VarLenFeature(tf.int64),
        'is_changes': tf.io.VarLenFeature(tf.int64),
    }
    data_set = tf.data.TFRecordDataset(paths)

    def _parse_example(_x):
        _x = tf.io.parse_single_example(_x, feature_description)

        if encoding == 'png':
            _frames = tf.map_fn(lambda a: tf.io.decode_png(a), _x['frames'].values, dtype=tf.uint8)[:max_seq_len]

        _label = _x['coherence_label'].values
        _change_label = _x['change_label'].values
        _direction_label = _x['direction_label'].values
        _dominant_direction = _x['dominant_direction'].values
        _trial_coherence = _x['trial_coherence'].values
        _is_changes = _x['is_changes'].values

        return _frames, dict(tf_op_layer_coherence=_label, # 0 or 1, length 4800
                             tf_op_layer_change=_change_label, # 0 or 1, length 4800
                             tf_op_dir=_direction_label, # 1,2,3,4, length 4800
                             tf_dom_dir=_dominant_direction, # 1,2,3,4, length 10 (i.e. per trial/grey screen)
                             tf_trial_coh=_trial_coherence, # 0 or 1, length 10
                             tf_is_changes=_is_changes,
                            )

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set

def read_dot(num_movies):
    """
    Select a random subset of drifting dots movies
    num_movies: number of 4800 frames-long drifting dots movies to select. Each contains 10 trials and 10 grey screens
    """
    # each tfrecord file corresponds to one movie (4800 frames)
    # file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(num_movies)]
    file_names = [MOVIE_PATH + f'processed_data_{i+1}.tfrecord' for i in range(num_movies)]
    data_set = create_dataset(file_names, 4800).batch(1)

    # ex[0] the frames (tensor), shape [1, 4800, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    movies = []
    trial_cohs = [] # original coherence level for each trial
    coh_labels = [] # coherence level at each frame
    is_changes = [] # whether coherence changes for each trial
    for ex in data_set:
        trials = []
        print("Movie ", k)

        trial_coh = ex[1]['tf_trial_coh'].numpy()[0] # len = 10, coh vector of this movie
        trial_cohs.append(trial_coh)
        coh_label = ex[1]['tf_op_layer_coherence'].numpy()[0] # len = 4800, for each frame.
        coh_labels.append(coh_label)
        is_change = ex[1]['tf_is_changes'].numpy()[0]
        is_changes.append(is_change)

        start_frames = np.arange(0, 80, 4)
        end_frames = np.arange(4, 84, 4)
        framerate = 60
        for i in range(2*len(trial_coh)): # 20
            trial = ex[0][:, start_frames[i]*framerate:end_frames[i]*framerate] # [1, 240, 36, 64, 3]
            trials.append(trial)

        # concatenate the trials and grey screens into a large tensor
        movie = tf.concat(trials, axis=0) # [20, 240, 36, 64, 3]
        movies.append(movie)
        k+=1

    all_movies = tf.concat(movies, axis=0) # [num_movies*20,240,36,64,3]
    all_trial_cohs = np.stack(trial_cohs) # [num_movies, 10]
    all_coh_labels = np.stack(coh_labels) # [num_movies, 4800]
    all_is_changes = np.stack(is_changes) # [num_movies, 10]

    return all_movies, all_trial_cohs, all_coh_labels, all_is_changes

def generate_mean_firingrates(all_movies, makepositive):
    # except, since we want, we will want to shuffle and regenerate a TON more of these
    # in total, we have 10 trials x 10 batches x 500 epochs = 50000 unique trials (spike trains) per full experiment
    # though we do not want ALL data sampled for every experiment / batch / epoch whatever.
    # so this is less of a pipeline thing and more of a generating a lot more data thing.
    # the crucial thing is to disentangle the precise firing rates from exact (static) spike trains each time.
    """

    Generate spikes based on firing rates
        Input: (num_dot_movies*20, 240, 36, 64, 3)
        Output: (num_dot_movies*10, 4080, 16), binary

    """
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                           outputs=model.layers[12].output)
    # load the data into the model in batches
    batch_size = 2
    dataset = tf.data.Dataset.from_tensor_slices(all_movies).batch(batch_size)
    batch_rates = []
    for d in dataset:
        output = intermediate_layer_model(d) # batch_size, 60, 16
        if makepositive == 'rec':
            batch_rate = tf.nn.relu(output).numpy() # batch_size, 60, 16
        elif makepositive == 'abs':
            batch_rate = tf.math.abs(output).numpy() # batch_size, 60, 16
        batch_rates.append(batch_rate)
    f_rates = np.concatenate(batch_rates, axis=0) # 20*num_dot_movies, 60, 16
    f_rates = f_rates[::2] # 10*num_dot_movies, 60, 16
    num_neurons = f_rates.shape[2]

    f_rates_r = np.repeat(f_rates, int(MS_PER_TRIAL/f_rates.shape[1])+2, axis=1) # 10*num_dot_movies, 4080, 16
    return f_rates_r

    # best way to do this is to shuffle & repeat for the size of the trials that we will need
    # and generate at the start of the experiment

def generate_spikes(f_rates_r):
    #f_rates_r = np.load('/home/macleanlab/stim/rates/ch8_abs_ccd_rates.npy')
    random_matrix = np.random.rand(f_rates_r.shape[0], f_rates_r.shape[1], num_neurons)
    spikes = (f_rates_r - random_matrix > 0)*1.
    spikes_sparse = scipy.sparse.csc_matrix(spikes.reshape((-1, spikes.shape[2])))
    return spikes_sparse

def main(num_dot_movies):
    """
    Difference between two inputs:

    num_dot_movies:
        number of drifting dot movies, each containing 10 moving dots and 10 grey screens, total length = 4800 frames
    num_natural_movies:
        number of natural movies, total length = 240 frames

    The binary spike train matrix is saved as a scipy sparse matrix (.npz)
    """
    print("Reading movies...")
    dot_movies, all_trial_cohs, all_coh_labels, all_is_changes = read_dot(num_dot_movies)

    trial_coh_labels = all_coh_labels.reshape((-1, 20, FRAMES_PER_TRIAL))[:, ::2] # num_dot_movies, 10, 240
    trial_coh_labels = trial_coh_labels.reshape((-1, 10*FRAMES_PER_TRIAL)) # num_dot_movies, 2400
    coh_labels_ms = np.repeat(trial_coh_labels, int(MS_PER_TRIAL/FRAMES_PER_TRIAL)+1, axis=1)
    coh_labels_sparse = scipy.sparse.csc_matrix(coh_labels_ms) # num_dot _movies, 40800

    f_rates_r = generate_mean_firingrates(dot_movies, makepositive)
    #generate_spikes(f_rates_r)

    save_path = '/home/macleanlab/CNN_outputs/rates/'
    np.save(save_path+f'/ch8_abs_ccd_rates.npy', f_rates_r)

    # spikes, [num_dot_movies*10*4080, 16]
    #scipy.sparse.save_npz(save_path+f'/spike_train_{training_mode}_{makepositive}.npz', spikes_sparse)

    # frame-by-frame coherence of each movie, [num_dot_movies, 40800], sparsified
    scipy.sparse.save_npz(save_path+f'/ch8_abs_ccd_coherences.npz', coh_labels_sparse)

    # trial-by-trial coherence changes (0 or 1), [num_dot_movies, 10]
    np.save(save_path+f'/ch8_abs_ccd_changes.npy', all_is_changes)

if __name__ == '__main__':
    main(num_dot_movies)
