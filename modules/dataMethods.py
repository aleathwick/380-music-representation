import pretty_midi
import numpy as np
import pandas as pd
import pickle
from modules.midiMethods import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
data_path = 'F:\Google Drive\Study\Machine Learning\Datasets\MaestroV2.00\maestro-v2.0.0/'



# print(maestro.head)
# print(maestro['canonical_composer'].unique())

#[1885, 1872, 1861, 1729, 1786, ]


# 1 = roughly baroque
# 2 = roughly classical
# 3 = roughly romantic
# 4 = roughly 20th century

def add_period_numbers(df):
    """ This is how I added in period numbers to the pandas dataframe.

    Periods:
    # 1 = roughly baroque
    # 2 = roughly classical
    # 3 = roughly romantic
    # 4 = roughly 20th century

    """
    #here is the list of periods, in order of composers
    periods = [4,3,3,2,2,3,4,3,1,3,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,2,1,1,1,3,3,1,3,3,2,3,2,3,3,3,2,3,3,3,1,3,3,3,3,3,3,3,3,3,3,2]

    periods_dict = dict(zip(maestro.index.unique(), periods))
    # Do it this way if composer isn't the index:
    # periods_dict = dict(zip(maestro['canonical_composer'].unique(), periods))

    maestro['period'] = maestro.index.map(periods_dict)
    # Again, do it this way if composer isn't the index:  
    # maestro['period'] = maestro['canonical_composer'].map(periods_dict)


def get_processed_data(data_path, skip = 50, print_cut_events=False, n = 8, n_events=256):
    """Reads in midi files, converts to oore, splits into training examples
    
    Parameters:
    ----------
    data_path: str
        path to the data directory, where all the midi files are

    skip : int
        take every nth file, controls number of files in between those taken

    print_cut_events : bool
        If true, then lists of numbers of discarded events will be printed, that didn't make the training examples, because they
        would have made the example too long, or they weren't long enough to form an example. 

    n : int
        sends every nth training example to the validation set

    n_events : int
        no. of events per training example

    Returns
    ----------
    X : list
        list of training examples

    Y : list
        Essentially the same as X, but displaced by a timestep, so it can act as the truth set at each step
    
    X_val : list
        list of validation examples

    Y_val : list
        Essentially the same as X_val, but displaced by a timestep, so it can act as the truth set at each step


    n_events: int
        number of events in a training example
    
    """
    maestro = pd.read_csv('training_data/maestro-v2.0.0withPeriod.csv', index_col=0)
    filenames = list(maestro['midi_filename'])
    #just want a selection at this stage
    X = []
    Y = []
    X_val = []
    Y_val = []
    val_counter = 1
    # We'll check how many events have to be discarded because they're longer than target sequence length
    leftover = []
    # And we'll check too how many sequences are too short
    too_short = []
    for i in range(0, len(filenames) - 1, skip):
        pm = pretty_midi.PrettyMIDI(data_path + filenames[i])
        sustain_only(pm)
        desus(pm)
        all_notes = pm.instruments[0].notes
        #n_notes is the number of notes to process at once
        n_notes = 80
        #n_events is the number of events in a training example
        for i in range(n_notes, len(all_notes) - 1, n_notes):
            pm.instruments[0].notes = all_notes[i - n_notes:i]
            trim_silence(pm)
            events = midi2oore(pm)
            if len(events) >= n_events + 1:
                val_counter += 1
                if val_counter % n == 0:
                    X_val.append(events[0:n_events])
                    Y_val.append(events[1:n_events+1])
                    leftover.append(len(events) - (n_events+1))
                else:
                    X.append(events[0:n_events])
                    Y.append(events[1:n_events+1])
                    leftover.append(len(events) - (n_events+1))
            else:
                too_short.append(len(events))
    if print_cut_events:
        print('leftover: ', leftover)
        print('too_short: ', too_short)

    return (X, Y, X_val, Y_val, n_events)


def dump_pickle_data(item, filename):
    with open(filename, 'wb') as f:
        pickle.dump(item, f, protocol=2)

def get_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def make_one_hot(examples, n_values=333):
    """ takes in a list of training examples, returns them in one hot format
    
    Inputs
    ----------
    examples : list
        should contain training examples, which themselves are lists of events expressed in integers
    
    """
    arr = np.empty((len(examples), len(examples[0]), n_values), dtype=object)
    for i in range(len(examples)):
        one_hots = to_categorical(examples[i], num_classes=n_values, dtype='float32')
        arr[i] = one_hots
    return arr

def get_max_pred(l):
    array = np.zeros((1, 1, 333))
    array[0][0] = to_categorical(np.argmax(l), num_classes=333)
    return tf.convert_to_tensor(array, dtype=tf.float32)




# print(get_max_pred([1,2,3,2,1]))

# a = np.array([np.array([1]),np.array([2]), np.array([1])])

# print(get_max_pred(a))

# print(tf.convert_to_tensor(a))

# training_events = get_processed_data(data_path)

# print(len(training_events[0]))
# print(len(training_events))
            








# print(maestro.loc[:,'canonical_composer'])

# add_period_numbers(maestro)
# print(maestro[maestro['canonical_composer']=='Alban Berg'])
# print(maestro[maestro.index=='Alban Berg'])
# print(maestro.head())


# for filename in maestro['midi_filename']:
#     print(filename)
# print(maestro.index)

# pm = pretty_midi.PrettyMIDI(data_path + '2011/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_16_Track16_wav.midi')
# print(pm)
# pm.write('hahahahahahahahahahaha.midi')