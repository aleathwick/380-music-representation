import pretty_midi
import numpy as np
import pandas as pd
import pickle
from modules.midiMethods import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
data_path = 'F:\Google Drive\Study\Machine Learning\Datasets\MaestroV2.00\maestro-v2.0.0/'

### notes for adding period numbers:
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

    Roughly 200 baroque, 200 classical, 800 romantic, not many 20th century

    """
    #here is the list of periods, in order of composers
    periods = [4,3,3,2,2,3,4,3,1,3,3,3,3,3,3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,2,1,1,1,3,3,1,3,3,2,3,2,3,3,3,2,3,3,3,1,3,3,3,3,3,3,3,3,3,3,2]

    periods_dict = dict(zip(maestro.index.unique(), periods))
    # Do it this way if composer isn't the index:
    # periods_dict = dict(zip(maestro['canonical_composer'].unique(), periods))

    maestro['period'] = maestro.index.map(periods_dict)
    # Again, do it this way if composer isn't the index:  
    # maestro['period'] = maestro['canonical_composer'].map(periods_dict)


def get_processed_oore_data(data_path, filenames, skip = 1, print_cut_events=True, n = 10,
                            n_events=600, speed=1):
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
        # keep a copy of the notes for later (well, technically a reference...)
        # but we'll reassign to notes with a slice from all_notes, so it will work
        if speed != 1:
            stretch(pm, speed)
        all_notes = pm.instruments[0].notes
        #n_notes is the number of notes to process at once
        # originally 80, now 200
        n_notes = 200
        #n_events is the number of events in a training example
        for i in range(n_notes, len(all_notes) - 1, n_notes):
            pm.instruments[0].notes = all_notes[i - n_notes:i]
            trim_silence(pm)
            events = midi2oore(pm)
            if len(events) >= n_events + 1:
                val_counter += 1
                if val_counter % n == 0:
                    X_val.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
                else:
                    X.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
            else:
                too_short.append(len(events))
    if print_cut_events:
        print('leftover: ', leftover)
        print('too_short: ', too_short)

    return (X, X_val)

def get_processed_oore2_data(data_path, filenames, skip = 1, print_cut_events=True, n = 10,
                            n_events=600, speed=1):
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
        # keep a copy of the notes for later (well, technically a reference...)
        # but we'll reassign to notes with a slice from all_notes, so it will work
        if speed != 1:
            stretch(pm, speed)
        all_notes = pm.instruments[0].notes
        #n_notes is the number of notes to process at once
        # originally 80, now 180
        n_notes = 180
        #n_events is the number of events in a training example
        for i in range(n_notes, len(all_notes) - 1, n_notes):
            pm.instruments[0].notes = all_notes[i - n_notes:i]
            trim_silence(pm)
            events = midi2oore2(pm)
            if len(events) >= n_events + 1:
                val_counter += 1
                if val_counter % n == 0:
                    X_val.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
                else:
                    X.append(events[0:n_events])
                    leftover.append(len(events) - (n_events+1))
            else:
                too_short.append(len(events))
    if print_cut_events:
        print('leftover: ', leftover)
        print('too_short: ', too_short)

    return (X, X_val)


def stretch(pm, speed):
    '''stretches a pm midi file'''
    for note in pm.instruments[0].notes:
        note.start = note.start * speed
        note.end = note.end * speed

def files2note_bin_examples(data_path, filenames, skip = 1, starting_note=0, print_cut_events=True, n_notes=256, speed=1):
    """Reads in midi files, converts to oore, splits into training examples
    
    Arguments:
    data_path -- str, path to the data directory, where all the midi files are
    filenames -- list of filenames to read in
    skip -- int, 'take every nth file'
    starting_note -- int, 'start reading files at nth note'
    print_cut_events -- bool: If true, then lists of numbers of discarded events will be printed, that didn't make the training examples, because they
        would have made the example too long, or they weren't long enough to form an example. 
    n_events -- int, no. of events per training example

    Returns:
    X -- list of training examples, each of length n_notes + 1. X[:-1] for input, X[1:] for output.
    
    """

    n_files = len(filenames)
    file_n = 1
    #just want a selection at this stage
    X = []
    exceeded = []
    max_shift = 9 # 10 total ticks... one of them is ZERO!
    max_duration = 17
    shifts_exceeded = 0
    durations_exceeded = 0
    notes_lost = 0
    # We'll check how many events have to be discarded because they're longer than target sequence length
    leftover = []
    # And we'll check too how many sequences are too short
    too_short = []
    # iterate over the files, taking every skipth file
    for i in range(0, len(filenames), skip):
        pm = pretty_midi.PrettyMIDI(data_path + filenames[i])
        sustain_only(pm)
        desus(pm)
        if speed != 1:
            stretch(pm, speed)
        note_bin = pm2note_bin(pm)

        # iterate over all the notes, in leaps of n_notes
        print('######## Example no.', file_n, 'of', n_files, ', length ' + str(len(note_bin)))
        file_n += skip
        for i in range(starting_note, len(note_bin), n_notes):
            # check there are enough notes left for a training example
            if len(note_bin[i:]) >= n_notes + 1:
                # example, initially, has one extra note, so it can be X and Y
                example = note_bin[i:(i+n_notes+1)]
                # check that there are no notes that are too long, or shifted too much
                if max([note[1] for note in example]) <= max_shift:
                    if max([note[3] for note in example]) <= max_duration:
                        X.append(example)
                    else:
                        # print('exceeded: ' + str(max([note[3] for note in example])))
                        durations_exceeded += 1
                        notes_lost += n_notes + 1
                        exceeded.append(example)

                else:
                    # print('exceeded: ' + str(max([note[1] for note in example])))
                    shifts_exceeded +=1
                    notes_lost += n_notes + 1
                    exceeded.append(example)
            else:
                notes_lost += len(note_bin[i:])
        if print_cut_events:
            print('notes: ', (n_notes + 1) * len(X))
            print('total_durations_exceeded: ', durations_exceeded)
            print('total_shifts_exceeded: ', shifts_exceeded)
            print('notes_lost: ', notes_lost)
    
    return X, exceeded

def nb_data2chroma(examples, mode = 'normal'):
    chroma = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        # print(f'processing example {i} of {len(chroma)}')
        if mode == 'normal':
            chroma[i,:,:] = nb2chroma(e)
        elif mode == 'weighted':
            chroma[i,:,:] = nb2chroma_weighted(e)
        elif mode == 'lowest':
            chroma[i,:,:] = nb2lowest(e)
    return(chroma)

def nb_data2lowest(examples):
    lowest = np.empty((examples.shape[0], examples.shape[1], 12))
    for i, e in enumerate(examples):
        # print(f'processing example {i} of {len(chroma)}')
        lowest[i,:,:] = nb2lowest(e)
    return(chroma)


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

