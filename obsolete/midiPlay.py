import pretty_midi
import numpy as np
import pandas as pd
from modules.playingWithPrettyMidi import *
from modules.playingWithData import *


# cc64 is sustain
# cc66 is sostenuto
# cc67 is soft


data_path = 'F:\Google Drive\Study\Machine Learning\Datasets\MaestroV2.00\maestro-v2.0.0/'
midi_path = 'midi/'

################ Some different files to work with ################
pedal_processed = midi_path + 'tf.midi'
test1 = midi_path + 'MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi'
test2 = midi_path + 'MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi'
test3 = midi_path + 'ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--1.midi'
short = midi_path + 'short.midi'
shortonlysus = midi_path + 'shortonlysus.midi'
one_hot_test = midi_path + 'one_hot_test.mid'

################ Load a file ################
# pm = pretty_midi.PrettyMIDI(shortonlysus)

# desus(pm)
# pm.write('npmbefore.midi')
# events = midi_to_events(pm)
# # print(events)
# npm = events_to_midi(events)
# print(npm.instruments[0].notes)

results, indices = get_pickle_data('20something_epochs_try_2xlstm.pkl')
print(indices)
# print(type(ohv))
# print(ohv)

# X_events, Y_events, Tx = get_pickle_data('training_data/training_data300V2.pkl')
# X_events, Y_events, Tx = get_processed_data(data_path, 500)
# print(len(X_events))
# pm = events_to_midi(X_events[-1])
# pm.write('doesit.midi')

# print(X_events[40][0:5])
# print(Y_events[40][0:5])

def show_duplicates(X_events):
    """list duplicate sequential events. Input is a list of examples."""
    for x in X_events:
        duplicates = []
        for s in x:
            if duplicates == []:
                duplicates.append(s)
            elif duplicates[-1] == s:
                duplicates.append(s)
            elif len(duplicates) > 1:
                print(duplicates)
                duplicates = []
            else:
                duplicates = []

# pmY = events_to_midi(Y_events[10])
# pmX = events_to_midi(X_events[10])

# print('########## BEFORE ##########')
# print('\n'.join([str(note) for note in pm.instruments[0].notes[:30]]))
# print(midi_to_one_hot(pm))

# print('\n'.join([str(cc) for cc in pm.instruments[0].control_changes[:30]]))
# print('########## AFTER ###########')
# sustain_only(pm)
# desus(pm)
# print('\n'.join([str(note.start) for note in pm.instruments[0].notes[:30]]))
# print('\n'.join([str(cc) for cc in pm.instruments[0].control_changes[:30]]))

# print(pmX.instruments[0].notes[:5])
# print('################################')
# zero_start_time(pmX)
# print(pmX.instruments[0].notes[:5])
# pmX.write('X1.midi')
