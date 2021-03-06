from importlib import reload
import json
import pretty_midi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from modules.midiMethods import *

from modules.dataMethods import *

import modules.models as models

import modules.mlClasses as mlClasses


experiments = (23,24,25)
# chroma modes are what I'm interested in varying at the moment. Other hyperparameters I won't have iterable right now.
chroma_modes = ('normal',)
augment_time = False
epochs=75
hidden_state = 512
lstm_layers = 3
lr = 0.001
# choose what the callbacks monitor
monitor = 'loss'
transpose = False
st = 0

for i in range(len(experiments)):
    chroma_mode = chroma_modes[i]
    no = experiments[i]
    
    data_dir = 'training_data/oore_v2/'
    # save text file with the basic parameters used
    with open(f'models/oorec/oorec{no}/description.txt', 'w') as f:
        f.write(f'no: {no}\n')
        f.write(f'lstm_layers: {lstm_layers}\n')
        f.write(f'chroma_mode: {chroma_mode}\n')
        f.write(f'augment_time: {augment_time}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'hidden_state: {hidden_state}\n')
        f.write(f'learning rate: {lr}\n')
        f.write(f'transpose: {transpose}\n')
        f.write(f'st: {st}\n')


    
# load training data
with open(data_dir + 'oore2_train_1.json', 'r') as f:
    X_train = json.load(f)
# with open(data_dir + 'oore2_train_0.9.json', 'r') as f:
#     X2 = json.load(f)
# with open(data_dir + 'oore2_train_1.1.json', 'r') as f:
#     X3 = json.load(f)
# X_train = np.concatenate((X_train, X2, X3))
with open(data_dir + 'oore2_val.json', 'r') as f:
    X_val = json.load(f)


    with open(data_dir + 'oore2_train_1.json', 'r') as f:
        examples = json.load(f)
    
    with open(data_dir + 'oore2_val.json', 'r') as f:
        val = json.load(f)

    if chroma_mode != 'none':
        with open(data_dir + f'oore2_train_1_chroma{chroma_mode}.json', 'r') as f:
            chroma = json.load(f)               

        with open(data_dir + f'oore2_val_chroma{chroma_mode}.json', 'r') as f:
            chroma_val = json.load(f)
    

    # if required, add in data for other speeds (assuming 0.9 and 1.1 speeds are going to be added)
    if augment_time:
        with open(data_dir + 'oore2_train_0.9.json', 'r') as f:
            examples9 = json.load(f)
        with open(data_dir + 'oore2_train_1.1.json', 'r') as f:
            examples11 = json.load(f)
        examples = np.concatenate((examples, examples9, examples11))

        if chroma_mode != 'none':
            with open(data_dir + f'oore2_train_0.9_chroma{chroma_mode}.json', 'r') as f:
                chroma9 = json.load(f)
            with open(data_dir + f'oore2_train_1.1_chroma{chroma_mode}.json', 'r') as f:
                chroma11 = json.load(f)
            chroma = np.concatenate((chroma, chroma9, chroma11))
    
    # if this run is without chroma, we still need to add in zeros to replace chroma, so that the same input shape is retained
    # could be done more efficiently, in the data generator... but oh well.
    if chroma_mode == 'none':
        chroma = np.zeros((len(examples), len(examples[0]), 12))
        chroma_val = np.zeros((len(val), len(examples[0]), 12))

    seq_length = len(examples[0]) - 1

    # set up callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/oorec/oorec" + str(no) + "/{epoch:02d}-{" + monitor + ":.2f}.hdf5",
                                monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True)
    # early stopping, if needed
    # stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=5)
    callbacks = [checkpoint]

    # create model
    model = models.create_ooremodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                                seq_length=seq_length, chroma=True)

    # Get data generators for train and validation
    training_generator = mlClasses.OoreDataGenerator(examples, chroma = np.array(chroma), augment=transpose, st = 5)
    val_gen = mlClasses.OoreDataGenerator(val, chroma = np.array(chroma_val), augment=False)

    # optimizer, and compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
            callbacks=callbacks, verbose=2)
    
    # save the model weights and history
    model.save_weights(f'models/oorec/oorec{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
    with open(f'models/oorec/oorec{no}/history{epochs}e.json', 'w') as f:
        json.dump(str(history.history), f)
    
    # save a graph of the training vs validation progress
    models.plt_metric(history.history)
    plt.savefig(f'models/oorec/oorec{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l')
    # clear the output
    plt.clf()