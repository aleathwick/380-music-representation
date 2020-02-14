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
augment_time_list = (True, False, True)
epochs=99
hidden_state = 512
lstm_layers = 3
lr = 0.001
# choose what the callbacks monitor
monitor = 'loss'
transpose_list = (True, True, False)
st_list = (5, 5, 0)

for i in range(len(experiments)):
    no = experiments[i]
    model_path = f'models/nb/nb{no}/'

    augment_time = augment_time_list[i]
    transpose = transpose_list[i]
    st = st_list[i]

        # save text file with the basic parameters used
    with open(model_path + 'description.txt', 'w') as f:
        f.write(f'no: {no}\n')
        f.write(f'lstm_layers: {lstm_layers}\n')
        f.write(f'augment_time: {augment_time}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'hidden_state: {hidden_state}\n')
        f.write(f'learning rate: {lr}\n')
        f.write(f'transpose: {transpose}\n')
        f.write(f'st: {st}\n')

    with open('training_data/note_bin_v2/nb_220_train1.json', 'r') as f:
        examples = json.load(f)
    
    with open('training_data/note_bin_v2/nb_220_val.json', 'r') as f:
        val = json.load(f)
    

    # if required, add in data for other speeds (assuming 0.9 and 1.1 speeds are going to be added)
    epoch_per_dataset=1
    if augment_time:
        with open('training_data/note_bin_v2/nb_220_train0.9.json', 'r') as f:
            examples9 = json.load(f)
        with open('training_data/note_bin_v2/nb_220_train1.1.json', 'r') as f:
            examples11 = json.load(f)
        epoch_per_dataset=3
        examples = np.concatenate((examples, examples9, examples11))

    seq_length = len(examples[0]) - 1

    # set up callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path + "{epoch:02d}-{" + monitor + ":.2f}.hdf5",
                                monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True)
    # early stopping, if needed
    # stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=5)
    callbacks = [checkpoint]

    # create model
    model = models.create_nbmodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                                seq_length=seq_length, chroma=False)

    # Get data generators for train and validation
    training_generator = mlClasses.NbDataGenerator(examples, augment=transpose, st = 5, epoch_per_dataset=epoch_per_dataset)
    val_gen = mlClasses.NbDataGenerator(val, augment=False)

    # optimizer, and compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
            callbacks=callbacks, verbose=2)
    
    # save the model weights and history
    model.save_weights(model_path + f'model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
    with open(model_path + f'history{epochs}e.json', 'w') as f:
        json.dump(str(history.history), f)
    
    # save a graph of the training vs validation progress
    models.plt_metric(history.history)
    plt.savefig(model_path + f'model{no}-{epochs}e{hidden_state}ss{lstm_layers}l')
    # clear the output
    plt.clf()
