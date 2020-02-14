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


experiments = (9,10,11,12,13)
augment_time = False
epochs=50
hidden_state = 512
lstm_layers = 3
learning_rates = (0.0001,0.0003,0.00075,0.001,0.003)
# choose what the callbacks monitor
monitor = 'loss'
transpose = True
st = 5

for i in range(len(experiments)):
    no = experiments[i]
    lr = learning_rates[i]
    model_path = f'models/oore/oore{no}/'

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


    with open('training_data/oore_v2/oore2_train_1.json', 'r') as f:
        examples = json.load(f)

    with open('training_data/oore_v2/oore2_val.json', 'r') as f:
        val = json.load(f)
    

    # if required, add in data for other speeds (assuming 0.9 and 1.1 speeds are going to be added)
    epoch_per_dataset=1
    if augment_time:
        with open('training_data/oore_v2/oore2_train_0.9.json', 'r') as f:
            examples9 = json.load(f)
        with open('training_data/oore_v2/oore2_train_1.1.json', 'r') as f:
            examples11 = json.load(f)
        epoch_per_dataset=3
        examples = np.concatenate((examples, examples9, examples11))



    seq_length = len(examples[0]) - 1

    # set up callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path + "{epoch:02d}-{" + monitor + ":.2f}.hdf5",
                                monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True)
    # early stopping, if needed
    stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0.001, patience=2)
    callbacks = [checkpoint, stop]

    # create model
    model = models.create_ooremodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                                seq_length=seq_length, chroma=False)



    # Get data generators for train and validation
    training_generator = mlClasses.OoreDataGenerator(examples, augment=transpose, st = 5, epoch_per_dataset=epoch_per_dataset)
    val_gen = mlClasses.OoreDataGenerator(val, augment=False)

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


