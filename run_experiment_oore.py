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



# load data
with open('training_data/oore_v2/oore2_train_1.json', 'r') as f:
    X1 = json.load(f)
with open('training_data/oore_v2/oore2_train_0.9.json', 'r') as f:
    X2 = json.load(f)
with open('training_data/oore_v2/oore2_train_1.1.json', 'r') as f:
    X3 = json.load(f)
X_train = np.concatenate((X1, X2, X3))

with open('training_data/oore_v2/oore2_val.json', 'r') as f:
    X_val = json.load(f)



# build simple model
# excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
hidden_state = 512
lstm_layers = 3
seq_length = len(X_train[0]) - 1
model = models.create_ooremodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                              seq_length=seq_length)
training_generator = mlClasses.OoreDataGenerator(X_train, augment=True, st=5)
val_gen = mlClasses.OoreDataGenerator(X_val, augment=False)


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint("models/oore9/{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=True)
stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=80)
epochs=32
history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, stop])
# model.save_weights(f'models/oore2/model{epochs}e{hidden_state}ss{lstm_layers}l.h5')
with open(f'models/oore9/history{epochs}e.json', 'w') as f:
    json.dump(str(history.history), f)


