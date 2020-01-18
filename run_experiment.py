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




with open('training_data/note_bin_v1/nb_256_train1.00.json', 'r') as f:
    examples100 = json.load(f)
with open('training_data/note_bin_v1/nb_256_train0.95.json', 'r') as f:
    examples95 = json.load(f)
with open('training_data/note_bin_v1/nb_256_train1.05.json', 'r') as f:
    examples105 = json.load(f)

examples = np.concatenate((examples100, examples95, examples105))

with open('training_data/note_bin_v1/nb_256_val.json', 'r') as f:
    val = json.load(f)


# build simple model
# excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
hidden_state = 200
lstm_layers = 4
seq_length = len(examples[0]) - 1
model2 = models.create_model1(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                              seq_length=seq_length, recurrent_dropout=0.0)
training_generator = mlClasses.DataGenerator(examples, augment=True, st = 5)
val_gen = mlClasses.DataGenerator(val, augment=False)


model2.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model1/{epoch:02d}-{train_loss:.2f}.hdf5", monitor='train_loss', verbose=2, save_best_only=True, save_weights_only=True)
epochs=30
history = model2.fit_generator(training_generator, validation_data=val_gen, epochs=epochs)
model2.save_weights(f'models/nbmodel8/model8{epochs}e{hidden_state}ss{lstm_layers}l.h5')
with open(f'models/nbmodel8/history{epochs}e.json', 'w') as f:
    json.dump(str(history.history), f)





# # build simple model
# # excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
# hidden_state = 512
# lstm_layers = 3
# seq_length = len(examples[0]) - 1
# model2 = models.create_model1(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
#                               seq_length=seq_length)
# training_generator = mlClasses.DataGenerator(examples, augment=True, st = 6)
# val_gen = mlClasses.DataGenerator(val, augment=False)


# model2.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model1/{epoch:02d}-{train_loss:.2f}.hdf5", monitor='train_loss', verbose=2, save_best_only=True, save_weights_only=True)
# epochs=20
# history = model2.fit_generator(training_generator, validation_data=val_gen, epochs=epochs)
# model2.save_weights(f'models/nbmodel5/model6{epochs}e{hidden_state}ss{lstm_layers}l.h5')
# with open(f'models/nbmodel6/history{epochs}e.json', 'w') as f:
#     json.dump(str(history.history), f)