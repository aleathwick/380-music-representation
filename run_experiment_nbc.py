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
    examples = json.load(f)
# with open('training_data/note_bin_v1/nb_256_train0.95.json', 'r') as f:
#     examples95 = json.load(f)
# with open('training_data/note_bin_v1/nb_256_train1.05.json', 'r') as f:
#     examples105 = json.load(f)
# examples = np.concatenate((examples, examples95, examples105))

with open('training_data/note_bin_v1/nb_256_val.json', 'r') as f:
    val = json.load(f)

chroma = nb_data2chroma(np.array(examples), weighted=True)
chroma_val = nb_data2chroma(np.array(val), weighted=True)


# build simple model
# excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
hidden_state = 512
lstm_layers = 3
seq_length = len(examples[0]) - 1

checkpoint = tf.keras.callbacks.ModelCheckpoint("models/nbcmodel3/{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2)
no = 3

model2 = models.create_model1(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                              seq_length=seq_length, recurrent_dropout=0.0, chroma=True)
training_generator = mlClasses.DataGenerator(examples, chroma = chroma, augment=False, st = 0)
val_gen = mlClasses.DataGenerator(val, chroma = chroma_val, augment=False)

opt = tf.keras.optimizers.Adam(learning_rate=0.003)
model2.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model1/{epoch:02d}-{train_loss:.2f}.hdf5", monitor='train_loss', verbose=2, save_best_only=True, save_weights_only=True)
epochs=40
history = model2.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
        callbacks=[checkpoint, stop])
model2.save_weights(f'models/nbcmodel{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
with open(f'models/nbcmodel{no}/history{epochs}e.json', 'w') as f:
    json.dump(str(history.history), f)





# # build simple model
# # excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
# hidden_state = 400
# lstm_layers = 3
# seq_length = len(examples[0]) - 1

# checkpoint = tf.keras.callbacks.ModelCheckpoint("models/nbmodel8/{epoch:02d}-{val_loss:.2f}.hdf5",
#             monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
# stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)

# model2 = models.create_model1(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
#                               seq_length=seq_length, recurrent_dropout=0.0, chroma=False)
# training_generator = mlClasses.DataGenerator(examples, augment=True, st = 5)
# val_gen = mlClasses.DataGenerator(val, augment=False)


# model2.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model1/{epoch:02d}-{train_loss:.2f}.hdf5", monitor='train_loss', verbose=2, save_best_only=True, save_weights_only=True)
# epochs=40
# history = model2.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
#         callbacks=[checkpoint, stop])
# model2.save_weights(f'models/nbmodel8/model8{epochs}e{hidden_state}ss{lstm_layers}l.h5')
# with open(f'models/nbmodel8/history{epochs}e.json', 'w') as f:
#     json.dump(str(history.history), f)