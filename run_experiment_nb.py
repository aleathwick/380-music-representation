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
with open('training_data/note_bin_v1/nb_256_train0.95.json', 'r') as f:
    examples95 = json.load(f)
with open('training_data/note_bin_v1/nb_256_train1.05.json', 'r') as f:
    examples105 = json.load(f)
examples = np.concatenate((examples, examples95, examples105))



with open('training_data/note_bin_v1/nb_256_val.json', 'r') as f:
    val = json.load(f)

# chroma = nb_data2chroma(np.array(examples), mode = 'weighted')
# chroma_val = nb_data2chroma(np.array(val), mode = 'weighted')


# build simple model
# excellent example of recurrent model here https://www.tensorflow.org/tutorials/text/text_generation
hidden_state = 512
lstm_layers = 3
seq_length = len(examples[0]) - 1

checkpoint = tf.keras.callbacks.ModelCheckpoint("models/nbmodel20/{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4)
no = 20

reg = tf.keras.regularizers.l1_l2(l1=0.00003, l2=0.00003)
model = models.create_nbmodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
                              seq_length=seq_length, kernel_reg=reg, chroma=False)

training_generator = mlClasses.NbDataGenerator(examples, augment=True, st = 5)
val_gen = mlClasses.NbDataGenerator(val,augment=False)




# opt = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model1/{epoch:02d}-{train_loss:.2f}.hdf5", monitor='train_loss', verbose=2, save_best_only=True, save_weights_only=True)
epochs=20
history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
        callbacks=[checkpoint, stop])
# model.save_weights(f'models/nbcmodel{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
with open(f'models/nbmodel{no}/history{epochs}e.json', 'w') as f:
    json.dump(str(history.history), f)
plt_metric(history)
plt.savefig(f'models/nbmodel{no}/history{no}.json')





# checkpoint = tf.keras.callbacks.ModelCheckpoint("models/nbmodel17/{epoch:02d}-{val_loss:.2f}.hdf5",
#             monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
# stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4)
# no = 17

# reg = tf.keras.regularizers.l1_l2(l1=0.0005, l2=0.0005)
# model = models.create_nbmodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
#                               seq_length=seq_length, kernel_reg=reg, chroma=False)


# # opt = tf.keras.optimizers.Adam(learning_rate=0.002)
# model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
#         callbacks=[checkpoint, stop])
# # model.save_weights(f'models/nbcmodel{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
# with open(f'models/nbmodel{no}/history{epochs}e.json', 'w') as f:
#     json.dump(str(history.history), f)


# hidden_state = 200
# checkpoint = tf.keras.callbacks.ModelCheckpoint("models/nbmodel18/{epoch:02d}-{val_loss:.2f}.hdf5",
#             monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
# stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4)
# no = 18

# model = models.create_nbmodel(hidden_state_size=hidden_state, lstm_layers=lstm_layers,
#                               seq_length=seq_length, kernel_reg=None, chroma=False)


# # opt = tf.keras.optimizers.Adam(learning_rate=0.002)
# model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit_generator(training_generator, validation_data=val_gen, epochs=epochs,
#         callbacks=[checkpoint, stop])
# # model.save_weights(f'models/nbcmodel{no}/model{no}{epochs}e{hidden_state}ss{lstm_layers}l.h5')
# with open(f'models/nbmodel{no}/history{epochs}e.json', 'w') as f:
#     json.dump(str(history.history), f)