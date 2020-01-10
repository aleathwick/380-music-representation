import pretty_midi
import numpy as np
import pandas as pd
from modules.playingWithPrettyMidi import *
from modules.playingWithData import *
from modules.my_classes import *
import pickle
import json
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# layer=keras.layers.Lambda(lambda x:K.one_hot(K.cast(x,'int64'),number_of_classes))(previous_layer) see https://github.com/keras-team/keras/issues/4838
data_path = 'F:\Google Drive\Study\Machine Learning\Datasets\MaestroV2.00\maestro-v2.0.0/'
training_data_path = 'training_data'

# X_events, Y_events, X_val, Y_val, Tx = get_processed_data(data_path, 1, n = 30)
# X_events, Y_events, Tx = get_pickle_data('training_data/training_data300V2.pkl')
# dump_pickle_data((X_events, Y_events, Tx), filename = 'training_data/training_data50.pkl')

# with open('training_data/training_data_completeVAL30.json', 'w') as f:
#     json.dump((X_events, Y_events, X_val, Y_val, Tx), f)

with open('training_data/training_data_300VAL.json', 'r') as f:
    X_events, Y_events, X_val, Y_val, Tx = json.load(f)

m = len(X_events)
print('m', m)
Ty = Tx # length of an example

X = np.array(X_events)
Y = np.array(Y_events)

print("Y dimensions: ", Y.shape)
print('training data collected')
print("X dimensions: ", X.shape)
print("X: ", type(X))
print("X[0]: ", type(X[0]))
print("X[0][0]: ", type(X[0][0]))


#number of hidden LSTM states
n_a = 256

# no of unique events
n_values = 333

# # some of the objects for our network
reshapor = Reshape((1, n_values))
reshapor2 = Reshape((1, n_a))
reshapor3 = Reshape((1, n_a))
LSTM_cell = LSTM(n_a, return_state = True)
LSTM_cell2 = LSTM(n_a, return_state = True, batch_input_shape=(1, n_a))
LSTM_cell3 = LSTM(n_a, return_state = True, batch_input_shape=(1, n_a))
densor = Dense(n_values, activation='softmax')

training_generator = DataGenerator((X, Y), n_a = n_a)
validation_generator = DataGenerator((X_val, Y_val), n_a = n_a)

training_generator = RiggedDataGenerator(n_a = n_a)
validation_generator = RiggedDataGenerator(n_a = n_a)

def musicmodel_LambdaOneHotLayer(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of a sequence in the corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model with the 
    """
    
    # Define the input with a shape 
    X = Input(shape=(Tx,))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    a2 = a0
    c2 = c0
    a3 = a0
    c3 = c0

    # Create empty list to append the outputs to while iterating
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # select the "t"th time step from X. 
        x = Lambda(lambda x: x[:,t])(X)
        # This will be a float indicating class. But we need the class represented in one hot fashion:
        x = Lambda(lambda x: tf.one_hot(K.cast(x, dtype='int32'), 333))(x)
        # We then reshape x to be (1, n_values)
        x = reshapor(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        x = reshapor2(a)
        a2, _, c2 = LSTM_cell2(x, initial_state=[a2, c2])
        # x = reshapor3(a2)
        # a3, _, c3 = LSTM_cell3(x, initial_state=[a3, c3])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a2)
        # Add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X,a0,c0],outputs=outputs)
    
    return model

model = musicmodel_LambdaOneHotLayer(Tx, n_a, n_values)

#Learning rate of 0.01, 20 epochs later loss is 700.51
#Learning rate of 0.1 sucks!
#0.003 is way better. Half an epoch to under 705 from pretrained!
#0.001, and in half an epoch we're down to around 700. Next, increase batch size up to 64, from 16...
#...and in around half an epoch, we're around 700.5. Maybe learning rate could go up slightly.



opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)

# model.load_weights('weights/local0718weights.2x256.20-700.51.hdf5')
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# tb = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=16, write_graph=True,
# write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='batch')
checkpoint = ModelCheckpoint("weights/rigged2.2x256.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
model.fit_generator(generator=training_generator, validation_data=validation_generator,
                    use_multiprocessing=False, epochs=200, max_queue_size=2, callbacks = [checkpoint])



# checkpoint = ModelCheckpoint("weights/0714weights.256.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True)
# callbacks_list = [checkpoint]
# add in callbacks=callbacks_list in model.fit

# history = model.fit([X, a0, c0], list(Y), validation_split=0.25, epochs=600, verbose=2, batch_size=4)

# model.save_weights('weights/model1layer256lstm404epochs.h5')


##### summarize history for loss, from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# model.load_weights('weights/local0718weights.2x256.20-700.51.hdf5')


def music_inference_model_onehot(LSTM_cell, densor, n_values, n_a, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    a2 = a0
    c2 = c0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        x = reshapor2(a)
        a2, _, c2 = LSTM_cell2(x, initial_state=[a2, c2])


        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a2)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        
        x = Lambda(get_max_pred)(x)
        # x = tf.one_hot(np.argmax(out), 333)
        # x = tf.expand_dims(x, axis=-1)
        # x = tf.expand_dims(x, axis=-1)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model

inference_model = music_inference_model_onehot(LSTM_cell, densor, n_values, n_a, Ty)

x_initializer = np.zeros((1, 1, 333))
x_initializer[0][0][308] = 1
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 256), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer,a_initializer,c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    #pred is a list of numpy arrays, with probabilities of events at every time step
    indices = np.argmax(pred, axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices, num_classes=n_values)
    ### END CODE HERE ###
    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

dump_pickle_data((results, indices), '20something_epochs_try_2xlstm.pkl')

# print(callbacks_list)