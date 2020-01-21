import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

def create_nbmodel(hidden_state_size = 512, seq_length = 256, batch_size=128, stateful = False,
    lstm_layers = 3, recurrent_dropout = 0.0, chroma=False,
    vocab={"pitch":88, "shift_M":10, "shift_m":60, "duration_M":18, "duration_m":30, "velocity":32}):
    """creates a simple model
    
    Arguments:
    n_inputs -- int, describing how many inputs there are per note, after unrolling
    
    """
    n_inputs = 6
    if chroma:
        n_inputs += 12
    # get inputs
    # pitch = tf.keras.Input(shape=(128,))
    # shift_M = tf.keras.Input(shape=(128,))
    # shift_m = tf.keras.Input(shape=(128,))
    # duration_M = tf.keras.Input(shape=(128,))
    # duration_m = tf.keras.Input(shape=(128,))
    # velocity = tf.keras.Input(shape=(128,))

    # inputs = [pitch, shift_M, shift_m, duration_M, duration_m, velocity]
    if stateful:
        inputs = tf.keras.Input(batch_shape=(batch_size,seq_length,n_inputs))
    else:
        inputs = tf.keras.Input(shape=(seq_length,n_inputs))
    # Could feed these into embedding layers? Would it need time distributed, or can it do 3d?
    # pitch_emb = layers.embedding(vocab['pitch'], embed_dim, batch_input_shape=[batch_size, None])(pitch_input)
    # run them through one hot layers
    x1 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,0], dtype='int32'), vocab['pitch']))(inputs)
    x2 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,1], dtype='int32'), vocab['shift_M']))(inputs)
    x3 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,2], dtype='int32'), vocab['shift_m']))(inputs)
    x4 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,3], dtype='int32'), vocab['duration_M']))(inputs)
    x5 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,4], dtype='int32'), vocab['duration_m']))(inputs)
    x6 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,5], dtype='int32'), vocab['velocity']))(inputs)
    if chroma:
        x7 = layers.Lambda(lambda x: x[:,:,6:])(inputs)
        x6 = layers.concatenate([x6, x7])


    # concatenate output
    x = layers.concatenate([x1,x2,x3,x4,x5,x6])
    # don't think I need return state here, as I'm not doing the for loop manually?
    for i in range(lstm_layers):
        x = layers.LSTM(hidden_state_size, return_sequences=True, stateful=stateful,
                        recurrent_dropout=recurrent_dropout)(x)

    pitch_out = layers.Dense(vocab['pitch'], activation='softmax', )(x)
    shift_M_out = layers.Dense(vocab['shift_M'], activation='softmax')(x)
    shift_m_out = layers.Dense(vocab['shift_m'], activation='softmax')(x)
    duration_M_out = layers.Dense(vocab['duration_M'], activation='softmax')(x)
    duration_m_out = layers.Dense(vocab['duration_m'], activation='softmax')(x)
    velocity_out = layers.Dense(vocab['velocity'], activation='softmax')(x)

    outputs = [pitch_out, shift_M_out, shift_m_out, duration_M_out, duration_m_out, velocity_out]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'{lstm_layers}layerLSTM')

    model.summary()

    return model


def generate_nbmusic(model, num_generate=256, temperatures=[0.2] * 6, input_notes=[[34,0,0,3,3,16]], chroma=False):
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # Number of notes to generate
    notes_generated = []
    input_notes = np.array(input_notes)

    # Here batch size == 1
    model.reset_states()

    # prime the model with the input notes
    for i, input_note in enumerate(input_notes[:-1]):
        notes_generated.append(input_note)
        # I think I need to do this? batch size of 1...
        input_note = tf.expand_dims(input_note, 0)
        input_note = tf.expand_dims(input_note, 0)
        predictions = model(input_note)


    input_note = input_notes[-1]
    input_note = np.array(input_note)
    input_note = tf.expand_dims(input_note, 0)
    input_note = tf.expand_dims(input_note, 0)
    for i in range(num_generate):
        predictions = model(input_note)

        note = []

        # using a categorical distribution to predict the note returned by the model
        # have to do this for each output attribute of the note
        for attribute, temperature in zip(predictions, temperatures):
            # remove the batch dimension
            attribute = tf.squeeze(attribute, 0)
            attribute = attribute / temperature
            predicted_id = tf.random.categorical(attribute, num_samples=1)[-1,0].numpy()
            note.append(predicted_id)

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_note = tf.expand_dims(tf.expand_dims(note, 0), 0)

        notes_generated.append(note)

    return(notes_generated)


def create_ooremodel(hidden_state_size = 512, seq_length = 256, batch_size=128, stateful = False,
    lstm_layers = 3, recurrent_dropout = 0.0, chroma=False):
    """creates a simple model
    
    Arguments:
    n_inputs -- int, describing how many inputs there are per note, after unrolling
    
    """
    n_inputs = 1
    if chroma:
        n_inputs += 12

    if stateful:
        inputs = tf.keras.Input(batch_shape=(batch_size,seq_length,n_inputs))
    else:
        inputs = tf.keras.Input(shape=(seq_length,n_inputs))
    # run through a one hot layer
    x = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,0], dtype='int32'), 333))(inputs)
    if chroma:
        x1 = layers.Lambda(lambda x: x[:,:,1:])(inputs)
        x = layers.concatenate([x, x1])

    # don't think I need return state here, as I'm not doing the for loop manually?
    for i in range(lstm_layers):
        x = layers.LSTM(hidden_state_size, return_sequences=True, stateful=stateful,
                        recurrent_dropout=recurrent_dropout)(x)

    event_out = layers.Dense(333, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=event_out, name=f'{lstm_layers}layerLSTM')

    model.summary()

    return model



def generate_ooremusic(model, num_generate=256, temperature=0.2, input_events=[34,0,0,3,3,16], chroma=False):
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # Number of notes to generate
    events_generated = []
    input_events = np.array(input_events)

    # Here batch size == 1
    model.reset_states()

    # prime the model with the input notes
    for i, input_event in enumerate(input_events[:-1]):
        events_generated.append(input_event)
        # I think I need to do this? batch size of 1...
        input_event = tf.expand_dims(input_event, 0)
        input_event = tf.expand_dims(input_event, 0)
        input_event = tf.expand_dims(input_event, 0)
        predictions = model(input_event)


    input_event = input_events[-1]
    input_event = np.array(input_event)
    input_event = tf.expand_dims(input_event, 0)
    input_event = tf.expand_dims(input_event, 0)
    input_event = tf.expand_dims(input_event, 0)
    for i in range(num_generate):
        prediction = model(input_event)

        # using a categorical distribution to predict the note returned by the model
        # have to do this for each output attribute of the note
            # remove the batch dimension
        prediction = tf.squeeze(prediction, 0)
        prediction = prediction / temperature
        predicted_id = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_event = tf.expand_dims(tf.expand_dims(tf.expand_dims(predicted_id, 0), 0), 0)

        events_generated.append(predicted_id)

    return(events_generated)



def plt_metric(history, metric='loss'):
    """plots metrics from the history of a model
    
    Arguments:
    history -- history of a keras model
    metric -- str, metric to be plotted
    
    """

    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

