import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def create_model1(hidden_state_size = 128, seq_length = 128, batch_size=128, stateful = False,
    vocab={"pitch":88, "shift_M":10, "shift_m":60, "duration_M":18, "duration_m":30, "velocity":32}):
    """creates a simple model
    
    Arguments:
    n_inputs -- int, describing how many inputs there are per note, after unrolling
    
    """

    # get inputs
    # pitch = tf.keras.Input(shape=(128,))
    # shift_M = tf.keras.Input(shape=(128,))
    # shift_m = tf.keras.Input(shape=(128,))
    # duration_M = tf.keras.Input(shape=(128,))
    # duration_m = tf.keras.Input(shape=(128,))
    # velocity = tf.keras.Input(shape=(128,))

    # inputs = [pitch, shift_M, shift_m, duration_M, duration_m, velocity]
    if stateful:
        inputs = tf.keras.Input(batch_shape=(batch_size,seq_length,6))
    else:
        inputs = tf.keras.Input(shape=(seq_length,6))
    # Could feed these into embedding layers? Would it need time distributed, or can it do 3d?
    # pitch_emb = layers.embedding(vocab['pitch'], embed_dim, batch_input_shape=[batch_size, None])(pitch_input)
    # run them through one hot layers
    x1 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,0], dtype='int32'), vocab['pitch']))(inputs)
    x2 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,1], dtype='int32'), vocab['shift_M']))(inputs)
    x3 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,2], dtype='int32'), vocab['shift_m']))(inputs)
    x4 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,3], dtype='int32'), vocab['duration_M']))(inputs)
    x5 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,4], dtype='int32'), vocab['duration_m']))(inputs)
    x6 = layers.Lambda(lambda x: tf.one_hot(tf.cast(x[:,:,5], dtype='int32'), vocab['velocity']))(inputs)

    # concatenate output
    concatenated = layers.concatenate([x1,x2,x3,x4,x5,x6])
    # don't think I need return state here, as I'm not doing the for loop manually?
    x = layers.LSTM(hidden_state_size, return_sequences=True, stateful=stateful)(concatenated)

    pitch_out = layers.Dense(vocab['pitch'], activation='softmax', )(x)
    shift_M_out = layers.Dense(vocab['shift_M'], activation='softmax')(x)
    shift_m_out = layers.Dense(vocab['shift_m'], activation='softmax')(x)
    duration_M_out = layers.Dense(vocab['duration_M'], activation='softmax')(x)
    duration_m_out = layers.Dense(vocab['duration_m'], activation='softmax')(x)
    velocity_out = layers.Dense(vocab['velocity'], activation='softmax')(x)

    outputs = [pitch_out, shift_M_out, shift_m_out, duration_M_out, duration_m_out, velocity_out]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='1layerLSTM')

    model.summary()

    return model


def generate_music(model):
    # Number of notes to generate
    num_generate = 128

    # we need a start note
    input_note = np.array([34,0,3,3,3,16])
    # I think I need to do this? batch size of 1...
    input_note = tf.expand_dims(input_note, 0)
    input_note = tf.expand_dims(input_note, 0)
    print(input_note)

    # Empty list to store results
    notes_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.2

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_note)

        note = []

        # using a categorical distribution to predict the note returned by the model
        # have to do this for each output attribute of the note
        for attribute in predictions:
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



