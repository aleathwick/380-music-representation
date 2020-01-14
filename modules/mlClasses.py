import numpy as np
import tensorflow as tf
def hi():
    print('h')
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras. This is a subclass of Sequence'
    def __init__(self, data, batch_size=64, dim=(128,6), shuffle=True):
        """Initialization
        Note that data should be a list of X
        """
        self.X_data = data
        self.dim = dim #the dimension of a single example
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.X_data) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)
        
        # Different note attribute targets are separate outputs
        return X, [Y[:,:,i] for i in range(6)]

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        # Initialization
        X = np.empty((self.batch_size,) + self.dim)
        Y = np.empty((self.batch_size,) + self.dim) #I think this is right...? Because I'll use sparse categorical cross entropy.

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample, leaving off the last time step
            X[i,:,:] = self.X_data[index][:-1]
            # Store expected output, i.e. leave off the first time step
            Y[i,:,:] = self.X_data[index][1:]

        return X, Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class RiggedDataGenerator(tf.keras.utils.Sequence):
    'Generates RIGGED  data for Keras. This is a subclass of Sequence'
    def __init__(self, batch_size=64, dim=256, n_channels=1,
                 n_classes=333, shuffle=True, n_a = 64):
        """Initialization
        Note that data should be a tuple containing (X, Y)
        """
        X = np.empty((batch_size, dim))
        Y = np.empty((batch_size, dim))
        for j in range(0, dim, 2):
            X[0,j] = 13
            Y[0,j] = 312
            X[0,j+1] = 312
            Y[0,j+1] = 13
        for i in range(batch_size):
            X[i] = X[0]
            Y[i] = Y[0]
        self.X_data, self. Y_data = X, Y
        self.dim = dim #the dimension of a single example. Should it be (256, 333), the shape of a training example?
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.a0 = np.zeros((batch_size, n_a))
        self.c0 = np.zeros((batch_size, n_a))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, Y = self.X_data, list(self.Y_data.transpose(1,0))
        
        return [X, self.a0, self.c0], Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class DataGenerator_onehot(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=8, dim=(256,333), n_channels=1,
                 n_classes=333, shuffle=True):
        """Initialization
        Note that data should be a tuple containing (X, Y)
        """
        self.X_data, self. Y_data = data
        self.dim = dim #the dimension of a single example. Should it be (256, 333), the shape of a training example?
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, Tx)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size), dtype=int) #I think this is right...? Because I'll use sparse categorical cross entropy.

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            X[i,] = keras.utils.to_categorical(self.X_data[index], num_classes=self.n_classes) # don't think I need to specify datatype of float32?

            # Store expected output
            Y[i] = keras.utils.to_categorical(self.Y_data[index], num_classes=self.n_classes)

        return X, Y.transpose(1,0,2)

class MySequence(tf.keras.utils.Sequence): # https://stackoverflow.com/questions/51057123/keras-one-hot-encoding-memory-management-best-possible-way-out
  def __init__(self, data, batch_size = 16):
    self.X = data[0]
    self.Y = data[1]
    self.batch_size = batch_size

  def __len__(self):
     return int(np.ceil(len(self.X) / float(self.batch_size)))

  def __getitem__(self, batch_id):
    # Get corresponding batch data...
    # one-hot encode
    return X, Y


################# then the keras script looks like so: #################

# import numpy as np

# from keras.models import Sequential
# from my_classes import DataGenerator

# # Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)

# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)