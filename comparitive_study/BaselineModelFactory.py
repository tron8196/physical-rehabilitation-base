import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Activation, Convolution1D, LeakyReLU, Flatten, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
dropout_rate = 0.2
def deep_rnn_network(timesteps, n_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape = (timesteps, n_dim)))
    model.add(Dropout(0.25))
    model.add(Dense(30, activation = 'tanh'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(10)))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def deep_cnn_network(timesteps, n_dim):
    model = Sequential()
    model.add(Convolution1D(60, 5, padding ='same', strides = 2, input_shape = (timesteps,n_dim)))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    model.add(Convolution1D(30, 3, padding ='same', strides = 2))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    model.add(Convolution1D(10, 3, padding ='same'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    model.add(Dense(100))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
