# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import time
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, LSTM
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.8):
    """Returns points smoothed over an exponential."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_history(history, smooth_fac=0.0):
    """Plots the given history object."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, smooth_curve(acc, factor=smooth_fac), 'bo',
             label='Smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc, factor=smooth_fac), 'b',
             label='Smoothed validation acc')
    plt.title('Training and validation accuracy, smoothing = {0}'.format(
        smooth_fac))
    plt.legend()
    plt.figure()
    plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b',
             label='Smoothed validation loss')
    plt.title('Training and validation loss, smoothing = {0}'.format(smooth_fac))
    plt.legend()
    plt.show()


def numpy_rnn():
    """Runs the forward pass of a numpy-based RNN. This shows you what
    is under the hood of an RNN."""
    timesteps = 100
    input_features = 32
    output_features = 64
    inputs = np.random.random((timesteps, input_features))
    state_t = np.zeros((output_features,))
    W = np.random.random((output_features, input_features))
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features,))
    successive_outputs = []
    for input_t in inputs:
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
        successive_outputs.append(output_t)
        state_t = output_t
    final_output_sequence = np.concatenate(successive_outputs, axis=0)
    print(final_output_sequence.shape)
    print(final_output_sequence)


def imdb_rnn():
    """Runs an RNN on the IMDB dataset."""
    max_features = 10000
    maxlen = 500
    batch_size = 32
    print('Loading data.')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(
        num_words=max_features)
    print('{0} training sequences.'.format(len(input_train)))
    print('{0} test sequences.'.format(len(input_test)))
    print('Pad sequences (samples x time).')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape: {0}'.format(input_train.shape))
    print('input_test shape: {0}'.format(input_test.shape))
    model = Sequential()
    model.add(Embedding(max_features, 32, input_length=maxlen))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    start = time.time()
    history = model.fit(input_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         validation_split=0.2)
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    plot_history(history)


def imdb_lstm():
    """Runs an LSTM on the IMDB dataset."""
    max_features = 10000
    maxlen = 500
    batch_size = 32
    print('Loading data.')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(
        num_words=max_features)
    print('{0} training sequences.'.format(len(input_train)))
    print('{0} test sequences.'.format(len(input_test)))
    print('Pad sequences (samples x time).')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape: {0}'.format(input_train.shape))
    print('input_test shape: {0}'.format(input_test.shape))
    model = Sequential()
    model.add(Embedding(max_features, 32, input_length=maxlen))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    start = time.time()
    history = model.fit(input_train,
                         y_train,
                         epochs=10,
                         batch_size=128,
                         validation_split=0.2)
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    plot_history(history)


def main():
    """Runs the program."""
    #numpy_rnn()
    #imdb_rnn()
    imdb_lstm()


if __name__ == '__main__':
    main()
