# Leonard R. Kosta Jr.

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
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


def reverse_lstm():
    """Trains and evaluates an LSTM using reversed text sequences from
    the IMDB dataset."""
    max_features = 10000
    maxlen = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    plot_history(history)


def bidirectional_lstm():
    """Trains and evaluates a bidirectional LSTM using reversed text
    sequences from the IMDB dataset."""
    max_features = 10000
    maxlen = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)
    x_train = [x[::-1] for x in x_train]
    x_test = [x[::-1] for x in x_test]
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    model = Sequential()
    model.add(layers.Embedding(max_features, 32))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    plot_history(history)


def main():
    """Runs the program."""
    #reverse_lstm()
    bidirectional_lstm()


if __name__ == '__main__':
    main()
