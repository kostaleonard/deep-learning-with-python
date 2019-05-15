# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DATA_DIR = '/Users/leo/Downloads/jena_climate'


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


def evaluate_naive_method(val_gen, val_steps, std):
    """Predicts the temperature for any given day to be the temperature
    from 24 hours prior."""
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    m = np.mean(batch_maes)
    print('Naive MAE: {0} = {1} degrees C'.format(m, m * std[1]))


def evaluate_dense_network(float_data, lookback, step, train_gen, val_gen,
                           val_steps):
    """Predicts the temperature using a Dense network. This is another
    baseline."""
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step,
                                          float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_history(history)


def evaluate_rnn_baseline(float_data, lookback, step, train_gen, val_gen,
                          val_steps):
    """Predicts the temperature using an RNN (v1)."""
    model = Sequential()
    #model.add(layers.GRU(32, input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    print(model.summary())
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_history(history)


def evaluate_rnn_dropout(float_data, lookback, step, train_gen, val_gen,
                         val_steps):
    """Predicts the temperature using an RNN with dropout (v2)."""
    model = Sequential()
    #model.add(layers.GRU(32, input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.GRU(32,
                         dropout=0.2,
                         recurrent_dropout=0.2,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    print(model.summary())
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_history(history)


def evaluate_rnn_stacked(float_data, lookback, step, train_gen, val_gen,
                         val_steps):
    """Predicts the temperature using an RNN with stacked layers (v3)."""
    model = Sequential()
    #model.add(layers.GRU(32, input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64, activation='relu',
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    print(model.summary())
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_history(history)


def evaluate_rnn_bidirectional(float_data, lookback, step, train_gen, val_gen,
                               val_steps):
    """Predicts the temperature using a bidirectional RNN (v3). This
    architecture doesn't work well for this problem because the most
    important information in the data is the most recent (i.e. the best
    predictor of tomorrow's weather is yesterday's weather, not weather
    from 5 days ago)."""
    model = Sequential()
    #model.add(layers.GRU(32, input_shape=(lookback // step, float_data.shape[-1])))
    model.add(layers.Bidirectional(
        layers.GRU(32), input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    print(model.summary())
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    plot_history(history)


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    """Generates an infinite stream of data to train or validate the
    neural network. data should be normalized already. By default,
    takes one point per hour."""
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def weather_rnn():
    """Runs an RNN on the weather dataset."""
    fname = os.path.join(DATA_DIR, 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print(header)
    print(len(lines))
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    temp = float_data[:, 1]
    #plt.plot(range(len(temp)), temp)
    #plt.show()
    # Normalize data.
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
    # Get the training, validation, and test generators.
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)
    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(float_data) - 300001 - lookback) // batch_size
    evaluate_naive_method(val_gen, val_steps, std)
    #evaluate_dense_network(float_data, lookback, step, train_gen, val_gen,
    #                       val_steps)
    #evaluate_rnn_baseline(float_data, lookback, step, train_gen, val_gen,
    #                      val_steps)
    #evaluate_rnn_dropout(float_data, lookback, step, train_gen, val_gen,
    #                     val_steps)
    evaluate_rnn_stacked(float_data, lookback, step, train_gen, val_gen,
                         val_steps)
    #evaluate_rnn_bidirectional(float_data, lookback, step, train_gen, val_gen,
    #                           val_steps)


def main():
    """Runs the program."""
    weather_rnn()


if __name__ == '__main__':
    main()
