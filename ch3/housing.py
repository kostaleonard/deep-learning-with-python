# Leonard R. Kosta Jr.

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import copy
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def normalize(train_data, test_data):
    """Returns a tuple of normalized training and testing datasets."""
    train_data_copy = copy.copy(train_data)
    test_data_copy = copy.copy(test_data)
    mean = train_data_copy.mean(axis=0)
    train_data_copy -= mean
    std = train_data_copy.std(axis=0)
    train_data_copy /= std
    test_data_copy -= mean
    test_data_copy /= std
    return (train_data_copy, test_data_copy)


def get_model(input_shape):
    """Returns the model."""
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def smooth_curve(points, factor=0.9):
    """Returns points smoothed over an exponential."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_history(average_mae_history):
    """Plots the history object."""
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


def main():
    """Runs the program."""
    (train_data, train_targets), (test_data, test_targets) = \
        boston_housing.load_data()
    print(train_data.shape)
    (train_data, test_data) = normalize(train_data, test_data)
    input_shape = train_data.shape[1:]
    k = 4
    num_val_examples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    start = time.time()
    for i in range(k):
        print('Processing fold {0}.'.format(i))
        val_data = train_data[i * num_val_examples:(i + 1) * num_val_examples]
        val_targets = train_targets[i * num_val_examples:
                                    (i + 1) * num_val_examples]
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_examples],
             train_data[(i + 1) * num_val_examples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_examples],
             train_targets[(i + 1) * num_val_examples:]],
            axis=0)
        model = get_model(input_shape)
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)
    stop = time.time()
    print('Time: {0}'.format(stop - start))
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    plot_history(average_mae_history)
    print('Training final model.')
    model = get_model(input_shape)
    model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print('Test score: {0}'.format(test_mae_score))


if __name__ == '__main__':
    main()
