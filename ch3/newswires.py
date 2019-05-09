# Leonard R. Kosta Jr.

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def to_one_hot(labels, dimensions=46):
    """Returns a numpy matrix representing each label as a one-hot
    vector of the specified dimensions.

    NOTE: Use Keras built-in to_categorical instead.
    """
    result = np.zeros((len(labels), dimensions))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result


def vectorize_sequences(sequences, dimensions=10000):
    """Returns a numpy matrix representing each sequence as a one-hot
    vector of the specified dimensions."""
    result = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    return result


def example_to_text(example):
    """Returns a string representing the training example as text."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in example])


def get_model():
    """Returns the model."""
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history):
    """Plots the history object."""
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def get_random_baseline_acc(test_labels):
    """Returns the accuracy of a random baseline on the test dataset."""
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(test_labels) == np.array(test_labels_copy)
    return float(np.sum(hits_array)) / len(test_labels)


def main():
    """Runs the program."""
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000)
    print(train_data[0])
    print(max(train_labels))
    print(example_to_text(train_data[5]))
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    model = get_model()
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    results = model.evaluate(x_test, one_hot_test_labels)
    print('Results: {0}'.format(results))
    predictions = model.predict(x_test)
    print('Predictions:\n{0}'.format(predictions[:10]))
    print('Baseline performance: {0}'.format(get_random_baseline_acc(
        test_labels)))
    plot_history(history)


if __name__ == '__main__':
    main()
