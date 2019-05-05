# Leonard R. Kosta Jr.

from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])


def index_to_word(index):
    """Returns the word at the given index. Indices 0, 1, and 2 are
    reserved."""
    return reverse_word_index.get(index - 3, '?')


def vector_to_sentence(v):
    """Returns the sentence from the given input vector from the
    training data."""
    return ' '.join([index_to_word(i) for i in v])


def vectorize_sequences(sequences, dimensions=10000):
    """Returns a numpy matrix representing each sequence as a one-hot
    vector of the specified dimensions."""
    result = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    return result


def get_model():
    """Returns the model."""
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
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


def main():
    """Runs the program."""
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000)
    print(train_data.shape)
    print(vector_to_sentence(train_data[0]))
    print(train_labels[:10])
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    model = get_model()
    # If you try to fit for 20 epochs, you will overfit.
    # Based on those results, you can see that 4 epochs is optimal.
    epochs = 4
    history = model.fit(partial_x_train, partial_y_train, epochs=epochs,
                        batch_size=512, validation_data=(x_val, y_val))
    plot_history(history)
    results = model.evaluate(x_test, y_test)
    print('Results: {0}'.format(results))
    predictions = model.predict(x_test)
    print('Predictions:\n{0}'.format(predictions[:10]))


if __name__ == '__main__':
    main()
