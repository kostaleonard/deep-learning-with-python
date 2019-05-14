# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

IMDB_DIR = '/Users/leo/Downloads/aclImdb'
GLOVE_DIR = '/Users/leo/Downloads/glove'
MODEL_SAVE_DIR = '/Users/leo/tmp/'


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


def example_one_hot():
    """Listing 6.3: using Keras for word-level one-hot encoding."""
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    print('Printing samples and their word index sequences.')
    for i in range(len(samples)):
        print(samples[i])
        print(sequences[i])
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    print('Printing samples and their bag-of-words sequences.')
    for i in range(len(samples)):
        print(samples[i])
        print(one_hot_results[i])
    word_index = tokenizer.word_index
    print('Found {0} unique tokens.'.format(len(word_index)))


def imdb_embedding():
    """Uses an embedding layer and classifier on the IMDB data."""
    max_features = 10000
    maxlen = 20
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    model = Sequential()
    model.add(Embedding(max_features, 8, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)


def imdb_embedding_raw_pretrained():
    """Runs embedding on the IMDB data, but uses the raw IMDB data
    rather than that prepackaged in Keras."""
    train_dir = os.path.join(IMDB_DIR, 'train')
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    maxlen = 100
    training_samples = 200
    validation_samples = 10000
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {0} unique tokens.'.format(len(word_index)))
    data = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)
    print('Shape of data tensor: {0}'.format(data.shape))
    print('Shape of label tensor: {0}'.format(labels.shape))
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found {0} word vectors.'.format(len(embeddings_index)))
    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Load the embeddings.
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights(os.path.join(MODEL_SAVE_DIR,
                                    'pre_trained_glove_model.h5'))
    plot_history(history)


def imdb_embedding_raw_from_scratch():
    """Runs embedding on the IMDB data, but uses the raw IMDB data
    rather than that prepackaged in Keras. The model is not pretrained."""
    train_dir = os.path.join(IMDB_DIR, 'train')
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    maxlen = 100
    training_samples = 200
    validation_samples = 10000
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {0} unique tokens.'.format(len(word_index)))
    data = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(labels)
    print('Shape of data tensor: {0}'.format(data.shape))
    print('Shape of label tensor: {0}'.format(labels.shape))
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found {0} word vectors.'.format(len(embeddings_index)))
    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Don't load the embeddings.
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    plot_history(history)
    test_dir = os.path.join(IMDB_DIR, 'test')
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in sorted(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    sequences = tokenizer.texts_to_sequences(texts)
    x_test = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
    y_test = np.asarray(labels)
    # Uses the pretrained model:
    #model.load_weights(os.path.join(MODEL_SAVE_DIR,
    #                                'pre_trained_glove_model.h5'))
    results = model.evaluate(x_test, y_test)
    print(results)


def main():
    """Runs the program."""
    #example_one_hot()
    #imdb_embedding()
    #imdb_embedding_raw_pretrained()
    imdb_embedding_raw_from_scratch()


if __name__ == '__main__':
    main()
