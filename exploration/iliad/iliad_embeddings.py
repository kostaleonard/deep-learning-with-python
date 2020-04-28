# TODO learn the embeddings of all the characters in the Iliad and visualize them in a 2D space.
# TODO what can we tell from the relationships? What is the relationship of Achilles and Hector? Achilles and Patroclus?

import re
import itertools
import random
import numpy as np
from graphviz import Graph
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Embedding, Input

DICTIONARY_FILENAME = '../../resources/dictionary_words.txt'
ILIAD_FILENAME = '../../resources/iliad.txt'
ILIAD_NAMES_FILENAME = '../../resources/iliad_names.txt'
DATASET_PREFIX = '../../resources/iliad_embeddings_dataset'
GRAPH_FILENAME = '../../resources/iliad_relationships'
EMBEDDINGS_PLOT_FILENAME = '../../resources/iliad_embeddings.png'
# TODO deities flag.
INCLUDE_DEITIES = True
EMBEDDING_NAME_FILTER = {
    'achilles', 'hector', 'priam', 'jove', 'juno', 'minerva', 'patroclus', 'neptune', 'menelaus', 'agamemnon'
}
RANDOM_SEED = 52017
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 1.0 - TRAIN_SPLIT - VAL_SPLIT
# Hyperparameters.
INPUT_SEQUENCE_LENGTH = 2
WINDOW_SIZE = 200
EMBEDDING_SIZE = 2
EPOCHS = 20
BATCH_SIZE = 32


def get_dictionary_words(dictionary_filename):
    """Returns all of the words in a standard dictionary, as a set."""
    with open(dictionary_filename, 'r') as infile:
        words = infile.read().lower().split('\n')
    return set(words)


def get_file_contents(filename):
    """Returns the contents of the given file."""
    with open(filename, 'r') as infile:
        return infile.read()


def get_lines(filename):
    """Returns the lines of the given file as a list."""
    with open(filename, 'r') as infile:
        return [line.strip().lower() for line in infile.read().split('\n')]


def get_sorted_proper_nouns(original_text):
    """Returns the proper nouns of the text, sorted."""
    text_lower = original_text.lower().replace('\n', ' ')
    words = list(set(text_lower.split(' ')))
    for i, word in enumerate(words):
        words[i] = re.sub('[^A-Za-z0-9]+', '', word)
    words = filter(lambda word: word != '', words)
    words = filter(lambda word: word[0].upper() + word[1:] in original_text, words)
    return sorted(list(set(words)))


def get_non_dictionary_words(words_to_filter, dictionary_words):
    """Returns the words from words_to_filter that do not appear in dictionary_words.
    NOTE: this code works, but the problem is that important proper nouns (e.g., "Achilles") appear in the dictionary.
    If you don't filter on dictionary words, you get way too many useless words."""
    return list(filter(lambda word: word not in dictionary_words, words_to_filter))


def write_words(words, filename):
    """Writes the words to the given filename."""
    new_words = []
    for w in words:
        new_words.append(w)
        new_words.append('\n')
    new_words = new_words[:-1]
    with open(filename, 'w') as outfile:
        outfile.writelines(new_words)


def get_samples(keywords, text):
    """Returns a dict where the keys are INPUT_SEQUENCE_LENGTH-tuples of words and the values are 1 if those words
    ever appear in a WINDOW_SIZE window in the text, 0 otherwise. This will be used to build the dataset. Because order
    matters in tuples, every possible permutation of the same words will have the same label, and will be different
    samples."""
    # TODO could be faster, but not urgent if we expect to do this just once.
    proper_noun_perms = list(itertools.permutations(keywords, INPUT_SEQUENCE_LENGTH))
    samples_dict = {k: 0 for k in proper_noun_perms}
    text_lower_words = text.lower().replace('\n', ' ').split(' ')
    for i, text_word in enumerate(text_lower_words):
        text_lower_words[i] = re.sub('[^A-Za-z0-9]+', '', text_word)
    text_lower_words = list(filter(lambda word: word != '', text_lower_words))
    for i in range(len(text_lower_words) - WINDOW_SIZE):
        if i % 10000 == 0:
            print('Iteration {0}'.format(i))
        window = text_lower_words[i: i + WINDOW_SIZE]
        proper_nouns_in_window = list(filter(lambda word: word in keywords, list(set(window))))
        proper_noun_window_perms = list(itertools.permutations(proper_nouns_in_window, INPUT_SEQUENCE_LENGTH))
        for tup in proper_noun_window_perms:
            samples_dict[tup] += 1
    return samples_dict


def get_keyword_index(keywords):
    """Returns the keyword indices for all of the keywords. This is just a map that takes indices and maps them to
    their keywords entries, allowing for (near) constant time reverse lookups."""
    return dict(zip(keywords, range(len(keywords))))


def get_dataset_np_arrays(keylist, keywords, samples):
    """Returns the x, y np arrays for the given keylist. The keylist is of shape (samples, INPUT_SEQUENCE_LENGTH), where
    each sample is a combination of INPUT_SEQUENCE_LENGTH keywords. samples is the dict of samples where the key is the
    combination of INPUT_SEQUENCE_LENGTH keywords, and the value is whether or not those keywords were used in the same
    window in the text."""
    keyword_index = get_keyword_index(keywords)
    x = np.zeros((len(keylist), INPUT_SEQUENCE_LENGTH))
    y = np.zeros(len(keylist))
    for i, sample in enumerate(keylist):
        for j, keyword in enumerate(sample):
            index = keyword_index[keyword]
            x[i, j] = index
        if samples[sample] > 0:
            y[i] = 1.0
    return x, y


def get_dataset(keywords, text, save_samples=True):
    """Returns the dataset as a 3-tuple of 2-tuples (IAW standard keras notation): (x_train, y_train), (x_val, y_val),
    (x_test, y_test)."""
    try:
        x_train = np.load('{0}_x_train.npy'.format(DATASET_PREFIX))
        y_train = np.load('{0}_y_train.npy'.format(DATASET_PREFIX))
        x_val = np.load('{0}_x_val.npy'.format(DATASET_PREFIX))
        y_val = np.load('{0}_y_val.npy'.format(DATASET_PREFIX))
        x_test = np.load('{0}_x_test.npy'.format(DATASET_PREFIX))
        y_test = np.load('{0}_y_test.npy'.format(DATASET_PREFIX))
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    except FileNotFoundError:
        print('No dataset found; building dataset.')
    samples = get_samples(keywords, text)
    visualize_graph(samples, keywords)
    shuffled_keys = list(samples.keys())
    random.shuffle(shuffled_keys)
    keys_train = shuffled_keys[:int(len(shuffled_keys) * TRAIN_SPLIT)]
    keys_val = shuffled_keys[len(keys_train):len(keys_train) + int(len(shuffled_keys) * VAL_SPLIT)]
    keys_test = shuffled_keys[len(keys_train) + len(keys_val):]
    x_train, y_train = get_dataset_np_arrays(keys_train, keywords, samples)
    x_val, y_val = get_dataset_np_arrays(keys_val, keywords, samples)
    x_test, y_test = get_dataset_np_arrays(keys_test, keywords, samples)
    if save_samples:
        np.save('{0}_x_train.npy'.format(DATASET_PREFIX), x_train)
        np.save('{0}_y_train.npy'.format(DATASET_PREFIX), y_train)
        np.save('{0}_x_val.npy'.format(DATASET_PREFIX), x_val)
        np.save('{0}_y_val.npy'.format(DATASET_PREFIX), y_val)
        np.save('{0}_x_test.npy'.format(DATASET_PREFIX), x_test)
        np.save('{0}_y_test.npy'.format(DATASET_PREFIX), y_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_model(keywords):
    """Returns the model as an instance of keras.models.Model. The model assumes that the order of keywords in each
    input sequence does not matter (this assumption is true for our dataset)."""
    model = Sequential()
    model.add(Embedding(len(keywords), EMBEDDING_SIZE, input_length=INPUT_SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def get_multi_output_model(keywords):
    """Returns the model as an instance of keras.models.Model. The returned model will have 2 outputs: the embeddings
    and the classifier results. This allows us to read the embeddings. Embeddings do not contribute to model loss."""
    input_layer = Input(shape=(INPUT_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(len(keywords), EMBEDDING_SIZE, input_length=INPUT_SEQUENCE_LENGTH,
                                name='embedding')(input_layer)
    flatten_layer = Flatten()(embedding_layer)
    dense_layer = Dense(1, activation='sigmoid', name='classification')(flatten_layer)
    model = Model(input_layer, [embedding_layer, dense_layer])
    model.compile(optimizer='rmsprop',
                  loss={'classification': 'binary_crossentropy',
                        'embedding': 'mse'},
                  loss_weights={'classification': 1.0,
                                'embedding': 0.0},
                  metrics=['acc'])
    return model


def train_model(model, train_dataset, val_dataset):
    """Trains the model on the dataset. Returns the history."""
    x_train, y_train = train_dataset
    x_val, y_val = val_dataset
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val, y_val))
    return history


def train_mulit_output_model(model, train_dataset, val_dataset):
    """Trains the model on the dataset. Returns the history."""
    x_train, y_train = train_dataset
    x_val, y_val = val_dataset
    embedding_dummy_labels_train = np.zeros((y_train.shape[0], INPUT_SEQUENCE_LENGTH, EMBEDDING_SIZE))
    embedding_dummy_labels_val = np.zeros((y_val.shape[0], INPUT_SEQUENCE_LENGTH, EMBEDDING_SIZE))
    history = model.fit(x_train,
                        {'classification': y_train,
                         'embedding': embedding_dummy_labels_train},
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val,
                                         {'classification': y_val,
                                          'embedding': embedding_dummy_labels_val}))
    return history


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


def plot_multi_output_history(history, smooth_fac=0.0):
    """Plots the given history object."""
    acc = history.history['classification_acc']
    val_acc = history.history['val_classification_acc']
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


def visualize_graph(samples, keywords):
    """Displays the graph of connections between keywords. samples is the dict of samples where the key is the
    combination of INPUT_SEQUENCE_LENGTH keywords, and the value is whether or not those keywords were used in the same
    window in the text."""
    g = Graph(name='iliad_relationships')
    adj = np.zeros((len(keywords), len(keywords)))
    for i in range(len(keywords)):
        for j in range(i):
            kw0 = keywords[i]
            kw1 = keywords[j]
            adj[i, j] = samples[(kw0, kw1)]
    flat_adj = np.reshape(adj, -1)
    flat_adj = np.argsort(flat_adj)[::-1]
    flat_adj = flat_adj[:30]
    for k in range(len(flat_adj)):
        i = flat_adj[k] // len(adj)
        j = flat_adj[k] % len(adj)
        kw0 = keywords[i]
        kw1 = keywords[j]
        g.edge(kw0, kw1)
    print('Rendering graph.')
    g.render(filename=GRAPH_FILENAME, format='png')
    print('Done.')


def visualize_embeddings(keywords, model, name_filter=None):
    """Displays the embeddings on a 2D plot. The name_filter argument can be optionally supplied to limit the number of
    keywords that are plotted; the argument should be a set of string names."""
    if name_filter:
        zipped_keywords = [(i, kw) for (i, kw) in enumerate(keywords) if kw in name_filter]
    else:
        zipped_keywords = [(i, kw) for (i, kw) in enumerate(keywords)]
    model_input = np.zeros((len(zipped_keywords), INPUT_SEQUENCE_LENGTH))
    for i, (kw_index, kw) in enumerate(zipped_keywords):
        model_input[i, 0] = kw_index
    preds = model.predict(model_input)
    word_embeddings = np.array([arr[0] for arr in preds[0]])
    print(word_embeddings)
    if word_embeddings.shape[1] != 2:
        raise NotImplementedError('t-SNE not yet implemented.')
    for i, row in enumerate(word_embeddings):
        plt.scatter(row[0], row[1], label=zipped_keywords[i][1])
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('Embedding Dimension 0')
    plt.ylabel('Embedding Dimension 1')
    plt.title('Embeddings of Key Character Names in the Iliad')
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
    plt.savefig(EMBEDDINGS_PLOT_FILENAME, bbox_inches="tight")


def main():
    """Runs the program."""
    random.seed(a=RANDOM_SEED)
    iliad_text = get_file_contents(ILIAD_FILENAME)
    iliad_names = get_lines(ILIAD_NAMES_FILENAME)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset(iliad_names, iliad_text)
    print('Total number of examples: {0}'.format(len(x_train) + len(x_val) + len(x_test)))
    print('Number of training examples: {0}'.format(len(x_train)))
    print('Number of validation examples: {0}'.format(len(x_val)))
    print('Number of test examples: {0}'.format(len(x_test)))
    print('Number of positive examples: {0}'.format(int(sum(y_train) + sum(y_val) + sum(y_test))))
    model = get_multi_output_model(iliad_names)
    model.summary()
    history = train_mulit_output_model(model, (x_train, y_train), (x_val, y_val))
    #plot_multi_output_history(history)
    visualize_embeddings(iliad_names, model, name_filter=EMBEDDING_NAME_FILTER)
    #visualize_embeddings(iliad_names, model)
    # TODO t-SNE if not 2D already
    # TODO plot embeddings


if __name__ == '__main__':
    main()
