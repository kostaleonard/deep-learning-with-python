# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import random
import sys
import numpy as np
import keras
from keras import layers


def reweight_distribution(original_distribution, temperature=0.5):
    """Returns a new distribution that has reweighted probability
    values according to the temperature. A higher temperature means
    higher entropy, or randomness."""
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def download_data():
    """Downloads the Nietzsche dataset."""
    path = keras.utils.get_file(
        'nietzsche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    # TODO looks like you're opening a file without a close.
    text = open(path).read().lower()
    print('Corpus length: {0}'.format(len(text)))
    return text


def sample(preds, temperature=1.0):
    """Returns a sample from the given distribution using a
    temperature. Higher temperature means more entropy."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def run_lstm_model(text):
    """Runs the LSTM model on the given text."""
    maxlen = 60
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Number of sequences: {0}'.format(len(sentences)))
    chars = sorted(list(set(text)))
    print('Unique characters: {0}'.format(len(chars)))
    char_indices = dict((char, chars.index(char)) for char in chars)
    print('Vectorization.')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    for epoch in range(1, 60):
        print('Epoch {0}'.format(epoch))
        model.fit(x, y, batch_size=128, epochs=1)
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('--- Generating with seed: {0}'.format(generated_text))
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ Temperature: {0}'.format(temperature))
            sys.stdout.write(generated_text)
            for i in range(400):
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]
                generated_text += next_char
                generated_text = generated_text[1:]
                sys.stdout.write(next_char)


def main():
    """Runs the program."""
    run_lstm_model(download_data())


if __name__ == '__main__':
    main()
