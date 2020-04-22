import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import random
import sys
import numpy as np
import keras
from keras import layers

ILIAD_PATH = '../../resources/iliad.txt'
MAX_LEN = 60
STEP = 3


def get_model(chars):
    """Returns the model as an instance of keras.models."""
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(MAX_LEN, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    # TODO metrics??
    model.compile(optimizer, loss='categorical_crossentropy')
    return model


def sample(preds, temperature=1.0):
    """Returns a character from the model, reweighting the dirstribution with temperature."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def main():
    """Runs the program."""
    with open(ILIAD_PATH, 'r') as infile:
        text = infile.read().replace('\n', ' ').lower()
    print('Corpus length: {0}'.format(len(text)))
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAX_LEN, STEP):
        sentences.append(text[i:i + MAX_LEN])
        next_chars.append(text[i + MAX_LEN])
    chars = sorted(list(set(text)))
    print('Unique characters: {0}'.format(len(chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    print('Vectorization...')
    x = np.zeros((len(sentences), MAX_LEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    model = get_model(chars)
    for epoch in range(1, 60):
        print('epoch {0}'.format(epoch))
        # TODO can this batch_size param be different from the LSTM layer size? What are the effects?
        model.fit(x, y, batch_size=128, epochs=1)
        start_index = random.randint(0, len(text) - MAX_LEN - 1)
        generated_text = text[start_index:start_index + MAX_LEN]
        print('--- Generating with seed: "{0}"'.format(generated_text))
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ temperature: {0}'.format(temperature))
            sys.stdout.write(generated_text)
            for i in range(400):
                sampled = np.zeros((1, MAX_LEN, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1.0
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature=temperature)
                next_char = chars[next_index]
                generated_text += next_char
                generated_text = generated_text[1:]
                sys.stdout.write(next_char)


if __name__ == '__main__':
    main()
