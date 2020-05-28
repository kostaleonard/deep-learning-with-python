import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import shutil
import cv2
import json
from exploration.hateful_memes.TextOnlyDataGenerator import TextOnlyDataGenerator
# If you don't have plaidml, get rid of this line and the try-except:
from plaidml.exceptions import Unknown

DATASET_BASE_DIR = '/Users/leo/Documents/Datasets/hateful_memes_data'
TRAIN_LINES = os.path.join(DATASET_BASE_DIR, 'train.jsonl')
DEV_LINES = os.path.join(DATASET_BASE_DIR, 'dev.jsonl')
TEST_LINES = os.path.join(DATASET_BASE_DIR, 'test.jsonl')
TEXT_DIR = os.path.join(DATASET_BASE_DIR, 'text')
MODEL_SAVE_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/saved_models'
PREDICTION_OUTPUT_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/predictions'
# There are 9691 words in the dataset.
TOKENIZER_NUM_WORDS = 9691
# Longest sequence is 77 words.
TEXT_SEQUENCE_LENGTH = 30
EPOCHS = 10
BATCH_SIZE = 20
IMAGE_SCALE_SIZE = (150, 150)
NUM_CHANNELS = 3
MODEL_INPUT_DIM = IMAGE_SCALE_SIZE + (NUM_CHANNELS,)
EMBEDDING_SIZE = 8


def smooth_curve(points, factor=0.0):
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
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def deprocess_image(x):
    """Returns a tensor with uint8 image values corresponding to the
    floating point values in x."""
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def build_dataset(partition):
    """Builds the dataset from the download directory. The partition
    defines the image files used in training, development, and testing."""
    if not os.path.exists(TEXT_DIR):
        os.mkdir(TEXT_DIR)
    num_existing = len(os.listdir(TEXT_DIR))
    if num_existing == len(partition['train']) + \
            len(partition['dev']) + \
            len(partition['test']):
        print('Dataset appears to be built already; skipping build.')
        return
    with open(TRAIN_LINES, 'r') as infile:
        json_lines = [json.loads(line) for line in infile.read().splitlines()]
        for line in json_lines:
            id = line['img'].split('/')[1]
            if id in partition['train']:
                dst = os.path.join(TEXT_DIR, id).replace('.png', '.txt')
                with open(dst, 'w') as outfile:
                    outfile.write(line['text'])
    with open(DEV_LINES, 'r') as infile:
        json_lines = [json.loads(line) for line in infile.read().splitlines()]
        for line in json_lines:
            id = line['img'].split('/')[1]
            if id in partition['dev']:
                dst = os.path.join(TEXT_DIR, id).replace('.png', '.txt')
                with open(dst, 'w') as outfile:
                    outfile.write(line['text'])
    with open(TEST_LINES, 'r') as infile:
        json_lines = [json.loads(line) for line in infile.read().splitlines()]
        for line in json_lines:
            id = line['img'].split('/')[1]
            if id in partition['test']:
                dst = os.path.join(TEXT_DIR, id).replace('.png', '.txt')
                with open(dst, 'w') as outfile:
                    outfile.write(line['text'])
    print('Total texts: {0}'.format(len(os.listdir(TEXT_DIR))))


def get_texts(partition):
    """Returns a list of texts from the files in TEXT_DIR. Order doesn't
    matter here because this is only used to fit the tokenizer."""
    result = []
    filenames = partition['train'] + partition['dev'] + partition['test']
    for filename in filenames:
        with open(os.path.join(TEXT_DIR, filename.replace('.png', '.txt')), 'r') as infile:
            result.append(infile.read().strip())
    return result


def get_values_and_labels(json_filename):
    """Returns a 2-tuple where the first element is the list of
    filenames specified in the json file, and the second element is the
    labels dict for those filenames."""
    values = []
    labels = {}
    with open(json_filename) as infile:
        json_lines = [json.loads(line) for line in infile.read().splitlines()]
        for line in json_lines:
            values.append(line['img'].split('/')[1])
            if 'label' in line:
                labels[line['img'].split('/')[1]] = line['label']
    return values, labels


def get_partition_and_labels():
    """Returns a 2-tuple where the first element is the dict
    representing the dataset partition and the second is a dict
    representing the labels.
    The keys for the dataset partition are: train, dev, test. The values
    are the list of filenames that fall in each category
    The keys for the labels dict are the filenames, and the values are
    the binary labels as integers."""
    partition = {}
    labels = {}
    train_values, train_labels = get_values_and_labels(TRAIN_LINES)
    partition['train'] = train_values
    labels.update(train_labels)
    dev_values, dev_labels = get_values_and_labels(DEV_LINES)
    partition['dev'] = dev_values
    labels.update(dev_labels)
    test_values, test_labels = get_values_and_labels(TEST_LINES)
    partition['test'] = test_values
    labels.update(test_labels)
    return partition, labels


def train_model(model, partition, labels, tokenizer, augment=False):
    """Trains the model."""
    train_generator = TextOnlyDataGenerator(
        partition['train'], labels, TEXT_SEQUENCE_LENGTH, TEXT_DIR, 2, tokenizer)
    dev_generator = TextOnlyDataGenerator(
        partition['dev'], labels, TEXT_SEQUENCE_LENGTH, TEXT_DIR, 2, tokenizer)
    start = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(partition['train']) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=dev_generator,
        validation_steps=len(partition['dev']) // BATCH_SIZE,
        class_weight={0: 1, 1: 3})
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    model_fname = os.path.join(MODEL_SAVE_DIR, 'text_{0}'.format(str(datetime.now()).replace(' ', '_')))
    model.save(model_fname)
    plot_history(history)


def get_model():
    """Returns the model."""
    model = Sequential()
    model.add(layers.Embedding(TOKENIZER_NUM_WORDS, EMBEDDING_SIZE, input_length=TEXT_SEQUENCE_LENGTH))
    model.add(layers.SimpleRNN(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def test_labels(partition, labels, tokenizer):
    """Checks the labels to make sure that they are getting fed into
    the model correctly. This is just a quick and dirty check, not
    meant to be pretty."""
    train_generator = TextOnlyDataGenerator(
        partition['train'], labels, TEXT_SEQUENCE_LENGTH, TEXT_DIR, 2, tokenizer)
    num_mismatched = 0
    sample_count = 0
    for batch in train_generator:
        samples, batch_labels = batch
        for i in range(len(samples)):
            sample = samples[i] if 0 not in samples[i] else samples[i][:list(samples[i]).index(0)]
            sentence = ' '.join([tokenizer.index_word[int(word_num)] for word_num in sample])
            for fname in os.listdir(TEXT_DIR):
                with open(os.path.join(TEXT_DIR, fname), 'r') as infile:
                    if sentence in infile.read().strip():
                        if fname.replace('.txt', '.png') in labels and labels[fname.replace('.txt', '.png')] != batch_labels[i]:
                            print('Mismatched labels: {0}---{1}'.format(sentence, fname))
                            num_mismatched += 1
                            break
            sample_count += 1
        if sample_count >= 1000:
            break
    print('Number mismatched: {0}/{1}'.format(num_mismatched, sample_count))


def main():
    """Runs the program."""
    partition, labels = get_partition_and_labels()
    print('Training examples: {0}'.format(len(partition['train'])))
    print('Validation examples: {0}'.format(len(partition['dev'])))
    print('Test examples: {0}'.format(len(partition['test'])))
    print('Fraction positive examples (train, val only): {0}/{1}'.format(sum(labels.values()), len(labels.keys())))
    build_dataset(partition)
    texts = get_texts(partition)
    tokenizer = Tokenizer(num_words=TOKENIZER_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    model = get_model()
    train_model(model, partition, labels, tokenizer)


if __name__ == '__main__':
    main()
