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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import shutil
import cv2
import json
from exploration.hateful_memes.ImageOnlyDataGenerator import DataGenerator
# If you don't have plaidml, get rid of this line and the try-except:
from plaidml.exceptions import Unknown

DATASET_BASE_DIR = '/Users/leo/Documents/Datasets/hateful_memes_data'
RAW_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'img')
TRAIN_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'train')
TRAIN_IMAGE_POS_DIR = os.path.join(TRAIN_IMAGE_DIR, 'pos')
TRAIN_IMAGE_NEG_DIR = os.path.join(TRAIN_IMAGE_DIR, 'neg')
DEV_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'dev')
DEV_IMAGE_POS_DIR = os.path.join(DEV_IMAGE_DIR, 'pos')
DEV_IMAGE_NEG_DIR = os.path.join(DEV_IMAGE_DIR, 'neg')
TEST_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'test')
TEST_IMAGE_POS_DIR = os.path.join(TEST_IMAGE_DIR, 'pos')
TEST_IMAGE_NEG_DIR = os.path.join(TEST_IMAGE_DIR, 'neg')
MODEL_SAVE_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/saved_models'
PREDICTION_OUTPUT_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/predictions'
EPOCHS = 10
BATCH_SIZE = 20
IMAGE_SCALE_SIZE = (150, 150)
NUM_CHANNELS = 3
MODEL_INPUT_DIM = IMAGE_SCALE_SIZE + (NUM_CHANNELS,)


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


def build_dataset(partition, labels):
    """Builds the dataset from the download directory. The partition
    defines the image files used in training, development, and testing."""
    if not os.path.exists(TRAIN_IMAGE_DIR):
        os.mkdir(TRAIN_IMAGE_DIR)
    if not os.path.exists(TRAIN_IMAGE_POS_DIR):
        os.mkdir(TRAIN_IMAGE_POS_DIR)
    if not os.path.exists(TRAIN_IMAGE_NEG_DIR):
        os.mkdir(TRAIN_IMAGE_NEG_DIR)
    if not os.path.exists(DEV_IMAGE_DIR):
        os.mkdir(DEV_IMAGE_DIR)
    if not os.path.exists(DEV_IMAGE_POS_DIR):
        os.mkdir(DEV_IMAGE_POS_DIR)
    if not os.path.exists(DEV_IMAGE_NEG_DIR):
        os.mkdir(DEV_IMAGE_NEG_DIR)
    if not os.path.exists(TEST_IMAGE_DIR):
        os.mkdir(TEST_IMAGE_DIR)
    if not os.path.exists(TEST_IMAGE_POS_DIR):
        os.mkdir(TEST_IMAGE_POS_DIR)
    if not os.path.exists(TEST_IMAGE_NEG_DIR):
        os.mkdir(TEST_IMAGE_NEG_DIR)
    num_existing_train = len(os.listdir(TRAIN_IMAGE_POS_DIR)) + \
        len(os.listdir(TRAIN_IMAGE_NEG_DIR))
    num_existing_dev = len(os.listdir(DEV_IMAGE_POS_DIR)) + \
        len(os.listdir(DEV_IMAGE_NEG_DIR))
    num_existing_test = len(os.listdir(TEST_IMAGE_POS_DIR)) + \
        len(os.listdir(TEST_IMAGE_NEG_DIR))
    if num_existing_train == len(partition['train']) and \
            num_existing_dev == len(partition['dev']) and \
            num_existing_test == len(partition['test']):
        print('Dataset appears to be built already; skipping build.')
        return
    for fname in partition['train']:
        src = os.path.join(RAW_IMAGE_DIR, fname)
        if labels[fname] == 1:
            dst = os.path.join(TRAIN_IMAGE_POS_DIR, fname)
        else:
            dst = os.path.join(TRAIN_IMAGE_NEG_DIR, fname)
        shutil.copyfile(src, dst)
    for fname in partition['dev']:
        src = os.path.join(RAW_IMAGE_DIR, fname)
        if labels[fname] == 1:
            dst = os.path.join(DEV_IMAGE_POS_DIR, fname)
        else:
            dst = os.path.join(DEV_IMAGE_NEG_DIR, fname)
        shutil.copyfile(src, dst)
    for fname in partition['test']:
        src = os.path.join(RAW_IMAGE_DIR, fname)
        # They didn't give us the labels for the test data, so this is just to make the generator work.
        dst = os.path.join(TEST_IMAGE_NEG_DIR, fname)
        shutil.copyfile(src, dst)
    print('Total training images: {0}'.format(len(os.listdir(
        TRAIN_IMAGE_DIR))))
    print('Total dev images: {0}'.format(len(os.listdir(
        DEV_IMAGE_DIR))))
    print('Total test images: {0}'.format(len(os.listdir(
        TEST_IMAGE_DIR))))


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
    train_values, train_labels = get_values_and_labels(os.path.join(DATASET_BASE_DIR, 'train.jsonl'))
    partition['train'] = train_values
    labels.update(train_labels)
    dev_values, dev_labels = get_values_and_labels(os.path.join(DATASET_BASE_DIR, 'dev.jsonl'))
    partition['dev'] = dev_values
    labels.update(dev_labels)
    test_values, test_labels = get_values_and_labels(os.path.join(DATASET_BASE_DIR, 'test.jsonl'))
    partition['test'] = test_values
    labels.update(test_labels)
    return partition, labels


def train_model(model, partition, labels, augment=False):
    """Trains the model."""
    train_generator = DataGenerator(partition['train'], labels,
                                    MODEL_INPUT_DIM, RAW_IMAGE_DIR, 2)
    dev_generator = DataGenerator(partition['dev'], labels,
                                  MODEL_INPUT_DIM, RAW_IMAGE_DIR, 2)
    start = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(partition['train']) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=dev_generator,
        validation_steps=len(partition['dev']) // BATCH_SIZE)
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    model_fname = os.path.join(MODEL_SAVE_DIR, str(datetime.now()).replace(' ', '_'))
    model.save(model_fname)
    plot_history(history)


def get_model():
    """Returns the model."""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=MODEL_INPUT_DIM))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def main():
    """Runs the program."""
    partition, labels = get_partition_and_labels()
    print('Training examples: {0}'.format(len(partition['train'])))
    print('Validation examples: {0}'.format(len(partition['dev'])))
    print('Test examples: {0}'.format(len(partition['test'])))
    print('Fraction positive examples (train, val only): {0}/{1}'.format(sum(labels.values()), len(labels.keys())))
    build_dataset(partition, labels)
    model = get_model()
    train_model(model, partition, labels)


if __name__ == '__main__':
    main()
