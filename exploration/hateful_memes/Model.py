# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras.models import load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import cv2
import json
from exploration.hateful_memes.DataGenerator import DataGenerator
# If you don't have plaidml, get rid of this line and the try-except:
from plaidml.exceptions import Unknown

DATASET_BASE_DIR = '/Users/leo/Documents/Datasets/hateful_memes_data'
RAW_IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'img')
MODEL_SAVE_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/saved_models'
PREDICTION_OUTPUT_DIR = '/Users/leo/PycharmProjects/py_deep_learning/resources/hateful_memes/predictions'


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


def run_model_from_scratch():
    """Builds and runs a model from scratch."""
    '''
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    # Check generators.
    for data_batch, labels_batch in train_generator:
        print('Data batch shape: {0}'.format(data_batch.shape))
        print('Labels batch shape: {0}'.format(labels_batch.shape))
        break
    model = get_model_from_scratch()
    start = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    model_fname = os.path.join(MODEL_SAVE_DIR, 'cats_and_dogs_small_2.h5')
    model.save(model_fname)
    plot_history(history)
    '''


def build_dataset(download_dir=DATASET_BASE_DIR):
    """Builds the dataset from the download directory. Data available on
    Kaggle."""
    original_dataset_dir = os.path.join(download_dir, 'train')
    base_dir = DATASET_BASE_DIR
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print('Total training cat images: {0}'.format(len(os.listdir(
        train_cats_dir))))
    print('Total training dog images: {0}'.format(len(os.listdir(
        train_dogs_dir))))
    print('Total validation cat images: {0}'.format(len(os.listdir(
        validation_cats_dir))))
    print('Total validation dog images: {0}'.format(len(os.listdir(
        validation_dogs_dir))))
    print('Total test cat images: {0}'.format(len(os.listdir(
        test_cats_dir))))
    print('Total test dog images: {0}'.format(len(os.listdir(
        test_dogs_dir))))


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


def main():
    """Runs the program."""
    partition, labels = get_partition_and_labels()
    print('Training examples: {0}'.format(len(partition['train'])))
    print('Validation examples: {0}'.format(len(partition['dev'])))
    print('Test examples: {0}'.format(len(partition['test'])))
    print('Fraction positive examples (train, val only): {0}/{1}'.format(sum(labels.values()), len(labels.keys())))
    #train_generator = DataGenerator(['01235.png'], {'01235.png': 0}, (200, 200, 3), RAW_IMAGE_DIR, 2, batch_size=32)
    # TODO need to memoize the image batches because it is very expensive to constantly load and transform the images.
    # TODO could just use the keras ImageDataGenerator, but need to pay overhead of organizing images.
    train_generator = DataGenerator(partition['train'], labels, (200, 200, 3), RAW_IMAGE_DIR, 2, batch_size=999)
    for data_batch, labels_batch in train_generator:
        print(data_batch.shape)
        print(labels_batch.shape)


if __name__ == '__main__':
    main()
