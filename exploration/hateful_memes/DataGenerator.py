"""Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

This is a great way to feed data into your model, but you'll need to
modify it to access and output your data correctly for every new
project."""

import os
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras models."""

    def __init__(self, id_list, labels, dim, data_dir, n_classes,
                 batch_size=32, shuffle=True):
        """Initializes the object. id_list is a list of strings where
        each element references a file in the dataset and represents
        one sample. labels is a dict mapping the ID strings in id_list
        to integer labels."""
        self.id_list = id_list
        self.labels = labels
        self.dim = dim
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.id_list))
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        """Returns one batch."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size] \
            if ((index + 1) * self.batch_size) <= len(self.indices) else self.indices[index * self.batch_size:]
        id_list_batch = [self.id_list[k] for k in batch_indices]
        X, y = self.__data_generation(id_list_batch)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indices = np.arange(len(self.id_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, id_list_batch):
        """Return one batch. X is of shape (batch_size, *dim).
        Transforms the given string IDs in id_list_batch into a batch
        of data."""
        X = np.empty((len(id_list_batch), *self.dim))
        y = np.empty(len(id_list_batch), dtype=int)
        for i, id_str in enumerate(id_list_batch):
            img = load_img(os.path.join(self.data_dir, id_str), target_size=(self.dim[0], self.dim[1]))
            X[i, :] = img_to_array(img) / 255.
            y[i] = self.labels[id_str]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
