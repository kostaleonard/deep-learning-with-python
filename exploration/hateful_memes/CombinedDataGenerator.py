"""Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

This is a great way to feed data into your model, but you'll need to
modify it to access and output your data correctly for every new
project."""

import os
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array


class CombinedDataGenerator(keras.utils.Sequence):
    """Generates data for Keras models."""

    def __init__(self, id_list, labels,
                 image_dim, image_dir,
                 text_seq_length, text_dir, tokenizer,
                 n_classes, batch_size=32, shuffle=True):
        """Initializes the object. id_list is a list of strings where
        each element references a file in the dataset and represents
        one sample. labels is a dict mapping the ID strings in id_list
        to integer labels."""
        self.id_list = id_list
        self.labels = labels
        self.image_dim = image_dim
        self.image_dir = image_dir
        self.text_seq_length = text_seq_length
        self.text_dir = text_dir
        self.tokenizer = tokenizer
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
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
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
        X_text = np.zeros((len(id_list_batch), self.text_seq_length),
                          dtype='float32')
        X_img = np.empty((len(id_list_batch), *self.image_dim))
        y = np.zeros(len(id_list_batch), dtype='int32')
        for i, id_str in enumerate(id_list_batch):
            text_fname = os.path.join(self.text_dir, id_str.replace(
                '.png', '.txt'))
            with open(text_fname, 'r') as infile:
                text = infile.read()
            seq = self.tokenizer.texts_to_sequences([text])[0]
            seq = keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=self.text_seq_length, dtype='float32',
                padding='post', truncating='post')[0]
            img = load_img(os.path.join(self.image_dir, id_str),
                           target_size=(self.image_dim[0], self.image_dim[1]))
            X_img[i, :] = img_to_array(img) / 255.
            X_text[i, :] = seq
            y[i] = self.labels[id_str]
        return {'text': X_text, 'img': X_img}, y
