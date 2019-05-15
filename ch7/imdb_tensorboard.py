# Leonard R. Kosta Jr.

import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import plot_model

LOG_DIR = '/Users/leo/tmp/log_dir'


def tb_visualization():
    """Runs a 1D CNN on the IMDB dataset and visualizes with
    tensorboard."""
    max_features = 2000
    max_len = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print(x_train.shape)
    print(x_train[0].shape)
    model = keras.models.Sequential()
    model.add(layers.Embedding(max_features, 128,
                               input_length=max_len,
                               name='embed'))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    print(model.summary())
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            embeddings_freq=1,
            embeddings_data=np.arange(0, max_len).reshape((1, max_len))
        )
    ]
    # TODO model graph doesn't work.
    #plot_model(model, to_file=os.path.join(LOG_DIR, 'model.png'))
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        validation_split=0.2,
                        callbacks=callbacks)


def main():
    """Runs the program."""
    tb_visualization()


if __name__ == '__main__':
    main()
