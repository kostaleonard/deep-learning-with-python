# Leonard R. Kosta Jr.

import os
# There's an unknown error on this one.
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 16
LATENT_DIM = 2


def sampling(args):
    """Returns a point in the latent space given (z_mean, z_log_var),
    contained in args."""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


def main():
    """Runs the program."""
    input_img = keras.Input(shape=IMG_SHAPE)
    # Encoder.
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    x = layers.Conv2D(
        64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    z_mean = layers.Dense(LATENT_DIM)(x)
    z_log_var = layers.Dense(LATENT_DIM)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    # Decoder.
    decoder_input = layers.Input(K.int_shape(z)[1:])
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(
        32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    decoder = Model(decoder_input, x)
    z_decoded = decoder(z)

    class CustomVariationalLayer(keras.layers.Layer):
        """Custom layer for adding up the VAE losses from the encoder
        and decoder."""

        def vae_loss(self, x, z_decoded):
            """Returns the VAE loss."""
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
            kl_loss = -5e-4 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs, **kwargs):
            """Overrides call method in Layer."""
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            # NOTE: this output is not actually used, but call must return.
            return x

    y = CustomVariationalLayer()([input_img, z_decoded])
    vae = Model(input_img, y)
    # The loss is taken care of by our CustomVariationalLayer.
    # This is also why we don't need to pass labels into the fit function.
    vae.compile(optimizer='rmsprop', loss=None)
    print(vae.summary())
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    vae.fit(x=x_train, y=None, shuffle=True, epochs=10, batch_size=BATCH_SIZE,
            validation_data=(x_test, None))
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([xi, yi])
            z_sample = np.tile(z_sample, BATCH_SIZE).reshape(BATCH_SIZE, 2)
            x_decoded = decoder.predict(z_sample, batch_size=BATCH_SIZE)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    main()
