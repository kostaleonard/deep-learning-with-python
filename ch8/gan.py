# Leonard R. Kosta Jr.

import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import layers
from keras.preprocessing import image
import numpy as np
import time

LATENT_DIM = 32
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
IMAGE_SAVE_DIR = '/Users/leo/tmp/gan'
GAN_SAVE_FILE = '/Users/leo/tmp/gan/gan.h5'


def main():
    """Runs the program."""
    # Generator.
    generator_input = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    print(generator.summary())
    # Discriminator.
    discriminator_input = layers.Input(shape=(HEIGHT, WIDTH, CHANNELS))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    discriminator = keras.models.Model(discriminator_input, x)
    print(discriminator.summary())
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,
        clipvalue=1.0,
        decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')
    # GAN.
    # Because discriminator has already been compiled, it is only frozen
    # inside the GAN.
    discriminator.trainable = False
    gan_input = keras.layers.Input(shape=(LATENT_DIM,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    gan_optimizer = keras.optimizers.RMSprop(
        lr=0.0004,
        clipvalue=1.0,
        decay=1e-8)
    gan.compile(optimizer=gan_optimizer,
                loss='binary_crossentropy')
    # Training.
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    # Get frog images.
    x_train = x_train[y_train.flatten() == 6]
    x_train = x_train.reshape(
        (x_train.shape[0],) +
        (HEIGHT, WIDTH, CHANNELS)).astype('float32') / 255.
    iterations = 10000
    batch_size = 20
    start = 0
    for step in range(iterations):
        iteration_start = time.time()
        random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
        generated_images = generator.predict(random_latent_vectors)
        stop = start + batch_size
        real_images = x_train[start:stop]
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        d_loss = discriminator.train_on_batch(combined_images, labels)
        random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
        misleading_targets = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        iteration_stop = time.time()
        # For testing CPU vs. GPU.
        #print('Iteration time: {0}'.format(iteration_stop - iteration_start))
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        if step % 100 == 0:
            gan.save_weights(GAN_SAVE_FILE)
            print('Discriminator loss: {0}'.format(d_loss))
            print('Adversarial loss: {0}'.format(a_loss))
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(IMAGE_SAVE_DIR,
                                  'generated_frog_{0}.png'.format(step)))
            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(IMAGE_SAVE_DIR,
                                  'real_frog_{0}.png'.format(step)))


if __name__ == '__main__':
    main()
