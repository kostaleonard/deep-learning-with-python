import os
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import layers
from keras.preprocessing import image
import Conv1DTranspose
import numpy as np
import time
import random

NUM_NUCLEOTIDES = 4
# See https://www.bioinformatics.org/sms/iupac.html for nucleotide info.
# There are more options than A, T, G, C, but we don't go there.
NUCLEOTIDE_DICT = {
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
}
EPOCHS = 10000
BATCH_SIZE = 32
LATENT_DIM = 16
SEQUENCE_LENGTH = 100
SEQUENCE_LENGTH_DOWNSAMPLE_1 = 50
NUM_FEATURE_MAPS_1 = 128
NUM_FEATURE_MAPS_2 = 256
TRAIN_FILENAME = '../../resources/datasets/covid19-genomes-train.npy'
GENOME_DIR = '../../resources/coronavirus/covid19-genomes'
GENERATED_SAMPLE_SAVE_DIR = '../../resources/coronavirus/generated-genomes'
GAN_SAVE_DIR = '../../resources/coronavirus/saved-models'
GAN_SAVE_FILE = os.path.join(GAN_SAVE_DIR, 'gan.h5')


def get_generator():
    """Returns the generator of the model."""
    generator_input = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(NUM_FEATURE_MAPS_1 * SEQUENCE_LENGTH_DOWNSAMPLE_1)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((SEQUENCE_LENGTH_DOWNSAMPLE_1, NUM_FEATURE_MAPS_1))(x)

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
    return generator


def get_discriminator():
    """Returns the discriminator of the model."""
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
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,
        clipvalue=1.0,
        decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')
    return discriminator


def get_model():
    """Returns the GAN as a 3-tuple: generator, discriminator, GAN. Because the discriminator is already compiled, the
    GAN's discriminator is the only one that is not trainable (looks really confusing)."""
    generator = get_generator()
    discriminator = get_discriminator()
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
    return generator, discriminator, gan


def make_dataset():
    """Creates the dataset and returns it."""
    x_train = np.zeros((0, SEQUENCE_LENGTH, NUM_NUCLEOTIDES), dtype='float32')
    for filename in os.listdir(GENOME_DIR):
        with open(os.path.join(GENOME_DIR, filename), 'r') as genome_file:
            print(genome_file)
            genome = [c for c in genome_file.read().strip() if c.isalnum()]
            for i in range(len(genome) // SEQUENCE_LENGTH):
                seq = genome[i * SEQUENCE_LENGTH: (i + 1) * SEQUENCE_LENGTH]
                if any([nuc not in NUCLEOTIDE_DICT.keys() for nuc in seq]):
                    continue
                sample = np.zeros((1, SEQUENCE_LENGTH, NUM_NUCLEOTIDES))
                for j in range(len(seq)):
                    sample[0, j, NUCLEOTIDE_DICT[seq[j]]] = 1.
                x_train = np.append(x_train, sample, axis=0)
    return x_train


def load_dataset(filename):
    """Returns the dataset from the given filename."""
    return np.load(filename)


def save_dataset(x, filename):
    """Saves the dataset to the given filename."""
    np.save(filename, x)


def train_model(model, x_train):
    """Trains the model on the given training data."""
    # TODO these items should be in a standalone class.
    generator, discriminator, gan = model
    x_train_shuffled = np.copy(x_train)
    random.shuffle(x_train_shuffled)
    start = 0
    for step in range(EPOCHS):
        iteration_start = time.time()
        random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
        generated_samples = generator.predict(random_latent_vectors)
        stop = start + BATCH_SIZE
        real_samples = x_train_shuffled[start: stop]
        combined_samples = np.concatenate([generated_samples, real_samples])
        labels = np.concatenate([np.ones((BATCH_SIZE, 1)),
                                 np.zeros((BATCH_SIZE, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        d_loss = discriminator.train_on_batch(combined_samples, labels)
        random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
        misleading_targets = np.zeros((BATCH_SIZE, 1))
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        iteration_stop = time.time()
        # For testing CPU vs. GPU.
        # print('Iteration time: {0}'.format(iteration_stop - iteration_start))
        start += BATCH_SIZE
        if start > len(x_train_shuffled) - BATCH_SIZE:
            start = 0
            # TODO this didn't appear in the original code? But it seems necessary to shuffle the training data.
            random.shuffle(x_train_shuffled)
        if step % 100 == 0:
            gan.save_weights(GAN_SAVE_FILE)
            print('Discriminator loss: {0}'.format(d_loss))
            print('Adversarial loss: {0}'.format(a_loss))

            genome = generated_samples[0]


            img = image.array_to_img(generated_samples[0] * 255., scale=False)
            img.save(os.path.join(GENERATED_SAMPLE_SAVE_DIR,
                                  'generated_frog_{0}.png'.format(step)))
            img = image.array_to_img(real_samples[0] * 255., scale=False)
            img.save(os.path.join(GENERATED_SAMPLE_SAVE_DIR,
                                  'real_frog_{0}.png'.format(step)))


def main():
    """Runs the program."""
    generator, discriminator, gan = get_model()
    if not os.path.exists(TRAIN_FILENAME):
        x_train = make_dataset()
        save_dataset(x_train, TRAIN_FILENAME)
    else:
        x_train = load_dataset(TRAIN_FILENAME)
    train_model((generator, discriminator, gan), x_train)


if __name__ == '__main__':
    main()
