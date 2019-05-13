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
# If you don't have plaidml, get rid of this line and the try-except:
from plaidml.exceptions import Unknown

DEFAULT_DOWNLOAD_DIR = '/Users/leo/Downloads/dogs-vs-cats'
BASE_DIR = '/Users/leo/Downloads/cats_and_dogs_small'
MODEL_SAVE_DIR = '/Users/leo/tmp/'


def build_dataset(download_dir=DEFAULT_DOWNLOAD_DIR):
    """Builds the dataset from the download directory. Data available on
    Kaggle."""
    original_dataset_dir = os.path.join(download_dir, 'train')
    base_dir = BASE_DIR
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


def get_model_from_scratch():
    """Returns the model, a CNN from scratch."""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
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


def get_model_feature_extraction():
    """Returns the model, a densely connected classifier."""
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def get_model_feature_extraction_data_augmentation(conv_base):
    """Returns the model, VGG16 plus a densely connected classifier. The
    weights of the convolution base are frozen! This must occur before
    model compilation."""
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print('Number of trainable weights before freezing conv_base: {0}'.format(
        len(model.trainable_weights)))
    conv_base.trainable = False
    print('Number of trainable weights after freezing conv_base: {0}'.format(
        len(model.trainable_weights)))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


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


def run_model_from_scratch():
    """Builds and runs a model from scratch."""
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')
    test_dir = os.path.join(BASE_DIR, 'test')
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


def extract_features(directory, sample_count, datagen, batch_size, conv_base):
    """Returns a numpy array of features extracted from the directory."""
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def run_feature_extraction_model():
    """Runs pretrained VGG16 on the input image data in order to extract
    features from the data, then runs dense layers on the extracted
    features. Only the convolution base layers are used during the
    feature extraction phase."""
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')
    test_dir = os.path.join(BASE_DIR, 'test')
    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20
    train_features, train_labels = extract_features(
        train_dir, 2000, datagen, batch_size, conv_base)
    validation_features, validation_labels = extract_features(
        validation_dir, 1000, datagen, batch_size, conv_base)
    test_features, test_labels = extract_features(
        test_dir, 1000, datagen, batch_size, conv_base)
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
    model = get_model_feature_extraction()
    start = time.time()
    history = model.fit(train_features,
                        train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features,
                                         validation_labels))
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    plot_history(history)


def run_model_feature_extraction_data_augmentation():
    """Runs a model that uses the convolution base from VGG16 (weights
    do not change) and dense layers (weights change) on augmented data.
    Also performs fine-tuning of the trained network by retraining the
    last layers of VGG16 with the dense layers of the classifier after
    the initial training has been completed. This is intractible
    without a GPU."""
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')
    test_dir = os.path.join(BASE_DIR, 'test')
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    # Check generators.
    for data_batch, labels_batch in train_generator:
        print('Data batch shape: {0}'.format(data_batch.shape))
        print('Labels batch shape: {0}'.format(labels_batch.shape))
        break
    model = get_model_feature_extraction_data_augmentation(conv_base)
    start = time.time()
    history1 = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    # Fine tuning.
    print('Number of trainable weights before fine tuning: {0}'.format(
        len(model.trainable_weights)))
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    print('Number of trainable weights after fine tuning: {0}'.format(
        len(model.trainable_weights)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])
    history2 = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    stop = time.time()
    print('Training time: {0}'.format(stop - start))
    model_fname = os.path.join(MODEL_SAVE_DIR, 'cats_and_dogs_small_3.h5')
    model.save(model_fname)
    plot_history(history1)
    plot_history(history2, smooth_fac=0.8)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    print('Test accuracy: {0}'.format(test_acc))


def view_intermediate_activations():
    """Shows visualizations of the intermediate activations of layers on
    a test image."""
    model_fname = os.path.join(MODEL_SAVE_DIR, 'cats_and_dogs_small_2.h5')
    model = load_model(model_fname)
    img_path = os.path.join(BASE_DIR, 'test/cats/cat.1700.jpg')
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    print(img_tensor.shape)
    plt.imshow(img_tensor[0])
    plt.show()
    layer_outputs = [layer.output for layer in model.layers[:8]]
    layer_names = [layer.name for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    #plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    #plt.show()
    #plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
    #plt.show()
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, size * images_per_row))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                    row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
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


def generate_pattern(model, layer_name, filter_index, size=150):
    """Returns an image representing the activsation of the nth filter
    of the model."""
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, size, size, 3))])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


def view_conv_filters():
    """Plots the visual pattern that each filter is meant to respond to
    in VGG16."""
    model = VGG16(weights='imagenet', include_top=False)
    layer_name = 'block4_conv1'
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img / 255.
    print(results)
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()


def view_class_activation_map():
    """Plots the heatmap of class activation over input images. A class
    activation heatmap indicates how important each location is with
    respect to the class under consideration."""
    model = VGG16(weights='imagenet')
    img_path = os.path.join(BASE_DIR, 'test_elephant.jpg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted: {0}'.format(decode_predictions(preds, top=3)[0]))
    # 386 is the class number for african elephant.
    african_elephant_output = model.output[:, 386]
    # block5_conv3 is the last convolutional layer in VGG16.
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    try:
        pooled_grads_value, conv_layer_output_value = iterate([x])
    except Unknown:
        print('You have to turn the GPU off for this one to work.')
        exit(1)
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(os.path.join(BASE_DIR, 'test_elephant_heatmap.jpg'), superimposed_img)
    print('Figure saved.')


def main():
    """Runs the program."""
    #build_dataset()
    #run_model_from_scratch()
    #run_feature_extraction_model()
    #run_model_feature_extraction_data_augmentation()
    #view_intermediate_activations()
    #view_conv_filters()
    view_class_activation_map()


if __name__ == '__main__':
    main()
