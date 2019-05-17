# Leonard R. Kosta Jr.

import os
import scipy
import scipy.ndimage
import scipy.misc
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K

# Disable training.
K.set_learning_phase(0)

BASE_DIR = '/Users/leo/tmp'
TEST_IMAGE_1 = '/Users/leo/tmp/tallon_overworld.jpg'
TEST_IMAGE_2 = '/Users/leo/tmp/dogwoods.jpg'
TEST_IMAGE_3 = '/Users/leo/tmp/xmas.jpg'
TEST_IMAGE_4 = '/Users/leo/tmp/graduation.jpg'


def resize_img(img, size):
    """Returns a resized image given the input image and target size."""
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname, use_base_dir=True):
    """Saves the image to the specified file."""
    pil_img = deprocess_img(np.copy(img))
    if use_base_dir:
        fname = os.path.join(BASE_DIR, fname)
    scipy.misc.imsave(fname, pil_img)


def preprocess_image(image_path):
    """Returns a preprocessed image for the neural network."""
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_img(x):
    """Converts the tensor into a valid image."""
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def run_deepdream():
    """Runs the deepdream algorithm."""
    model = inception_v3.InceptionV3(weights='imagenet',
                                     include_top=False)
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5
    }
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    loss = K.variable(0.)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        # Throws out pixels on the border to avoid border artifacts in the loss.
        loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
    dream = model.input
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    def eval_loss_and_grads(x):
        """Returns the value of the loss and gradients given an input
        image."""
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values

    def gradient_ascent(x, iterations, step, max_loss=None):
        """Runs gradient ascent on the input."""
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_loss:
                break
            print('Loss value at iteration {0}: {1}'.format(i, loss_value))
            x += step * grad_values
        return x
    step = 0.01
    num_octave = 3
    octave_scale = 1.4
    iterations = 20
    max_loss = 10.
    base_image_path = TEST_IMAGE_3
    img = preprocess_image(base_image_path)
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple(
            [int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    for shape in successive_shapes:
        print('Processing image shape {0}'.format(shape))
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        #save_img(img, fname='dream_at_scale_{0}.png'.format(str(shape)))
    save_img(img, fname='final_dream.png')


def main():
    """Runs the program."""
    run_deepdream()
    '''
    base_image_path = TEST_IMAGE_2
    img = preprocess_image(base_image_path)
    shape = (300, 400)
    img = resize_img(img, shape)
    save_img(img, fname='dream_at_scale_{0}.png'.format(str(shape)))
    '''


if __name__ == '__main__':
    main()
