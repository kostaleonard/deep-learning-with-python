# Leonard R. Kosta Jr.

import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

TARGET_IMAGE_PATH = '/Users/leo/tmp/tallon_overworld.jpg'
STYLE_REFERENCE_IMAGE_PATH = '/Users/leo/tmp/starry_night.jpg'
width, height = load_img(TARGET_IMAGE_PATH).size
img_height = 400
img_width = int(width * img_height / height)


def preprocess_image(image_path):
    """Returns a preprocessed image ready to be fed into VGG19."""
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    """Returns an image from the output of VGG19."""
    # These magic numbers reverse VGG19's mean pixel transforms.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    """Returns the content loss between the base and combination. This
    makes sure that the top layer of VGG19 has a similar view of the
    target image and the generated-genomes image."""
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    """Returns a map of the correlations found in the original feature
    matrix."""
    features = K.batch_flatten(K.permute_dimensions(x, [2, 0, 1]))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    """Returns the style loss between the style reference and
    combination."""
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    """Returns the total variation loss, which operates on the pixels of
    the generated-genomes image. Encourages spatial continuity, avoiding overly
    pixelated results."""
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def style_transfer():
    """Runs neural style transfer on the input images. Transfers the
    style of the style reference image to the target image while
    preserving the target content."""
    target_image = K.constant(preprocess_image(
        TARGET_IMAGE_PATH))
    style_reference_image = K.constant(preprocess_image(
        STYLE_REFERENCE_IMAGE_PATH))
    combination_image = K.placeholder((1, img_height, img_width, 3))
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features,
                                          combination_features)
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    grads = K.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    class Evaluator(object):
        """Computes the loss and gradients together, but allows you to
        return them from two separate method calls, which is required by
        the scipy optimizer."""
        def __init__(self):
            """Creates the object."""
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            """Returns the loss value."""
            assert self.loss_value is None
            x = x.reshape((1, img_height, img_width, 3))
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grads_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grads_values = grads_values
            return self.loss_value

        def grads(self, x):
            """Returns the gradients."""
            assert self.loss_value is not None
            grads_values = np.copy(self.grads_values)
            self.loss_value = None
            self.grads_values = None
            return grads_values

    evaluator = Evaluator()
    result_prefix = '/Users/leo/tmp/my_result'
    iterations = 20
    x = preprocess_image(TARGET_IMAGE_PATH)
    x = x.flatten()
    for i in range(iterations):
        print('Start of iteration {0}'.format(i))
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                         x,
                                         fprime=evaluator.grads,
                                         maxfun=20)
        print('Current loss value: {0}'.format(min_val))
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = '{0}_at_iteration_{1}.png'.format(result_prefix, i)
        imsave(fname, img)
        print('Image saved as {0}'.format(fname))
        end_time = time.time()
        print('Iteration {0} completed in {1}s.'.format(
            i, end_time - start_time))


def main():
    """Runs the program."""
    style_transfer()


if __name__ == '__main__':
    main()
