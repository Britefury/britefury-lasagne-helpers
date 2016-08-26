# Code taken from:
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
# and
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg19.py
#
# VGG-16 16-layer model and VGG-19 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/
#
# Download pretrained weights from:
# http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
# and
# http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

import sys, os, pickle
import numpy as np
import lasagne
import skimage.transform, skimage.color
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, Pool2DLayer, Conv2DLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
from . import config

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

_PARAMS_DIR_NAME = 'pretrained_models'

PARAMS_DIR = os.path.join(config.get_data_dir_path(), _PARAMS_DIR_NAME)


def download(path, source_url):
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
    if not os.path.exists(path):
        print('Downloading {0} to {1}'.format(source_url, path))
        urlretrieve(source_url, path)
    return path


def get_params_dir():
    return PARAMS_DIR

def set_params_dir(d):
    global PARAMS_DIR
    PARAMS_DIR = d


def _get_vgg16_path():
    return os.path.join(PARAMS_DIR, 'vgg16.pkl')

def _get_vgg19_path():
    return os.path.join(PARAMS_DIR, 'vgg19.pkl')


class VGGModel (object):
    """
    VGG model base class

    Override the `build_network` class method to define the network architecture
    """
    def __init__(self, mean_value, class_names, model_name, param_values, input_shape=None, last_layer_name=None):
        # Build the network
        final_layer = self.build_network(input_shape=input_shape)
        # Generate dictionary mapping layer name to layer
        network = self._final_layer_to_network_dict(final_layer)

        # Slice the network if necessary
        if last_layer_name is not None:
            try:
                final_layer = network[last_layer_name]
            except KeyError:
                raise KeyError('Could not find last layer: no layer named {}'.format(last_layer_name))
            network = self._final_layer_to_network_dict(final_layer)

        # Load in parameter values
        if last_layer_name is None:
            lasagne.layers.set_all_param_values(final_layer, param_values)
        else:
            n_params = len(lasagne.layers.get_all_params(final_layer))
            lasagne.layers.set_all_param_values(final_layer, param_values[:n_params])

        self.final_layer = final_layer
        self.network = network
        self.mean_value = mean_value
        self.class_names = class_names
        self.model_name = model_name


    @classmethod
    def _final_layer_to_network_dict(cls, final_layer):
        layers = lasagne.layers.get_all_layers(final_layer)
        for layer in layers:
            if layer.name is None or layer.name == '':
                raise ValueError('Layer {} has no name'.format(layer))
        return {layer.name: layer for layer in layers}



    @classmethod
    def build_network(cls, input_shape=None):
        raise NotImplementedError('Abstract for type {}'.format(cls))

    
    def prepare_image(self, im, image_size=224):
        """
        Prepare an image for classification with VGG; scale and crop to `image_size` x `image_size`.
        Convert RGB channel order to BGR.
        Subtract mean value.

        :param im: input RGB image as numpy array (height, width, channel)
        :param image_size: output image size, default=224. If `None`, scaling and cropping will not be done.
        :return: (raw_image, vgg_image) where `raw_image` is the scaled and cropped image with dtype=uint8 and
            `vgg_image` is the image with BGR channel order and axes (sample, channel, height, width).
        """
        # If the image is greyscale, convert it to RGB
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)

        if image_size is not None:
            # Scale the image so that its smallest dimension is the desired size
            h, w, _ = im.shape
            if h < w:
                if h != image_size:
                    im = skimage.transform.resize(im, (image_size, w * image_size / h), preserve_range=True)
            else:
                if w != image_size:
                    im = skimage.transform.resize(im, (h * image_size / w, image_size), preserve_range=True)

            # Crop the central `image_size` x `image_size` region of the image
            h, w, _ = im.shape
            im = im[h//2 - image_size // 2:h // 2 + image_size // 2, w // 2 - image_size // 2:w // 2 + image_size // 2]

        # Convert to uint8 type
        rawim = np.copy(im).astype('uint8')

        # Images come in RGB channel order, while VGG net expects BGR:
        im = im[:, :, ::-1]

        # Subtract the mean
        im = im - self.mean_value

        # Shuffle axes from (height, width, channel) to (channel, height, width)
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Add the sample axis to the image; (channel, height, width) -> (sample, channel, height, width)
        im = im[np.newaxis]

        return rawim, floatX(im)


    def inv_prepare_image(self, image):
        """
        Perform the inverse of `prepare_image`; usually used to display an image prepared for classification
        using a VGG net.

        :param im: the image to process
        :return: processed image
        """
        if len(image.shape) == 4:
            # We have a sample dimension; can collapse it if there is only 1 sample
            if image.shape[0] == 1:
                image = image[0]
            else:
                raise ValueError('Sample dimension has > 1 samples ({})'.format(image.shape[0]))

        # Move the channel axis: (C, H, W) -> (H, W, C)
        image = np.rollaxis(image, 0, 3)
        # Add the mean
        image = image + self.mean_value
        # Clip to [0,255] range
        image = image.clip(0.0, 255.0)
        # Convert to uint8 type
        image = image.astype('uint8')
        # Flip channel order BGR to RGB
        image = image[:,:,::-1]
        return image


    @classmethod
    def from_loaded_params(cls, loaded_params, input_shape=None, last_layer_name=None):
        """
        Construct a model given parameters loaded from a pickled VGG model
        :param loaded_params: a dictionary resulting from loading a pickled VGG model
        :return: the model
        """
        return cls(loaded_params['mean value'], loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'], input_shape=input_shape, last_layer_name=last_layer_name)

    @staticmethod
    def unpickle_from_path(path):
        # Oh... the joys of Py2 vs Py3
        with open(path, 'rb') as f:
            if sys.version_info[0] == 2:
                return pickle.load(f)
            else:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                return u.load()


class VGG16Model (VGGModel):
    @classmethod
    def build_network(cls, input_shape=None):
        if input_shape is None:
            # Default input shape: 3 channel images of size 224 x 224.
            input_shape = (3, 224, 224)

        # Input layer: shape is of the form (sample, channel, height, width).
        # We leave the sample dimension with no size (`None`) so that the
        # minibatch size is whatever we need it to be when we use it
        net = InputLayer((None,) + input_shape, name='input')

        # First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
        net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False, name='conv1_1')
        net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False, name='conv1_2')
        # 2x2 max-pooling; will reduce size from 224x224 to 112x112
        net = Pool2DLayer(net, 2, name='pool1')

        # Two convolutional layers, 128 filters
        net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False, name='conv2_1')
        net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False, name='conv2_2')
        # 2x2 max-pooling; will reduce size from 112x112 to 56x56
        net = Pool2DLayer(net, 2, name='pool2')

        # Three convolutional layers, 256 filters
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_1')
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_2')
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_3')
        # 2x2 max-pooling; will reduce size from 56x56 to 28x28
        net = Pool2DLayer(net, 2, name='pool3')

        # Three convolutional layers, 512 filters
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_1')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_2')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_3')
        # 2x2 max-pooling; will reduce size from 28x28 to 14x14
        net = Pool2DLayer(net, 2, name='pool4')

        # Three convolutional layers, 512 filters
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_1')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_2')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_3')
        # 2x2 max-pooling; will reduce size from 14x14 to 7x7
        net = Pool2DLayer(net, 2, name='pool5')

        # Dense layer, 4096 units
        net = DenseLayer(net, num_units=4096, name='fc6')
        # 50% dropout (only applied during training, turned off during prediction)
        net = DropoutLayer(net, p=0.5, name='fc6_dropout')

        # Dense layer, 4096 units
        net = DenseLayer(net, num_units=4096, name='fc7')
        # 50% dropout (only applied during training, turned off during prediction)
        net = DropoutLayer(net, p=0.5, name='fc7_dropout')

        # Final dense layer, 1000 units: 1 for each class
        net = DenseLayer(net, num_units=1000, nonlinearity=None, name='fc8')
        # Softmax non-linearity that will generate probabilities
        net = NonlinearityLayer(net, softmax, name='prob')

        return net

    @staticmethod
    def load_params():
        download(_get_vgg16_path(), 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
        return VGGModel.unpickle_from_path(_get_vgg16_path())

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None):
        return cls.from_loaded_params(cls.load_params(), input_shape=input_shape, last_layer_name=last_layer_name)


class VGG19Model (VGGModel):
    @classmethod
    def build_network(cls, input_shape=None):
        if input_shape is None:
            # Default input shape: 3 channel images of size 224 x 224.
            input_shape = (3, 224, 224)

        # Input layer: shape is of the form (sample, channel, height, width).
        # We leave the sample dimension with no size (`None`) so that the
        # minibatch size is whatever we need it to be when we use it
        net = InputLayer((None,) + input_shape, name='input')

        # First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
        net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False, name='conv1_1')
        net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False, name='conv1_2')
        # 2x2 max-pooling; will reduce size from 224x224 to 112x112
        net = Pool2DLayer(net, 2, name='pool1')

        # Two convolutional layers, 128 filters
        net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False, name='conv2_1')
        net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False, name='conv2_2')
        # 2x2 max-pooling; will reduce size from 112x112 to 56x56
        net = Pool2DLayer(net, 2, name='pool2')

        # Four convolutional layers, 256 filters
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_1')
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_2')
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_3')
        net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False, name='conv3_4')
        # 2x2 max-pooling; will reduce size from 56x56 to 28x28
        net = Pool2DLayer(net, 2, name='pool3')

        # Four convolutional layers, 512 filters
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_1')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_2')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_3')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv4_4')
        # 2x2 max-pooling; will reduce size from 28x28 to 14x14
        net = Pool2DLayer(net, 2, name='pool4')

        # Four convolutional layers, 512 filters
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_1')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_2')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_3')
        net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False, name='conv5_4')
        # 2x2 max-pooling; will reduce size from 14x14 to 7x7
        net = Pool2DLayer(net, 2, name='pool5')

        # Dense layer, 4096 units
        net = DenseLayer(net, num_units=4096, name='fc6')
        # 50% dropout (only applied during training, turned off during prediction)
        net = DropoutLayer(net, p=0.5, name='fc6_dropout')

        # Dense layer, 4096 units
        net = DenseLayer(net, num_units=4096, name='fc7')
        # 50% dropout (only applied during training, turned off during prediction)
        net = DropoutLayer(net, p=0.5, name='fc7_dropout')

        # Final dense layer, 1000 units: 1 for each class
        net = DenseLayer(net, num_units=1000, nonlinearity=None, name='fc8')
        # Softmax non-linearity that will generate probabilities
        net = NonlinearityLayer(net, softmax, name='prob')

        return net

    @staticmethod
    def load_params():
        download(_get_vgg19_path(), 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl')
        return VGGModel.unpickle_from_path(_get_vgg19_path())

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None):
        return cls.from_loaded_params(cls.load_params(), input_shape=input_shape, last_layer_name=last_layer_name)
