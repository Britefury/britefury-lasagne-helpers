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
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer, Pool2DLayer, Conv2DLayer
from lasagne.nonlinearities import softmax
from britefury_lasagne import config
from . import imagenet


class VGG16Model (imagenet.AbstractImageNetModel):
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

    @classmethod
    def load_params(cls):
        path = cls.get_pretrained_model_params_path('vgg16.pkl')
        config.download(path, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
        return imagenet.AbstractImageNetModel.unpickle_from_path(path)

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None):
        loaded_params = cls.load_params()
        return cls(loaded_params['mean value'], None, loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'], model_default_image_size=224,
                   input_shape=input_shape, last_layer_name=last_layer_name)


class VGG19Model (imagenet.AbstractImageNetModel):
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

    @classmethod
    def load_params(cls):
        path = cls.get_pretrained_model_params_path('vgg19.pkl')
        config.download(path, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl')
        return imagenet.AbstractImageNetModel.unpickle_from_path(path)

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None):
        loaded_params = cls.load_params()
        return cls(loaded_params['mean value'], None, loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'], model_default_image_size=224,
                   input_shape=input_shape, last_layer_name=last_layer_name)
