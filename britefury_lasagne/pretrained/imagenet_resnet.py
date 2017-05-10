# Code taken from:
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/resnet50.py
#
# ResNet-50, network from the paper:
# "Deep Residual Learning for Image Recognition"
# http://arxiv.org/pdf/1512.03385v1.pdf
# License (MIT as of 16/10/2016):
# see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE
#
# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl

import os
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from britefury_lasagne import config
from . import imagenet


def build_residual_layer(incoming_layer, names,
                         num_filters, filter_size, stride, pad,
                         use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    if use_bias:
        net = ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None, name=names[0])
    else:
        net = ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None, name=names[0])

    net = BatchNormLayer(net, name=names[1])
    if nonlin is not None:
        net = NonlinearityLayer(net, nonlinearity=nonlin, name=names[2])

    return net


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res{}_branch{}{}', 'res{}_branch{}{}_bn', 'res{}_branch{}{}_relu']

    # right branch
    right_tail = build_residual_layer(
        incoming_layer, map(lambda s: s.format(ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)

    right_tail = build_residual_layer(
        right_tail, map(lambda s: s.format(ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(right_tail)[1], 3, 1, 1)

    right_tail = build_residual_layer(
        right_tail, map(lambda s: s.format(ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(right_tail)[1]*upscale_factor, 1, 1, 0,
        nonlin=None)

    # left branch
    if has_left_branch:
        left_tail = build_residual_layer(
            incoming_layer, map(lambda s: s.format(ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
    else:
        left_tail = incoming_layer

    net = ElemwiseSumLayer([left_tail, right_tail], coeffs=1, name='res{}'.format(ix))
    net = NonlinearityLayer(net, nonlinearity=rectify, name='res{}_relu'.format(ix))

    return net


class ResNet50Model (imagenet.AbstractImageNetModel):
    def __init__(self, mean_image, class_names, model_name, param_values,
                 model_default_image_size, input_shape=None, last_layer_name=None):
        super(ResNet50Model, self).__init__(class_names, model_name, param_values,
                                            model_default_image_size, input_shape=input_shape,
                                            last_layer_name=last_layer_name)
        self.mean_image = mean_image
        self.mean_value = mean_image.mean(axis=(1,2))

    def standardise(self, image_tensor):
        return image_tensor - self.mean_image[None,:,:,:]

    def inv_standardise(self, image_tensor):
        return image_tensor + self.mean_image[None,:,:,:]

    @classmethod
    def build_network_final_layer(cls, input_shape=None, last_layer_name=None, **kwargs):
        if input_shape is None:
            # Default input shape: 3 channel images of size 224 x 224.
            input_shape = (3, 224, 224)

        # Input layer: shape is of the form (sample, channel, height, width).
        # We leave the sample dimension with no size (`None`) so that the
        # minibatch size is whatever we need it to be when we use it
        net = InputLayer((None,) + input_shape, name='input')

        net = build_residual_layer(
            net, ['conv1', 'bn_conv1', 'conv1_relu'],
            64, 7, 3, 2, use_bias=True)
        net = PoolLayer(net, pool_size=3, stride=2, pad=0, mode='max', ignore_border=False, name='pool1')

        block_size = list('abc')
        for c in block_size:
            if c == 'a':
                net = build_residual_block(net, 1, 1, True, 4, ix='2{}'.format(c))
            else:
                net = build_residual_block(net, 1.0 / 4, 1, False, 4, ix='2{}'.format(c))

        block_size = list('abcd')
        for c in block_size:
            if c == 'a':
                net = build_residual_block(net, 1.0 / 2, 1.0 / 2, True, 4, ix='3{}'.format(c))
            else:
                net = build_residual_block(net, 1.0 / 4, 1, False, 4, ix='3{}'.format(c))

        block_size = list('abcdef')
        for c in block_size:
            if c == 'a':
                net = build_residual_block(
                    net, 1.0/2, 1.0/2, True, 4, ix='4{}'.format(c))
            else:
                net = build_residual_block(net, 1.0 / 4, 1, False, 4, ix='4{}'.format(c))

        block_size = list('abc')
        for c in block_size:
            if c == 'a':
                net = build_residual_block(net, 1.0 / 2, 1.0 / 2, True, 4, ix='5{}'.format(c))
            else:
                net = build_residual_block(net, 1.0 / 4, 1, False, 4, ix='5{}'.format(c))

        net = PoolLayer(net, pool_size=7, stride=1, pad=0, mode='average_exc_pad', ignore_border=False, name='pool5')
        net = DenseLayer(net, num_units=1000, nonlinearity=None, name='fc1000')
        net = NonlinearityLayer(net, nonlinearity=softmax, name='prob')

        return net


    @classmethod
    def load_params(cls):
        path = cls.get_pretrained_model_params_path('resnet50.pkl')
        config.download(path, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl')
        return imagenet.AbstractImageNetModel.unpickle_from_path(path)

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None):
        loaded_params = cls.load_params()
        return cls(loaded_params['mean_image'], loaded_params['synset_words'], 'ResNet 50',
                   loaded_params['values'], model_default_image_size=224,
                   input_shape=input_shape, last_layer_name=last_layer_name)
