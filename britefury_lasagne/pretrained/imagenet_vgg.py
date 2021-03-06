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

import functools
import numpy as np
import six
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer,\
    Pool2DLayer, Conv2DLayer, DilatedConv2DLayer, PadLayer, NINLayer
from lasagne.nonlinearities import softmax
import lasagne
from britefury_lasagne import config
from . import imagenet
from .. import util



class AbstractVGGModel (imagenet.AbstractImageNetModel):
    def __init__(self, mean_value, class_names, model_name, param_values,
                 model_default_image_size, input_shape=None, last_layer_name=None,
                 **kwargs):
        super(AbstractVGGModel, self).__init__(class_names, model_name, param_values,
                                               model_default_image_size, input_shape=input_shape,
                                               last_layer_name=last_layer_name, **kwargs)
        self.mean_value = mean_value

    def standardise(self, image_tensor):
        return image_tensor - self.mean_value[None,:,None,None]

    def inv_standardise(self, image_tensor):
        return image_tensor + self.mean_value[None,:,None,None]

    @classmethod
    def set_param_values(cls, final_layer, network, param_values):
        params = lasagne.layers.get_all_params(final_layer)
        n_params = len(params)
        if n_params < len(param_values):
            param_values = param_values[:n_params]

        # Get all the layers
        layers = lasagne.layers.get_all_layers(final_layer)
        # Get the dilated convolutional layers
        dilated_conv_layers = [lyr for lyr in layers if isinstance(lyr, DilatedConv2DLayer)]
        # Get the weight parameters of the dilated conv layers
        dilated_conv_Ws = {lyr.W for lyr in dilated_conv_layers}
        # Get the weight parameter of fc6
        fc6 = network.get('fc6')
        fc6_W = fc6.W if fc6 is not None else None

        # Transpose (flip) the first two dimensions of the parameter value weights of
        # dilated conv weights as dilated convolution weight tensors are of the shape
        # (input_chn, output_chn, height, width) rather than
        # (output_chn, input_chn, height, width)
        #
        # If the parameter is the weights of the layer 'fc6' and the number of dimensions don't
        # match, this indicates that the dense layer has been converted to a convolution.
        for i, (param, value) in enumerate(zip(params, param_values)):
            if value.ndim != param.ndim and param is fc6_W:
                value = value.transpose(1, 0).reshape((4096, 512, 7, 7))
                param_values[i] = value
            if param in dilated_conv_Ws:
                value = value.transpose(1, 0, 2, 3)
                param_values[i] = value

        lasagne.layers.set_all_param_values(final_layer, param_values)

    @classmethod
    def conv_2d_layer(cls, cur_layer, name, num_filters, filter_size, dilation=1, pad=1):
        if dilation == 1:
            cur_layer = Conv2DLayer(cur_layer, num_filters=num_filters, filter_size=filter_size,
                                    pad=pad, flip_filters=False, name=name)
        else:
            if pad == 0:
                pass
            elif pad >= 1:
                cur_layer = PadLayer(cur_layer, width=pad * dilation, name='{}_pad'.format(name))
            else:
                raise ValueError('Only padding of 0 or >= 1 supported, not {}'.format(pad))
            cur_layer = DilatedConv2DLayer(cur_layer, num_filters=num_filters, filter_size=filter_size,
                                           flip_filters=False, dilation=dilation, name=name)

        return cur_layer, dilation

    @classmethod
    def nin_layer(cls, cur_layer, name, num_units, dilation=1, nonlinearity=lasagne.nonlinearities.rectify):
        cur_layer = NINLayer(cur_layer, num_units=num_units, nonlinearity=nonlinearity, name=name)

        return cur_layer, dilation

    @classmethod
    def pool_2d_layer(cls, cur_layer, name, pool_size, dilation, pool_layers_to_expand,
                      pool_layers_to_remove):
        new_dilation = dilation

        if name in pool_layers_to_expand or name in pool_layers_to_remove:
            new_dilation *= pool_size
            stride = 1
        else:
            stride = pool_size

        if name not in pool_layers_to_remove:
            pool_size *= dilation

            if stride < pool_size:
                pad0 = (pool_size - stride) // 2
                pad1 = (pool_size - stride) - pad0
                cur_layer = PadLayer(cur_layer, width=[(pad0, pad1), (pad0, pad1)],
                                     name='{}_pad'.format(name))

            cur_layer = Pool2DLayer(cur_layer, pool_size=pool_size, stride=stride, name=name)

        return cur_layer, new_dilation



class VGG16Model (AbstractVGGModel):
    @classmethod
    def build_network_final_layer(cls, input_shape=None, pool_layers_to_expand=None,
                                  pool_layers_to_remove=None,
                                  full_conv=False, pad_fc6=False, last_layer_name=None,
                                  input_layer_constructor=None,
                                  **kwargs):
        if pool_layers_to_expand is None:
            pool_layers_to_expand = set()
        elif isinstance(pool_layers_to_expand, six.string_types):
            pool_layers_to_expand = {pool_layers_to_expand}
        elif isinstance(pool_layers_to_expand, set):
            pass
        else:
            raise ValueError('pool_layers_to_expand should be None, a string or a set, not a {}'.format(
                type(pool_layers_to_expand)))

        if pool_layers_to_remove is None:
            pool_layers_to_remove = set()
        elif isinstance(pool_layers_to_remove, six.string_types):
            pool_layers_to_remove = {pool_layers_to_remove}
        elif isinstance(pool_layers_to_remove, set):
            pass
        else:
            raise ValueError('pool_layers_to_remove should be None, a string or a set, not a {}'.format(
                type(pool_layers_to_remove)))

        if input_shape is None:
            # Default input shape: 3 channel images of size 224 x 224.
            input_shape = (3, 224, 224)

        dilation = 1

        try:
            if input_layer_constructor is None:
                # Input layer: shape is of the form (sample, channel, height, width).
                # We leave the sample dimension with no size (`None`) so that the
                # minibatch size is whatever we need it to be when we use it
                net = InputLayer((None,) + input_shape, name='input')
            else:
                net = input_layer_constructor()
            if last_layer_name == 'input':
                raise StopIteration

            # First two convolutional layers: 64 filters, 3x3 convolution, 1 pixel padding
            net, dilation = cls.conv_2d_layer(net, 'conv1_1', 64, 3, dilation)
            if last_layer_name == 'conv1_1':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv1_2', 64, 3, dilation)
            if last_layer_name == 'conv1_2':
                raise StopIteration
            # 2x2 max-pooling; will reduce size from 224x224 to 112x112
            net, dilation = cls.pool_2d_layer(net, 'pool1', 2, dilation, pool_layers_to_expand,
                                              pool_layers_to_remove)
            if last_layer_name == 'pool1':
                raise StopIteration

            # Two convolutional layers, 128 filters
            net, dilation = cls.conv_2d_layer(net, 'conv2_1', 128, 3, dilation)
            if last_layer_name == 'conv2_1':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv2_2', 128, 3, dilation)
            if last_layer_name == 'conv2_2':
                raise StopIteration
            # 2x2 max-pooling; will reduce size from 112x112 to 56x56
            net, dilation = cls.pool_2d_layer(net, 'pool2', 2, dilation, pool_layers_to_expand,
                                              pool_layers_to_remove)
            if last_layer_name == 'pool2':
                raise StopIteration

            # Three convolutional layers, 256 filters
            net, dilation = cls.conv_2d_layer(net, 'conv3_1', 256, 3, dilation)
            if last_layer_name == 'conv3_1':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv3_2', 256, 3, dilation)
            if last_layer_name == 'conv3_2':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv3_3', 256, 3, dilation)
            if last_layer_name == 'conv3_3':
                raise StopIteration
            # 2x2 max-pooling; will reduce size from 56x56 to 28x28
            net, dilation = cls.pool_2d_layer(net, 'pool3', 2, dilation, pool_layers_to_expand,
                                              pool_layers_to_remove)
            if last_layer_name == 'pool3':
                raise StopIteration

            # Three convolutional layers, 512 filters
            net, dilation = cls.conv_2d_layer(net, 'conv4_1', 512, 3, dilation)
            if last_layer_name == 'conv4_1':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv4_2', 512, 3, dilation)
            if last_layer_name == 'conv4_2':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv4_3', 512, 3, dilation)
            if last_layer_name == 'conv4_3':
                raise StopIteration
            # 2x2 max-pooling; will reduce size from 28x28 to 14x14
            net, dilation = cls.pool_2d_layer(net, 'pool4', 2, dilation, pool_layers_to_expand,
                                              pool_layers_to_remove)
            if last_layer_name == 'pool4':
                raise StopIteration

            # Three convolutional layers, 512 filters
            net, dilation = cls.conv_2d_layer(net, 'conv5_1', 512, 3, dilation)
            if last_layer_name == 'conv5_1':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv5_2', 512, 3, dilation)
            if last_layer_name == 'conv5_2':
                raise StopIteration
            net, dilation = cls.conv_2d_layer(net, 'conv5_3', 512, 3, dilation)
            if last_layer_name == 'conv5_3':
                raise StopIteration
            # 2x2 max-pooling; will reduce size from 14x14 to 7x7
            net, dilation = cls.pool_2d_layer(net, 'pool5', 2, dilation, pool_layers_to_expand,
                                              pool_layers_to_remove)
            if last_layer_name == 'pool5':
                raise StopIteration

            if dilation == 1 and not full_conv:
                # Dense layer, 4096 units
                net = DenseLayer(net, num_units=4096, name='fc6')
                if last_layer_name == 'fc6':
                    raise StopIteration
                # 50% dropout (only applied during training, turned off during prediction)
                net = DropoutLayer(net, p=0.5, name='fc6_dropout')
                if last_layer_name == 'fc6_dropout':
                    raise StopIteration

                # Dense layer, 4096 units
                net = DenseLayer(net, num_units=4096, name='fc7')
                if last_layer_name == 'fc7':
                    raise StopIteration
                # 50% dropout (only applied during training, turned off during prediction)
                net = DropoutLayer(net, p=0.5, name='fc7_dropout')
                if last_layer_name == 'fc7_dropout':
                    raise StopIteration

                # Final dense layer, 1000 units: 1 for each class
                net = DenseLayer(net, num_units=1000, nonlinearity=None, name='fc8')
                if last_layer_name == 'fc8':
                    raise StopIteration
                # Softmax non-linearity that will generate probabilities
                net = NonlinearityLayer(net, softmax, name='prob')
            elif full_conv:
                # Dense layer as 7x7 convolution, 4096 units
                fc6_padding = 3 if pad_fc6 else 0
                net, dilation = cls.conv_2d_layer(net, 'fc6', 4096, 7, dilation, pad=fc6_padding)
                if last_layer_name == 'fc6':
                    raise StopIteration
                # 50% dropout (only applied during training, turned off during prediction)
                net = DropoutLayer(net, p=0.5, name='fc6_dropout')
                if last_layer_name == 'fc6_dropout':
                    raise StopIteration

                # Dense layer, 4096 units
                net, dilation = cls.nin_layer(net, 'fc7', 4096, dilation)
                if last_layer_name == 'fc7':
                    raise StopIteration
                # 50% dropout (only applied during training, turned off during prediction)
                net = DropoutLayer(net, p=0.5, name='fc7_dropout')
                if last_layer_name == 'fc7_dropout':
                    raise StopIteration

                # Final dense layer, 1000 units: 1 for each class
                net, dilation = cls.nin_layer(net, 'fc8', 1000, dilation, nonlinearity=None)
                if last_layer_name == 'fc8':
                    raise StopIteration
                # Softmax non-linearity that will generate probabilities
                net = NonlinearityLayer(net, functools.partial(util.flexible_softmax, axis=1), name='prob')
        except StopIteration:
            pass

        return net

    @classmethod
    def load_params(cls):
        path = cls.get_pretrained_model_params_path('vgg16.pkl')
        config.download(path, 'http://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
        return imagenet.AbstractImageNetModel.unpickle_from_path(path)

    @classmethod
    def load(cls, input_shape=None, last_layer_name=None, **kwargs):
        loaded_params = cls.load_params()
        return cls(loaded_params['mean value'], loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'], model_default_image_size=224,
                   input_shape=input_shape, last_layer_name=last_layer_name, **kwargs)


class VGG19Model (AbstractVGGModel):
    @classmethod
    def build_network_final_layer(cls, input_shape=None):
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
        return cls(loaded_params['mean value'], loaded_params['synset words'], loaded_params['model name'],
                   loaded_params['param values'], model_default_image_size=224,
                   input_shape=input_shape, last_layer_name=last_layer_name)
