import collections
import lasagne
try:
    from lasagne.layers.dnn import Conv2DDNNLayer
except ImportError:
    Conv2DDNNLayer = None

_CONV2D_STD_LAYER_TYPES = (lasagne.layers.Conv2DLayer,)
if Conv2DDNNLayer is not None:
    _CONV2D_STD_LAYER_TYPES = _CONV2D_STD_LAYER_TYPES + (Conv2DDNNLayer,)
_CONV2D_LAYER_TYPES = (lasagne.layers.Conv2DLayer, lasagne.layers.DilatedConv2DLayer)
if Conv2DDNNLayer is not None:
    _CONV2D_LAYER_TYPES = _CONV2D_LAYER_TYPES + (Conv2DDNNLayer,)
_DENSE_NIN_LAYER_TYPES = (lasagne.layers.DenseLayer, lasagne.layers.NINLayer)


def convert_network_parameters(dst_network, src_network, src_param_values=None):
    """
    Convert parameters used in a source network for use in a destination network, where the destination
    network may use dilated convolutions in place of standard convolutions or vice versa, or
    convolutional layers in place of dense layers or vice versa.

    :param dst_network: the destination network
    :param src_network: the source network
    :param src_param_values: an optional dictionary mapping parameters in the source network to values that
    should be transformed
    :return: a dictionary mapping destination network parameter to transformed value, suitable to be passed
    as the `givens` parameter of a `theano.function` call.
    """
    if src_param_values is None:
        src_param_values = {}

    dst_layers = lasagne.layers.get_all_layers(dst_network)
    src_layers = lasagne.layers.get_all_layers(src_network)

    dst_param_to_layer = {}
    src_param_to_layer = {}

    for dst_lyr in dst_layers:
        for dst_p in dst_lyr.params.keys():
            dst_param_to_layer[dst_p] = dst_lyr
    for src_lyr in src_layers:
        for src_p in src_lyr.params.keys():
            src_param_to_layer[src_p] = src_lyr

    dst_params = lasagne.layers.get_all_params(dst_network)
    src_params = lasagne.layers.get_all_params(src_network)

    dst_param_table = collections.OrderedDict()

    for dst_p, src_p in zip(dst_params, src_params):
        dst_lyr = dst_param_to_layer[dst_p]
        src_lyr = src_param_to_layer[src_p]
        # Default; copy value over
        dst_param_table[dst_p] = src_p
        if isinstance(dst_lyr, lasagne.layers.DilatedConv2DLayer) and isinstance(src_lyr, _CONV2D_STD_LAYER_TYPES) or \
                        isinstance(dst_lyr, _CONV2D_STD_LAYER_TYPES) and isinstance(src_lyr,
                                                                                    lasagne.layers.DilatedConv2DLayer):
            # Conversion between normal conv layers and dilated conv layers
            W = getattr(dst_lyr, 'W_param', dst_lyr.W)
            if dst_p is W:
                print('Info: {} -> {}: W {} -> W {}'.format(type(src_lyr).__name__, type(dst_lyr).__name__,
                                                            src_lyr.get_W_shape(), dst_lyr.get_W_shape()))
                assert W in dst_params
                src_p = src_param_values.get(src_p, src_p)
                # Dilated conv layers have their first two dimensions the other way round
                src_p = src_p.transpose(1, 0, 2, 3)
                # Handle different settings for flip filters
                if dst_lyr.flip_filters != src_lyr.flip_filters:
                    src_p = src_p.transpose(0, 1, 3, 2)
                print('Info: (conv-deconv) assigning a value of shape {}'.format(src_p.shape.eval()))
                dst_param_table[dst_p] = src_p
        elif isinstance(dst_lyr, _CONV2D_STD_LAYER_TYPES) and isinstance(src_lyr, _CONV2D_STD_LAYER_TYPES):
            # Conv to conv; handle different flip filters settings, otherwise do nothing
            W = getattr(dst_lyr, 'W_param', dst_lyr.W)
            if dst_p is W:
                print('Info: {} -> {}: W {} -> W {}'.format(type(src_lyr).__name__, type(dst_lyr).__name__,
                                                            src_lyr.get_W_shape(), dst_lyr.get_W_shape()))
                assert W in dst_params
                src_p = src_param_values.get(src_p, src_p)
                # Handle different settings for flip filters
                if dst_lyr.flip_filters != src_lyr.flip_filters:
                    dst_param_table[W] = src_p.transpose(0, 1, 3, 2)
        elif isinstance(dst_lyr, _CONV2D_LAYER_TYPES) and isinstance(src_lyr, lasagne.layers.DenseLayer):
            # Dense layer to convolutional layer
            print('Info: {} -> {}'.format(type(src_lyr).__name__, type(dst_lyr).__name__))
            assert dst_p in dst_params
            W = getattr(dst_lyr, 'W_param', dst_lyr.W)
            if dst_p is W:
                src_p = src_param_values.get(src_p, src_p)
                # Flip dense layer matrix from (num_inputs, num_units) -> (num_units, num_inputs)
                src_p = src_p.transpose(1, 0)
                # Reshape
                src_p = src_p.reshape((dst_lyr.num_filters, src_lyr.input_shape[1]) + dst_lyr.filter_size)
                # Transpose kernel H and W if flip_filters is enabled
                if dst_lyr.flip_filters:
                    src_p = src_p.transpose(0, 1, 3, 2)
                # If the convolutional layer is a DilatedConv2DLayer, swap the first two dimensions
                if isinstance(dst_lyr, lasagne.layers.DilatedConv2DLayer):
                    src_p = src_p.transpose(1, 0, 2, 3)
                dst_param_table[W] = src_p
        elif isinstance(dst_lyr, lasagne.layers.DenseLayer) and isinstance(src_lyr, _CONV2D_LAYER_TYPES):
            # Conv layer to dense layer
            print('Info: {} -> {}'.format(type(src_lyr).__name__, type(dst_lyr).__name__))
            W = dst_lyr.W
            if dst_p is W:
                assert W in dst_params
                src_p = src_param_values.get(src_p, src_p)
                # If the convolutional layer is a DilatedConv2DLayer, swap the first two dimensions
                if isinstance(dst_lyr, lasagne.layers.DilatedConv2DLayer):
                    src_p = src_p.transpose(1, 0, 2, 3)
                # Transpose kernel H and W if flip_filters is enabled
                if src_lyr.flip_filters:
                    src_p = src_p.transpose(0, 1, 3, 2)
                # Reshape: collapse filter input, filter H and filter W dimensions
                src_p = src_p.reshape((dst_lyr.num_units, -1))
                # Flip matrix from (num_units, num_inputs) -> (num_inputs, num_units)
                src_p = src_p.transpose(1, 0)
                dst_param_table[W] = src_p
        elif isinstance(dst_lyr, _DENSE_NIN_LAYER_TYPES) and isinstance(src_lyr, _DENSE_NIN_LAYER_TYPES):
            # Nothing to do; directly usable
            print('Info: {} -> {}'.format(type(src_lyr).__name__, type(dst_lyr).__name__))
        elif type(dst_lyr) != type(src_lyr):
            print('WARNING: directly converting parameters from layer {} to {}'.format(
                type(src_lyr), type(dst_lyr)
            ))
    return dst_param_table
