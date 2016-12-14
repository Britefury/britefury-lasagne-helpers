import os, sys, pickle, collections
import numpy as np
import lasagne
from lasagne.utils import floatX
import skimage.transform
from britefury_lasagne import config

_PARAMS_DIR = os.path.join(config.get_data_dir_path(), 'pretrained_models')


class AbstractImageNetModel (object):
    """
    Abstract base class for ImageNet models trained using Caffe

    Override the `build_network` class method to define the network architecture
    """

    def __init__(self, class_names, model_name, param_values,
                 model_default_image_size, input_shape=None, last_layer_name=None,
                 **kwargs):
        # Build the network
        final_layer = self.build_network_final_layer(input_shape=input_shape, **kwargs)
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
        if param_values is not None:
            self.set_param_values(final_layer, network, param_values)

        self.final_layer = final_layer
        self.network = network
        self.class_names = class_names
        self.model_name = model_name
        self.model_default_image_size = model_default_image_size

    @classmethod
    def _final_layer_to_network_dict(cls, final_layer):
        layers = lasagne.layers.get_all_layers(final_layer)
        net_dict = collections.OrderedDict()
        for layer in layers:
            if layer.name is None or layer.name == '':
                raise ValueError('Layer {} has no name'.format(layer))
            net_dict[layer.name] = layer
        return net_dict

    @classmethod
    def build_network_final_layer(cls, input_shape=None, **kwargs):
        raise NotImplementedError('Abstract for type {}'.format(cls))

    @classmethod
    def set_param_values(cls, final_layer, network, param_values):
        n_params = len(lasagne.layers.get_all_params(final_layer))
        if n_params < len(param_values):
            param_values = param_values[:n_params]
        lasagne.layers.set_all_param_values(final_layer, param_values)

    def standardise(self, image_tensor):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def inv_standardise(self, image_tensor):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def prepare_image(self, im, image_size=None):
        """
        Prepare an image for classification with VGG; scale and crop to `image_size` x `image_size`.
        Convert RGB channel order to BGR.
        Subtract mean value.

        :param im: input RGB image as numpy array (height, width, channel)
        :param image_size: [default=None]. Size to scale and crop image to. If `None`, the model's
            default image size will be used. If `image_size <= 0`, scaling and cropping will not be done.
        :return: (raw_image, vgg_image) where `raw_image` is the scaled and cropped image with dtype=uint8 and
            `vgg_image` is the image with BGR channel order and axes (sample, channel, height, width).
        """
        # If the image is greyscale, convert it to RGB
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)

        if image_size is None:
            image_size = self.model_default_image_size

        if image_size is not None and image_size > 0:
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
            im = im[h // 2 - image_size // 2:h // 2 + image_size // 2,
                 w // 2 - image_size // 2:w // 2 + image_size // 2]

        # Convert to uint8 type
        rawim = np.copy(im).astype('uint8')

        # Images come in RGB channel order, while VGG net expects BGR:
        im = im[:, :, ::-1]

        # Shuffle axes from (height, width, channel) to (channel, height, width)
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Add the sample axis to the image; (channel, height, width) -> (sample, channel, height, width)
        im = im[np.newaxis]

        # Standardise
        im = self.standardise(im)

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

        # Inverse standardise
        image = self.inv_standardise(image)

        # Move the channel axis: (C, H, W) -> (H, W, C)
        image = np.rollaxis(image, 0, 3)
        # Clip to [0,255] range
        image = image.clip(0.0, 255.0)
        # Convert to uint8 type
        image = image.astype('uint8')
        # Flip channel order BGR to RGB
        image = image[:, :, ::-1]
        return image

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

    @classmethod
    def get_pretrained_model_params_path(cls, filename):
        return os.path.join(_PARAMS_DIR, filename)
