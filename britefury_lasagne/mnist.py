# Code taken from:
# https://raw.githubusercontent.com/Lasagne/Lasagne/master/examples/mnist.py
__author__ = 'Britefury'

import numpy as np
import gzip

from . import dataset


def _download_mnist(filename, source='http://yann.lecun.com/exdb/mnist/'):
    return dataset.download_data(filename, source + filename)

def _load_mnist_images(filename):
    # Download if necessary
    path = _download_mnist(filename)

    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def _load_mnist_labels(filename):
    # Download if necessary
    path = _download_mnist(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


class MNIST (object):
    def __init__(self):
        # We can now download and read the training and test set images and labels.
        train_X = _load_mnist_images('train-images-idx3-ubyte.gz')
        train_y = _load_mnist_labels('train-labels-idx1-ubyte.gz')
        self.test_X = _load_mnist_images('t10k-images-idx3-ubyte.gz')
        self.test_y = _load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        self.train_X, self.val_X = train_X[:-10000], train_X[-10000:]
        self.train_y, self.val_y = train_y[:-10000], train_y[-10000:]
