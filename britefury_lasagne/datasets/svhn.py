__author__ = 'Britefury'

import numpy as np
from scipy.io import loadmat
import gzip

from britefury_lasagne import dataset


def _download_svhn(filename, source='http://ufldl.stanford.edu/housenumbers/'):
    return dataset.download_data(filename, source + filename)


def _load_svhn(filename):
    # Download if necessary
    path = _download_svhn(filename)

    # Load in the Matlab file
    data = loadmat(path)

    X = data['X'].astype(np.float32) / np.float32(255.0)
    X = X.transpose(3, 2, 0, 1)
    y = data['y'].astype(np.int32)[:, 0]
    y[y == 10] = 0
    return X, y


class SVHN (object):
    def __init__(self):
        self.train_X, self.train_y = _load_svhn('train_32x32.mat')
        self.test_X, self.test_y = _load_svhn('test_32x32.mat')


class SVHNExtra (object):
    def __init__(self):
        self.extra_X, self.extra_y = _load_svhn('extra_32x32.mat')
