import sys, os
import numpy as np
import six

from . import config

_DATA_DIR_NAME = 'datasets'

DATA_DIR = os.path.join(config.get_data_dir_path(), _DATA_DIR_NAME)


def download_data(filename, source_url):
    return config.download(os.path.join(DATA_DIR, filename), source_url)


def coerce_rng(rng):
    """
    Coerce a value to a random number generator:

    `None` -> results in `np.random.RandomState(12345)`
    int or long -> results in `np.random.RandomState(rng)`
    a RandomState instance -> left as is

    :param rng: input
    :return: resulting random number generator
    """
    if rng is None:
        return np.random.RandomState(12345)
    elif isinstance(rng, six.integer_types):
        return np.random.RandomState(rng)
    elif isinstance(rng, np.random.RandomState):
        return rng
    else:
        raise TypeError('Cannot coerce {} to np.random.RandomState'.format(type(rng)))


def balanced_subset_indices(y, n_classes, n_samples, shuffle=False, rng=None):
    """
    Generate an array of indices that select a balanced subset of a dataset, such that there are an equal number of
    samples chosen from each class, where the class is identified by the array of integers `y`.
    Note that the array returned will not have the size `n_samples` if `n_samples` is not divisible by `n_classes`.
    Note that the output indices will be in class order, e.g. samples from class 0, then from class 1, etc, so
    shuffling the result is advisable.

    :param y: an array of integers specifying the class of each sample
    :param n_classes: the number of classes
    :param n_samples: the number of samples desired in the subset; either an integer indicating the number of samples
    desired, or a float in the range 0-1 indicating the proportion of the size `y` to use.
    :param shuffle: if True, shuffle the samples within each class before choosing a subset so that the choice is
    random
    :param rng: a random number generator used for shuffling, see `coerce_rng`
    :return: an array of integer indices
    """
    if isinstance(n_samples, six.integer_types):
        n_per_class = n_samples // n_classes
    elif isinstance(n_samples, float):
        if n_samples < 0.0 or n_samples > 1.0:
            raise ValueError('n_samples is a float, indicating that it is a fraction, but it is outside the range '
                             '0-1; it is {}'.format(n_samples))
        n_per_class = int(y.shape[0] * n_samples) // n_classes
        n_per_class = max(n_per_class, 1)
    else:
        raise TypeError('n_samples should be an int/long or a float')

    if shuffle:
        rng = coerce_rng(rng)

    indices = np.arange(y.shape[0])
    selected_indices = []
    for cls_index in range(n_classes):
        indices_in_cls = indices[y==cls_index]
        if shuffle:
            rng.shuffle(indices_in_cls)
        selected_indices.append(indices_in_cls[:n_per_class])
    return np.concatenate(selected_indices, axis=0)
