import sys, time, re, six, itertools, collections
import numpy as np
try:
    from fuel.schemes import ShuffledScheme, SequentialScheme
    from fuel.streams import DataStream
    from fuel.datasets import Dataset
except ImportError:
    ShuffledScheme = SequentialScheme = DataStream = Dataset = NotImplemented


def is_indexable(x):
    """
    Determine if `x` is index-able. `x` is index-able if it provides the `__len__` and `__getitem__` methods. Note
    that `__getitem__` should accept 1D NumPy integer arrays as an index

    :param x: the value to test for being index-able
    :return: True if index-able, False if not
    """
    return hasattr(x, '__len__') and hasattr(x, '__getitem__')


def is_sequence_of_indexables(xs):
    """
    Determine if `x` is a sequence of index-able values. For definition of index-able see `is_indexable`. Tests the
    sequence by checking each value to see if it is index-able.
    Note that the containing sequence should be either a tuple or a list.

    :param xs: the value to test
    :return: True if sequence of index-ables.
    """
    if isinstance(xs, (tuple, list)):
        for x in xs:
            if not is_indexable(x):
                return False
        return True
    return False


def indexables_length(data):
    N = len(data[0])
    # Ensure remainder are consistent
    for i, d1 in enumerate(data[1:]):
        if len(d1) != N:
            raise ValueError('Index-ables have inconsistent length; index-able at 0 has length {}, while index-able at'
                             ' {} has length {}'.format(N, i+1, len(d1)))
    return N

def _batch_iterator_for_sequence_of_indexables(data, batchsize, shuffle_rng=None):
    N = indexables_length(data)
    if shuffle_rng is not None:
        indices = np.arange(N)
        shuffle_rng.shuffle(indices)
        for start_idx in range(0, N, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            yield [d[excerpt] for d in data]
    else:
        for start_idx in range(0, N, batchsize):
            yield [d[start_idx:start_idx+batchsize] for d in data]


class AbstractBatchIterable (object):
    def batch_iterator(self, batchsize, shuffle_rng=None):
        raise NotImplementedError('abstract for type {}'.format(type(self)))


def batch_iterator(dataset, batchsize, shuffle_rng=None):
    """
    Create an iterator that will iterate over the data in `dataset` in mini-batches consisting of `batchsize`
    samples, shuffled using the random number generate `shuffle_rng` if supplied or in-order if not.

    The data in `dataset` must take the form of:

    - a sequence of index-ables (see `is_indexable) (e.g. NumPy arrays) - one for each variable
        (input/target/etc) - where each index-able contains an entry for each sample in the complete dataset.
        The use of index-ables allows the use of NumPy arrays or other objects that support `__len__` and
        `__getitem__`:

    >>> train_X = np.random.normal(size=(5000,3,24,24)) # 5000 samples, 3 channel 24x24 images
    >>> train_y = np.random.randint(0, 5, size=(5000,)) # 5000 samples, classes
    >>> trainer.batch_iterator([train_X, train_y], batchsize=128, shuffle_rng=rng)

    - a Fuel Dataset instance:

    >>> train_dataset = load_fuel_dataset
    >>> trainer.batch_iterator(train_dataset, batchsize=128, shuffle_rng=rng)

    - an object that has the method `dataset.batch_iterator(batchsize, shuffle_rng=None) -> iterator` or a callable of
        the form `dataset(batchsize, shuffle_rng=None) -> iterator` that generates an iterator, where the iterator
        generates mini-batches, where each mini-batch is of the form of a list of numpy arrays:

    >>> def make_iterator(X, y):
    ...     def iter_minibatches(batchsize, shuffle_rng=None):
    ...         indices = np.arange(X.shape[0])
    ...         if shuffle_rng is not None:
    ...             shuffle_rng.shuffle(indices)
    ...         for i in range(0, indices.shape[0], batchsize):
    ...             batch_ndx = indices[i:i+batchsize]
    ...             batch_X = X[batch_ndx]
    ...             batch_y = y[batch_ndx]
    ...             yield [batch_X, batch_y]
    ...     return iter_minibatches
    >>> trainer.batch_iterator(make_iterator(train_X, train_y), batchsize=128, shuffle_rng=rng)

    :param dataset: the data to draw mini-batches from.
    :param batchsize: the mini-batch size
    :param shuffle_rng: [optional] a random number generator used to to shuffle the order of samples before
        building mini-batches
    :return: an iterator
    """
    if is_sequence_of_indexables(dataset):
        # First, try sequence of index-ables; likely the most common dataset type
        # Also, using the index-able interface is preferable to using `batch_iterator` method
        return _batch_iterator_for_sequence_of_indexables(dataset, batchsize, shuffle_rng=shuffle_rng)
    elif Dataset is not NotImplemented and isinstance(dataset, Dataset):
        # Next try Fuel Dataset
        if shuffle_rng is not None:
            train_scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=batchsize, rng=shuffle_rng)
        else:
            train_scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batchsize)
        # Use `DataStream.default_stream`, otherwise the default transformers defined by the dataset *wont*
        # be applied
        stream = DataStream.default_stream(dataset=dataset, iteration_scheme=train_scheme)
        return stream.get_epoch_iterator()
    elif hasattr(dataset, 'batch_iterator'):
        # Next, try `batch_iterator` method
        return dataset.batch_iterator(batchsize, shuffle_rng=shuffle_rng)
    elif callable(dataset):
        # Now try callable; basically the same as `batch_iterator`
        return dataset(batchsize, shuffle_rng=shuffle_rng)
    else:
        # Don't know how to handle this
        raise TypeError('dataset should be a fuel Dataset instance, list of index-ables, should have a `batch_iterator` method or should be or a callable, not a {}'.format(
            type(dataset)
        ))
