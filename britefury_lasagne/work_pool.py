import numpy as np
import collections

import joblib

from . import data_source


class WorkerPool (object):
    """
    Create a pool of worker processes.

    Call the `work_stream` method to create a `WorkStream` instance that can be used to perform tasks in
    a separate process.

    The work stream is provided a generator that generates tasks that are to be executed in a pool of processes.
    The work stream will attempt to ensure that a buffer of results from those tasks is kept full; retrieving
    a result will cause the work stream to top up the result buffer as necessary.
    """
    def __init__(self, processes=1):
        """
        Constructor

        :param processes: (default=1) number of processes to start
        """
        self.__pool = joblib.pool.MemmapingPool(processes=processes)

    def _apply_async(self, fn, args):
        return self.__pool.apply_async(fn, args)

    def work_stream(self, task_generator, task_buffer_size=20):
        """
        Create a work stream. Supply a task generator that will be used to generate tasks to execute in the
        worker processes.
        Note that creating multiple work streams from the same `WorkerPool` will result in all of the work streams
        sharing the same pool of processes; long-running tasks from one work stream will delay other work streams
        that use the same pool.

        >>> def task_generator():
        ...     for i in range(10):
        ...         yield function_to_invoke, (arg0, arg1)
        ...
        ... ws = pool.work_stream(task_generator())

        :param task_generator: a generator function if the form `task_generator() -> iterator` that yields tasks in
            the form of tuples that can be passed to `Pool.apply_async` methods; `(fn,)` or `(fn, args)` or
            `(fn, args, kwargs)` where `fn` is a function that is to be executed in a worker process,
        :param task_buffer_size: (default=20) the size of the buffer; the work stream will try to ensure that
        this number of tasks are awaiting completion
        :return: a `WorkStream` instance
        """
        return WorkStream(self, task_generator, task_buffer_size=task_buffer_size)

    def parallel_batch_iterator(self, dataset, batch_buffer_size=20):
        """
        Create a batch iterator that generates an iterator that yields mini-batches of data, where the data
        preparation is split among multiple processes. The batch iterator returned is suitable for passing to methods
        of the `Trainer` class.

        NOTE: `dataset` should be a sequence of index-ables (see `data_source.is_indexable` for definition);
        batch iterators cannot be passed here.
        ALSO NOTE: the objects that are passed in the sequence `dataset` SHOULD BE LIGHTWEIGHT as they must
        be passed to the child processes via serialization/deserialization, so you most probably *don't* want
        to pass large NumPy arrays here.

        PLEASE NOTE that the types of the index-ables in dataset *must* be defined in the top-level of a module so
        that the pickling system can locate them.

        Can be used with trainer like so:

        >>> class ExpensiveSource (object):
        ...     def __len__(self):
        ...         return self.number_of_samples
        ...
        ...     def __getitem__(self, indices):
        ...         batch_X = lots_of_work_X(indices)
        ...         return batch_X
        ...
        ... class ExpensiveTarget (object):
        ...     def __len__(self):
        ...         return self.number_of_samples
        ...
        ...     def __getitem__(self, indices):
        ...         batch_y = lots_of_work_y(indices)
        ...         return batch_y
        ...
        ... pool = WorkerPool()
        ...
        ... batch_iterator = pool.parallel_batch_iterator([ExpensiveSource(), ExpensiveTarget()], N=10000)
        ...
        ... trainer.train(batch_iterator, None, None, batchsize=128)

        :param dataset:  sequence of index-ables (see `data_source.is_indexable) (e.g. NumPy arrays are indexables
        but should not normally be used here for performance and memory usage reasons) - one for each variable
        (input/target/etc) - where each index-able contains an entry for each sample in the complete dataset.
        The use of index-ables allows the use of NumPy arrays or other objects that support `__len__` and
        `__getitem__`
        :param batch_buffer_size: the number of batches that will be buffered up to ensure that data is always
        ready when requested
        :return: a `ParallelBatchIterator` instance that has a `batch_iterator` method and is therefore suitable
        for use with methods of the `Trainer` class.
        """
        if not data_source.is_sequence_of_indexables(dataset):
            raise TypeError('dataset must be a sequence of index-ables (each one should support __len__ and '
                            '__getitem__ where __getitem__ should take a numpy int array as an index')
        return _WorkStreamParallelBatchIterator(dataset, batch_buffer_size, self)


def _extract_batch_by_index_from_sequence_of_iterables(data, batch_indices):
    return [d[batch_indices] for d in data]

class _WorkStreamParallelBatchIterator (data_source.AbstractBatchIterable):
    def __init__(self, dataset, batch_buffer_size, pool):
        self.__dataset = dataset
        self.__num_samples = data_source.indexables_length(dataset)
        self.__batch_buffer_size = batch_buffer_size
        self.__pool = pool


    def batch_iterator(self, batchsize, shuffle_rng=None):
        def task_generator():
            indices = np.arange(self.__num_samples)
            if shuffle_rng is not None:
                shuffle_rng.shuffle(indices)
            for i in range(0, self.__num_samples, batchsize):
                batch_ndx = indices[i:i + batchsize]
                yield _extract_batch_by_index_from_sequence_of_iterables, (self.__dataset, batch_ndx)

        ws = self.__pool.work_stream(task_generator(), task_buffer_size=self.__batch_buffer_size)
        return ws.retrieve_iter()


class WorkStream (object):
    """
    A work stream, normally constructed using the `WorkerPool.work_stream` method.
    """
    def __init__(self, worker_pool, task_generator, task_buffer_size=20, processes=1):
        assert isinstance(worker_pool, WorkerPool)
        self.__task_gen = task_generator
        self.__buffer_size = task_buffer_size
        self.__result_buffer = collections.deque()
        self.__worker_pool = worker_pool
        self.__populate_buffer()

    def __populate_buffer(self):
        while len(self.__result_buffer) < self.__buffer_size:
            if not self.__enqueue():
                break

    def __enqueue(self):
        try:
            task = self.__task_gen.next()
        except StopIteration:
            return False
        else:
            future = self.__worker_pool._apply_async(*task)
            self.__result_buffer.append(future)
            return True

    def retrieve(self):
        """
        Retrieve a result from executing a task. Note that tasks are executed in order and that if the next
        task has not yet completed, this call will block until the result is available.
        :return: the result returned by the task function.
        """
        if len(self.__result_buffer) > 0:
            res = self.__result_buffer.popleft()
            value = res.get()
        else:
            return None

        self.__populate_buffer()

        return value

    def retrieve_iter(self):
        """
        Retrieve a result from executing a task. Note that tasks are executed in order and that if the next
        task has not yet completed, this call will block until the result is available.
        :return: the result returned by the task function.
        """
        while len(self.__result_buffer) > 0:
            res = self.__result_buffer.popleft()
            value = res.get()
            self.__populate_buffer()
            yield value


import unittest, os

# Define example task function in the root of the module so that pickle can find it
def _example_task_fn(*args):
    return sum(args), os.getpid()

# Define example batch extractor function in the root of the module so that pickle can find it
class _AbstractExampleBatchIterator (object):
    def __init__(self, N):
        self.N = N

    # Define __len__ here to save us some work in the base classes that we will actually use
    def __len__(self):
        return self.N

class _ExampleBatchIteratorIdentity (_AbstractExampleBatchIterator):
    def __getitem__(self, indices):
        return indices

class _ExampleBatchIteratorSquare (_AbstractExampleBatchIterator):
    def __getitem__(self, indices):
        return indices**2

class TestCase_WorkStream (unittest.TestCase):
    def get_pool(self):
        try:
            return self.__pool
        except AttributeError:
            self.__pool = WorkerPool(processes=5)
            return self.__pool


    def test_ws(self):
        def task_generator():
            for i in range(5000):
                yield _example_task_fn, (i, i, i)

        pool = self.get_pool()
        stream = pool.work_stream(task_generator())

        pids = set()
        for i in range(5000):
            v, pid = stream.retrieve()
            pids.add(pid)
            self.assertEqual(v, i * 3)

        self.assertEqual(5, len(pids))


    def test_minibatches(self):
        pool = self.get_pool()
        parallel_batches = pool.parallel_batch_iterator([_ExampleBatchIteratorIdentity(100),
                                                         _ExampleBatchIteratorSquare(100)])

        # Arrays of flags indicating which numbers we got back
        n_flags = np.zeros((100,), dtype=bool)
        n_sqr_flags = np.zeros((10000,), dtype=bool)

        BATCHSIZE = 10

        for batch in parallel_batches.batch_iterator(batchsize=BATCHSIZE, shuffle_rng=np.random.RandomState(12345)):
            # batch should be a tuple
            self.assertIsInstance(batch, list)
            # batch should contain two arrays
            self.assertEqual(2, len(batch))
            # each array should be of length BATCHSIZE
            self.assertEqual(BATCHSIZE, batch[0].shape[0])
            self.assertEqual(BATCHSIZE, batch[1].shape[0])

            # Check off the numbers we got back
            n_flags[batch[0]] = True
            n_sqr_flags[batch[1]] = True

        # Check the flags arrays
        self.assertEqual(100, n_flags.sum())
        self.assertEqual(100, n_sqr_flags.sum())

        expected_n_flags = np.ones((100,), dtype=bool)
        expected_n_sqr_flags = np.zeros((10000,), dtype=bool)
        expected_n_sqr_flags[np.arange(100)**2] = True
        self.assertTrue((n_flags == expected_n_flags).all())
        self.assertTrue((n_sqr_flags == expected_n_sqr_flags).all())


