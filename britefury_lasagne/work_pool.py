import numpy as np
import collections

import joblib


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

    def minibatch_work_stream_iterator(self, batch_extract_fn, N):
        """
        Create a function that generates an iterator that yields mini-batches of data, where the data preparation
        is split among multiple processes. The function returned is suitable for passing to methods of the `Trainer`
        class. In contrast to other mini-batch iteration functions, `batch_extract_fn` takes an array
        of sample indices instead of a batch size and shuffle random number generator. Additionally, the number
        of samples in the data set must be known ahead of time.

        PLEASE NOTE that the function `batch_extract_fn` that is the first argument is passed to external processes,
        so it *must* be defined in the top-level of a module so that the pickling system can locate it.

        Can be used with trainer like so:

        >>> def batch_extractor(indices):
        ...     batch_X = lots_of_work_X(indices)
        ...     batch_y = lots_of_work_y(indices)
        ...     return batch_X, batch_y
        ...
        ... pool = WorkerPool()
        ...
        ... minibatch_iterator_fn = pool.minibatch_work_stream_iterator(batch_extractor, N=10000)
        ...
        ... trainer.train(minibatch_iterator_fn, None, None, batchsize=128)

        :param batch_extract_fn: a function of the form `function(batch_indices) -> (batchX, batchY, ...)`
        where `batch_indices` is a NumPy array containing the indices of samples used to make the mini-batch
        that returns a tuple of NumPy arrays that contain the data for the samples in the mini-batch
        :param N: the number of samples in the dataset
        :return: a function of the form `minibatch_iterator_fn(batchsize, shuffle_rng=None) -> iterator`
        """
        def minibatch_iterator_fn(batchsize, shuffle_rng=None):
            def task_generator():
                indices = np.arange(N)
                if shuffle_rng is not None:
                    shuffle_rng.shuffle(indices)
                for i in range(0, N, batchsize):
                    batch_ndx = indices[i:i+batchsize]
                    yield batch_extract_fn, (batch_ndx,)

            ws = self.work_stream(task_generator())
            return ws.retrieve_iter()

        return minibatch_iterator_fn


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
def _example_batch_extractor(indices):
    return (indices, indices**2)

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
        minibatch_iterator_fn = pool.minibatch_work_stream_iterator(_example_batch_extractor, N=100)

        # Arrays of flags indicating which numbers we got back
        n_flags = np.zeros((100,), dtype=bool)
        n_sqr_flags = np.zeros((10000,), dtype=bool)

        BATCHSIZE = 10

        for batch in minibatch_iterator_fn(batchsize=BATCHSIZE, shuffle_rng=np.random.RandomState(12345)):
            # batch should be a tuple
            self.assertIsInstance(batch, tuple)
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


