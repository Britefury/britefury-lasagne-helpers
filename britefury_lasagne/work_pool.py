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

        :param task_generator: a generator that yields tasks in the form of tuples that can be passed to
        `Pool.apply_async` methods; `(fn,)` or `(fn, args)` or `(fn, args, kwargs)` where `fn` is a function
        that is to be executed in a worker process,
        :param task_buffer_size: (default=20) the size of the buffer; the work stream will try to ensure that
        this number of tasks are awaiting completion
        :return: a `WorkStream` instance
        """
        return WorkStream(self, task_generator, task_buffer_size=task_buffer_size)


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

def _example_task_fn(*args):
    return sum(args), os.getpid()

class TestCase_WorkStream (unittest.TestCase):
    def test_ws(self):
        def task_generator():
            for i in range(5000):
                yield _example_task_fn, (i, i, i)

        pool = WorkerPool(processes=5)
        stream = pool.work_stream(task_generator())

        pids = set()
        for i in range(5000):
            v, pid = stream.retrieve()
            pids.add(pid)
            self.assertEqual(v, i * 3)

        self.assertEqual(5, len(pids))