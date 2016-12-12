import theano.tensor as T


def flatten_spatial(x, axis=1):
    other_dims = list(range(axis)) + list(range(axis + 1, x.ndim))
    axis_at_back = other_dims + [axis]
    x_reordered = x.transpose(*axis_at_back)
    return x_reordered.reshape((-1, x.shape[axis])), x_reordered.shape

def unflatten_spatial(x_flat, x_preflatten_shape, axis=1):
    x_reordered = x_flat.reshape(x_preflatten_shape)
    axis_return = list(range(axis)) + [x_reordered.ndim - 1] + list(range(axis, x_reordered.ndim - 1))
    x = x_reordered.transpose(*axis_return)
    return x

def flexible_softmax(x, axis=1):
    x_flat, x_preflatten_shape = flatten_spatial(x, axis=axis)
    x_flat = T.nnet.softmax(x_flat)
    return unflatten_spatial(x_flat, x_preflatten_shape, axis=axis)


import unittest

class TestCase_flex_softmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestCase_flex_softmax, cls).setUpClass()
        import theano

        f_in = T.matrix()
        cls.f_softmax = theano.function([f_in], T.nnet.softmax(f_in))


    def test_flatten_unflatten_2(self):
        import numpy as np
        X = np.random.normal(size=(2, 3)).astype(np.float32)

        self.assertTrue((flatten_spatial(X, axis=1)[0] == X).all())
        self.assertTrue((flatten_spatial(X, axis=0)[0] == X.transpose(1, 0)).all())

        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=0), axis=0)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=1), axis=1)).all())


    def test_flatten_unflatten_3(self):
        import numpy as np
        X = np.random.normal(size=(2, 3, 5)).astype(np.float32)

        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=0), axis=0)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=1), axis=1)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=2), axis=2)).all())


    def test_flatten_unflatten_4(self):
        import numpy as np
        X = np.random.normal(size=(2, 3, 5, 7)).astype(np.float32)

        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=0), axis=0)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=1), axis=1)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=2), axis=2)).all())
        self.assertTrue((X == unflatten_spatial(*flatten_spatial(X, axis=3), axis=3)).all())



    def test_flex_softmax_2(self):
        import numpy as np
        import theano

        f_in = T.matrix()
        f_flex_softmax_0 = theano.function([f_in], flexible_softmax(f_in, axis=0))
        f_flex_softmax_1 = theano.function([f_in], flexible_softmax(f_in, axis=1))

        X = np.random.normal(size=(20, 10)).astype(np.float32)

        a = self.f_softmax(X)
        b = f_flex_softmax_1(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(1, 0)).transpose(1, 0)
        b = f_flex_softmax_0(X)
        self.assertTrue(np.allclose(a, b))


    def test_flex_softmax_3(self):
        import numpy as np
        import theano

        f_in = T.tensor3()
        f_flex_softmax_0 = theano.function([f_in], flexible_softmax(f_in, axis=0))
        f_flex_softmax_1 = theano.function([f_in], flexible_softmax(f_in, axis=1))
        f_flex_softmax_2 = theano.function([f_in], flexible_softmax(f_in, axis=2))

        X = np.random.normal(size=(2, 3, 5)).astype(np.float32)

        a = self.f_softmax(X.reshape((-1, 5))).reshape((2, 3, 5))
        b = f_flex_softmax_2(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(0, 2, 1).reshape((-1, 3))).reshape((2, 5, 3)).transpose(0, 2, 1)
        b = f_flex_softmax_1(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(1, 2, 0).reshape((-1, 2))).reshape((3, 5, 2)).transpose(2, 0, 1)
        b = f_flex_softmax_0(X)
        self.assertTrue(np.allclose(a, b))


    def test_flex_softmax_4(self):
        import numpy as np
        import theano

        f_in = T.tensor4()
        f_flex_softmax_0 = theano.function([f_in], flexible_softmax(f_in, axis=0))
        f_flex_softmax_1 = theano.function([f_in], flexible_softmax(f_in, axis=1))
        f_flex_softmax_2 = theano.function([f_in], flexible_softmax(f_in, axis=2))
        f_flex_softmax_3 = theano.function([f_in], flexible_softmax(f_in, axis=3))

        X = np.random.normal(size=(2, 3, 5, 7)).astype(np.float32)

        a = self.f_softmax(X.reshape((-1, 7))).reshape((2, 3, 5, 7))
        b = f_flex_softmax_3(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(0, 1, 3, 2).reshape((-1, 5))).reshape((2, 3, 7, 5)).transpose(0, 1, 3, 2)
        b = f_flex_softmax_2(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(0, 2, 3, 1).reshape((-1, 3))).reshape((2, 5, 7, 3)).transpose(0, 3, 1, 2)
        b = f_flex_softmax_1(X)
        self.assertTrue(np.allclose(a, b))

        a = self.f_softmax(X.transpose(1, 2, 3, 0).reshape((-1, 2))).reshape((3, 5, 7, 2)).transpose(3, 0, 1, 2)
        b = f_flex_softmax_0(X)
        self.assertTrue(np.allclose(a, b))

