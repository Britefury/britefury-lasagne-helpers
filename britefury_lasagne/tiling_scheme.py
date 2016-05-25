
def _add_pad_or_crop(x, y):
    if x is not None and y is not None:
        return x[0] + y[0], x[1] + y[1]
    else:
        return x or y


TILING_MODE_VALID = 'valid'
TILING_MODE_PAD = 'pad'


class TilingScheme (object):
    def __init__(self, tile_shape, step_shape=None, mode=TILING_MODE_VALID, data_pad_or_crop=None):
        if step_shape is not None and len(step_shape) != len(tile_shape):
            raise ValueError('dimensionality of step_shape ({}) != dimensionality of tile_shape ({})'.format(
                len(step_shape), len(tile_shape)
            ))

        if data_pad_or_crop is not None and len(data_pad_or_crop) != len(tile_shape):
            raise ValueError('dimensionality of data_pad_or_crop ({}) != dimensionality of tile_shape ({})'.format(
                len(data_pad_or_crop), len(tile_shape)
            ))

        self.ndim = len(tile_shape)
        self.tile_shape = tile_shape
        self.step_shape = step_shape if step_shape is not None else tile_shape
        self.mode = mode
        self.data_pad_or_crop = data_pad_or_crop if data_pad_or_crop is not None else [None] * self.ndim


    def apply(self, data_shape):
        return DataTilingScheme(data_shape, tile_shape=self.tile_shape, step_shape=self.step_shape,
                                mode=self.mode, data_pad_or_crop=self.data_pad_or_crop)

    def pad_data_and_tiles(self, padding):
        padding_sum_by_dim = [(p[0] + p[1] if p is not None else 0) for p in padding]
        tile_shape = tuple([t + p for t, p in zip(self.tile_shape, padding_sum_by_dim)])
        data_pad_or_crop = [_add_pad_or_crop(p, q) for p, q in zip(self.data_pad_or_crop, padding)]
        return TilingScheme(tile_shape, self.step_shape, data_pad_or_crop=data_pad_or_crop)



class DataTilingScheme (object):
    def __init__(self, data_shape, tile_shape, step_shape=None, mode=TILING_MODE_VALID, data_pad_or_crop=None):
        if len(tile_shape) != len(data_shape):
            raise ValueError('dimensionality of tile_shape ({}) != dimensionality of data_shape ({})'.format(
                len(tile_shape), len(data_shape)
            ))

        if step_shape is not None and len(step_shape) != len(data_shape):
            raise ValueError('dimensionality of step_shape ({}) != dimensionality of data_shape ({})'.format(
                len(step_shape), len(data_shape)
            ))

        if data_pad_or_crop is not None and len(data_pad_or_crop) != len(data_shape):
            raise ValueError('dimensionality of data_pad_or_crop ({}) != dimensionality of data_shape ({})'.format(
                len(data_pad_or_crop), len(data_shape)
            ))

        self.ndim = len(data_shape)
        self.data_shape = data_shape
        self.tile_shape = tile_shape
        self.step_shape = step_shape if step_shape is not None else tile_shape
        self.data_pad_or_crop = data_pad_or_crop if data_pad_or_crop is not None else [None] * self.ndim

        res_tiles = []
        res_req_data_shape = []
        res_pad_or_crop = []

        for dim, dsize, tsize, ssize, pad_or_crop in zip(range(self.ndim), self.data_shape, self.tile_shape,
                                                       self.step_shape, self.data_pad_or_crop):
            # Subtract existing cropping, add existing padding
            if pad_or_crop is not None:
                dsize += (pad_or_crop[0] + pad_or_crop[1])

            # Compute the number of tiles
            n_tiles = (dsize - tsize) / ssize + 1
            if mode == TILING_MODE_PAD:
                # If using PAD mode and there is unused data left over, we need an extra tile
                if ((dsize - tsize) % ssize) > 0:
                    n_tiles += 1
            res_tiles.append(n_tiles)

            # Compute required size
            req_size = tsize + ssize * (n_tiles - 1)
            res_req_data_shape.append(req_size)

            # Compute the additional cropping or padding required
            if req_size != dsize:
                delta_pad_or_crop = (0, req_size - dsize)
                pad_or_crop = _add_pad_or_crop(pad_or_crop, delta_pad_or_crop)

            res_pad_or_crop.append(pad_or_crop)

        self.tiles = tuple(res_tiles)
        self.req_data_shape = tuple(res_req_data_shape)
        self.data_pad_or_crop = res_pad_or_crop

    @property
    def padding(self):
        padding = []
        non_zero = False
        for pc in self.data_pad_or_crop:
            p = 0,0
            if pc is not None:
                p = max(pc[0], 0), max(pc[1], 0)
            padding.append(p)
            non_zero = non_zero or p != (0,0)
        return padding if non_zero else None

    @property
    def cropping_as_slices(self):
        cropping = []
        non_zero = False
        for pc in self.data_pad_or_crop:
            c = slice(None)
            if pc is not None:
                start = -pc[0] if pc[0] < 0 else None
                stop = pc[1] if pc[1] < 0 else None
                c = slice(start, stop)
            cropping.append(c)
            non_zero = non_zero or c != slice(None)
        return cropping if non_zero else None

    def pad_data_and_tiles(self, padding):
        padding_sum_by_dim = [(p[0] + p[1] if p is not None else 0) for p in padding]
        tile_shape = tuple([t + p for t, p in zip(self.tile_shape, padding_sum_by_dim)])
        data_pad_or_crop = [_add_pad_or_crop(p, q) for p, q in zip(self.data_pad_or_crop, padding)]
        return DataTilingScheme(self.data_shape, tile_shape, self.step_shape, data_pad_or_crop=data_pad_or_crop)



import unittest

class TestCase_TilingScheme (unittest.TestCase):
    def test_01_simple_exact_fit(self):
        ts = TilingScheme((20, 20)).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (20,20))
        self.assertEqual(ts.step_shape, (20,20))
        self.assertEqual(ts.tiles, (5,5))
        self.assertEqual(ts.req_data_shape, (100,100))
        self.assertEqual(ts.data_pad_or_crop, [None,None])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, None)

        ts = TilingScheme((20, 20), mode=TILING_MODE_PAD).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (20,20))
        self.assertEqual(ts.step_shape, (20,20))
        self.assertEqual(ts.tiles, (5,5))
        self.assertEqual(ts.req_data_shape, (100,100))
        self.assertEqual(ts.data_pad_or_crop, [None,None])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, None)

    def test_02_simple_inexact_fit(self):
        ts = TilingScheme((15, 15)).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (15,15))
        self.assertEqual(ts.step_shape, (15,15))
        self.assertEqual(ts.tiles, (6,6))
        self.assertEqual(ts.req_data_shape, (90,90))
        self.assertEqual(ts.data_pad_or_crop, [(0,-10),(0,-10)])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, [slice(None,-10), slice(None, -10)])

        ts = TilingScheme((15, 15), mode=TILING_MODE_PAD).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (15,15))
        self.assertEqual(ts.step_shape, (15,15))
        self.assertEqual(ts.tiles, (7,7))
        self.assertEqual(ts.req_data_shape, (105,105))
        self.assertEqual(ts.data_pad_or_crop, [(0,5),(0,5)])
        self.assertEqual(ts.padding, [(0,5), (0,5)])
        self.assertEqual(ts.cropping_as_slices, None)

    def test_03_stepped_exact_fit(self):
        ts = TilingScheme((10, 10), step_shape=(7, 7)).apply((108, 108))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (108,108))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (15,15))
        self.assertEqual(ts.req_data_shape, (108,108))
        self.assertEqual(ts.data_pad_or_crop, [None,None])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, None)

        ts = TilingScheme((10, 10), step_shape=(7, 7), mode=TILING_MODE_PAD).apply((108, 108))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (108,108))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (15,15))
        self.assertEqual(ts.req_data_shape, (108,108))
        self.assertEqual(ts.data_pad_or_crop, [None,None])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, None)

    def test_04_stepped_inexact_fit(self):
        ts = TilingScheme((10, 10), step_shape=(7, 7)).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (13,13))
        self.assertEqual(ts.req_data_shape, (94,94))
        self.assertEqual(ts.data_pad_or_crop, [(0,-6),(0,-6)])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, [slice(None,-6), slice(None, -6)])

        ts = TilingScheme((10, 10), step_shape=(7, 7), mode=TILING_MODE_PAD).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (14,14))
        self.assertEqual(ts.req_data_shape, (101,101))
        self.assertEqual(ts.data_pad_or_crop, [(0,1),(0,1)])
        self.assertEqual(ts.padding, [(0,1),(0,1)])
        self.assertEqual(ts.cropping_as_slices, None)

    def test_05_pad_crop_inexact_fit(self):
        ts = TilingScheme((10, 10), step_shape=(7, 7), data_pad_or_crop=[(-2, -3), (-3, -2)]).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (13,13))
        self.assertEqual(ts.req_data_shape, (94,94))
        self.assertEqual(ts.data_pad_or_crop, [(-2,-4),(-3,-3)])
        self.assertEqual(ts.padding, None)
        self.assertEqual(ts.cropping_as_slices, [slice(2,-4), slice(3, -3)])

        ts = TilingScheme((10, 10), step_shape=(7, 7), mode=TILING_MODE_PAD,
                              data_pad_or_crop=[(-1,-3),(-2,-1)]).apply((100, 100))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (100,100))
        self.assertEqual(ts.tile_shape, (10,10))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (14,14))
        self.assertEqual(ts.req_data_shape, (101,101))
        self.assertEqual(ts.data_pad_or_crop, [(-1,2),(-2,3)])
        self.assertEqual(ts.padding, [(0,2), (0,3)])
        self.assertEqual(ts.cropping_as_slices, [slice(1,None), slice(2, None)])

    def test_06_pad_data_and_tiles(self):
        ts = TilingScheme((10, 10), step_shape=(7, 7)).pad_data_and_tiles([(6,6), (8,8)]).apply((108, 108))
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (108,108))
        self.assertEqual(ts.tile_shape, (22,26))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (15,15))
        self.assertEqual(ts.req_data_shape, (120,124))
        self.assertEqual(ts.data_pad_or_crop, [(6,6), (8,8)])
        self.assertEqual(ts.padding, [(6,6), (8,8)])
        self.assertEqual(ts.cropping_as_slices, None)

        ts2 = TilingScheme((10, 10), step_shape=(7, 7)).apply((108, 108))
        ts = ts2.pad_data_and_tiles([(6,6), (8,8)])
        self.assertEqual(ts.ndim, 2)
        self.assertEqual(ts.data_shape, (108,108))
        self.assertEqual(ts.tile_shape, (22,26))
        self.assertEqual(ts.step_shape, (7,7))
        self.assertEqual(ts.tiles, (15,15))
        self.assertEqual(ts.req_data_shape, (120,124))
        self.assertEqual(ts.data_pad_or_crop, [(6,6), (8,8)])
        self.assertEqual(ts.padding, [(6,6), (8,8)])
        self.assertEqual(ts.cropping_as_slices, None)





