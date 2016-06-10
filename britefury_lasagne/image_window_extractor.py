import numpy as np
import skimage.util
import skimage.transform
import tiling_scheme


class ImageWindowExtractor (object):
    def __init__(self, images, image_read_fn, tiling, pad_mode='reflect', downsample=None):
        """

        :param images: a list of images to read; these can be paths, IDs, objects
        :param image_read_fn: an image reader function of the form `fn(image) -> np.array[H,W,C]`
        :param tiling: a `tiling_scheme.TilingScheme` instance that describes how windows are to be extracted
        `from the data
        """
        self.images = images
        self.image_read_fn = image_read_fn
        self.N_images = len(images)
        img0 = self.image_read_fn(images[0])
        self.input_img_shape = img0.shape[:2]

        if tiling.ndim != 2:
            raise ValueError('tiling should have 2 dimensions, not {}'.format(tiling.ndim))

        if isinstance(tiling, tiling_scheme.TilingScheme):
            tiling = tiling.apply(self.input_img_shape)

        if downsample is not None:
            if len(downsample) != 2:
                raise ValueError('dimensionality of downsample ({}) != 2'.format(len(downsample)))
            ds_tiling = tiling.downsample(downsample)
        else:
            ds_tiling = tiling
        self.tiling = ds_tiling

        self.img_shape = ds_tiling.req_data_shape
        self.n_channels = img0.shape[2] if len(img0.shape) > 2 else 1
        self.dtype = img0.dtype

        self.X = np.zeros((self.N_images, self.n_channels) + self.img_shape, dtype=self.dtype)
        for i, img in enumerate(images):
            x = self.image_read_fn(img)
            assert x.shape[:2] == self.input_img_shape

            # Apply padding and cropping
            cropping = tiling.cropping_as_slices
            if cropping is not None:
                x = x[cropping[0], cropping[1]]
            padding = tiling.padding
            if padding is not None:
                if len(x.shape) == 3:
                    padding.append((0, 0))
                x = skimage.util.pad(x, padding, mode=pad_mode)

            if downsample is not None:
                if len(x.shape) > 2:
                    ds = downsample + (1,) * (len(x.shape)-2)
                else:
                    ds = downsample
                x = skimage.transform.downscale_local_mean(x, ds)

            if len(x.shape) == 2:
                x = x[None,:,:]
            else:
                x = np.rollaxis(x, 2, 0)
            self.X[i,:,:,:] = x

        self.img_windows = ds_tiling.tiles

        self.N = self.N_images * self.img_windows[0] * self.img_windows[1]


    def window_indices_to_coords(self, indices):
        block_x = indices % self.img_windows[1]
        block_y = (indices / self.img_windows[1]) % self.img_windows[0]
        img_i = (indices / (self.img_windows[0] * self.img_windows[1]))
        return img_i, block_y, block_x

    def get_windows(self, indices):
        img_i, block_y, block_x = self.window_indices_to_coords(indices)
        return self.get_windows_by_coords(np.concatenate([img_i[:,None], block_y[:,None], block_x[:,None]], axis=1))

    def get_windows_by_coords(self, coords):
        """
        coords - array of shape (N,3) where each row is (image_index, block_y, block_x)
        """
        block_x = coords[:,2] * self.tiling.step_shape[1]
        block_y = coords[:,1] * self.tiling.step_shape[0]
        img_i = coords[:,0]
        windows = np.zeros((coords.shape[0], self.n_channels) + self.tiling.tile_shape, dtype=self.dtype)
        for i in xrange(coords.shape[0]):
            win = self.X[img_i[i], :, block_y[i]:block_y[i]+self.tiling.tile_shape[0],
                  block_x[i]:block_x[i]+self.tiling.tile_shape[1]]
            windows[i,:,:,:] = win
        return windows

    def iterate_minibatches(self, batchsize, shuffle=False):
        indices = np.arange(self.N)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, self.N - batchsize + 1, batchsize):
            yield self.get_windows(indices[start_idx:start_idx + batchsize])


    @staticmethod
    def iterate_minibatches_multi(data, batchsize, shuffle=False):
        d0 = data[0]
        for d1 in data[1:]:
            assert d1.N == data[0].N
        indices = np.arange(data[0].N)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, d0.N, batchsize):
            batch_indices = indices[start_idx:start_idx + batchsize]
            yield [d.get_windows(batch_indices) for d in data]



import unittest

class Test_ImageWindowExtractor (unittest.TestCase):
    def test_simple(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0b = np.rollaxis(img0, 2, 0)
        img1b = np.rollaxis(img1, 2, 0)
        img2b = np.rollaxis(img2, 2, 0)
        img3b = np.rollaxis(img3, 2, 0)

        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                    tiling=tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(1,1)))

        self.assertEqual(wins.tiling.tile_shape, (16, 16))
        self.assertEqual(wins.N_images, 4)
        self.assertEqual(wins.input_img_shape, (100, 100))
        self.assertEqual(wins.img_shape, (100, 100))
        self.assertEqual(wins.n_channels, 3)
        self.assertEqual(wins.X.shape, (4,3,100,100))
        self.assertTrue((wins.X[0,:,:,:] == img0b).all())
        self.assertTrue((wins.X[1,:,:,:] == img1b).all())
        self.assertTrue((wins.X[2,:,:,:] == img2b).all())
        self.assertTrue((wins.X[3,:,:,:] == img3b).all())
        self.assertEqual(wins.img_windows, (85, 85))
        self.assertEqual(wins.N, 85*85*4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23]]))[0,:,:,:] ==
                         img1b[:,34:50,23:39]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23], [2, 9, 61]]))[:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())

        self.assertTrue((wins.get_windows(np.array([1*85*85+34*85+23]))[0,:,:,:] ==
                         img1b[:,34:50,23:39]).all())
        self.assertTrue((wins.get_windows(np.array([1*85*85+34*85+23, 2*85*85+9*85+61]))[:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())


    def test_stepped(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0b = np.rollaxis(img0, 2, 0)
        img1b = np.rollaxis(img1, 2, 0)
        img2b = np.rollaxis(img2, 2, 0)
        img3b = np.rollaxis(img3, 2, 0)

        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                    tiling=tiling_scheme.TilingScheme(tile_shape=(15, 15), step_shape=(2,2)))

        self.assertEqual(wins.tiling.tile_shape, (15, 15))
        self.assertEqual(wins.N_images, 4)
        self.assertEqual(wins.input_img_shape, (100, 100))
        self.assertEqual(wins.img_shape, (99, 99))
        self.assertEqual(wins.n_channels, 3)
        self.assertEqual(wins.X.shape, (4,3,99,99))
        self.assertTrue((wins.X[0,:,:,:] == img0b[:,:99,:99]).all())
        self.assertTrue((wins.X[1,:,:,:] == img1b[:,:99,:99]).all())
        self.assertTrue((wins.X[2,:,:,:] == img2b[:,:99,:99]).all())
        self.assertTrue((wins.X[3,:,:,:] == img3b[:,:99,:99]).all())
        self.assertEqual(wins.img_windows, (43, 43))
        self.assertEqual(wins.N, 43*43*4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23]]))[0,:,:,:] ==
                         img1b[:,68:83,46:61]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23], [2, 9, 21]]))[:,:,:,:] ==
                         np.append(img1b[None,:,68:83,46:61], img2b[None,:,18:33, 42:57], axis=0)).all())

        self.assertTrue((wins.get_windows(np.array([1*43*43+34*43+23]))[0,:,:,:] ==
                         img1b[:,68:83,46:61]).all())
        self.assertTrue((wins.get_windows(np.array([1*43*43+34*43+23, 2*43*43+9*43+21]))[:,:,:,:] ==
                         np.append(img1b[None,:,68:83,46:61], img2b[None,:,18:33, 42:57], axis=0)).all())


    def test_pad_crop(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0b = np.rollaxis(img0, 2, 0)
        img1b = np.rollaxis(img1, 2, 0)
        img2b = np.rollaxis(img2, 2, 0)
        img3b = np.rollaxis(img3, 2, 0)

        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                    tiling=tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(1,1),
                                                                      data_pad_or_crop=[(4,4), (-3,-3)]))

        self.assertEqual(wins.tiling.tile_shape, (16, 16))
        self.assertEqual(wins.N_images, 4)
        self.assertEqual(wins.input_img_shape, (100, 100))
        self.assertEqual(wins.img_shape, (108, 94))
        self.assertEqual(wins.n_channels, 3)
        self.assertEqual(wins.X.shape, (4,3,108,94))
        self.assertTrue((wins.X[0,:,4:-4,:] == img0b[:,:,3:-3]).all())
        self.assertTrue((wins.X[1,:,4:-4,:] == img1b[:,:,3:-3]).all())
        self.assertTrue((wins.X[2,:,4:-4,:] == img2b[:,:,3:-3]).all())
        self.assertTrue((wins.X[3,:,4:-4,:] == img3b[:,:,3:-3]).all())
        self.assertTrue((wins.X[0,:,:4,:] == img0b[:,1:5,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[1,:,:4,:] == img1b[:,1:5,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[2,:,:4,:] == img2b[:,1:5,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[3,:,:4,:] == img3b[:,1:5,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[0,:,-4:,:] == img0b[:,-5:-1,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[1,:,-4:,:] == img1b[:,-5:-1,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[2,:,-4:,:] == img2b[:,-5:-1,3:-3][:,::-1,:]).all())
        self.assertTrue((wins.X[3,:,-4:,:] == img3b[:,-5:-1,3:-3][:,::-1,:]).all())
        self.assertEqual(wins.img_windows, (93, 79))
        self.assertEqual(wins.N, 93*79*4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23]]))[0,:,:,:] ==
                         img1b[:,30:46,26:42]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23], [2, 9, 61]]))[:,:,:,:] ==
                         np.append(img1b[None,:,30:46,26:42], img2b[None,:,5:21, 64:80], axis=0)).all())

        self.assertTrue((wins.get_windows(np.array([1*93*79+34*79+23]))[0,:,:,:] ==
                         img1b[:,30:46,26:42]).all())
        self.assertTrue((wins.get_windows(np.array([1*93*79+34*79+23, 2*93*79+9*79+61]))[:,:,:,:] ==
                         np.append(img1b[None,:,30:46,26:42], img2b[None,:,5:21, 64:80], axis=0)).all())


    def test_stepped_downsamples(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0b = np.rollaxis(skimage.transform.downscale_local_mean(img0, (2,4,1)), 2, 0)
        img1b = np.rollaxis(skimage.transform.downscale_local_mean(img1, (2,4,1)), 2, 0)
        img2b = np.rollaxis(skimage.transform.downscale_local_mean(img2, (2,4,1)), 2, 0)
        img3b = np.rollaxis(skimage.transform.downscale_local_mean(img3, (2,4,1)), 2, 0)

        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                    tiling=tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(8,8)),
                                    downsample=(2,4))

        self.assertEqual(wins.tiling.tile_shape, (8, 4))
        self.assertEqual(wins.N_images, 4)
        self.assertEqual(wins.input_img_shape, (100, 100))
        self.assertEqual(wins.img_shape, (48, 24))
        self.assertEqual(wins.n_channels, 3)
        self.assertEqual(wins.X.shape, (4,3,48,24))
        self.assertTrue((wins.X[0,:,:,:] == img0b[:,:48,:24]).all())
        self.assertTrue((wins.X[1,:,:,:] == img1b[:,:48,:24]).all())
        self.assertTrue((wins.X[2,:,:,:] == img2b[:,:48,:24]).all())
        self.assertTrue((wins.X[3,:,:,:] == img3b[:,:48,:24]).all())
        self.assertEqual(wins.img_windows, (11, 11))
        self.assertEqual(wins.N, 11*11*4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 3, 8]]))[0,:,:,:] ==
                         img1b[:,12:20,16:20]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 3, 8], [2, 9, 6]]))[:,:,:,:] ==
                         np.append(img1b[None,:,12:20,16:20], img2b[None,:,36:44, 12:16], axis=0)).all())

        self.assertTrue((wins.get_windows(np.array([1*11*11+3*11+8]))[0,:,:,:] ==
                         img1b[:,12:20,16:20]).all())
        self.assertTrue((wins.get_windows(np.array([1*11*11+3*11+8, 2*11*11+9*11+6]))[:,:,:,:] ==
                         np.append(img1b[None,:,12:20,16:20], img2b[None,:,36:44, 12:16], axis=0)).all())


