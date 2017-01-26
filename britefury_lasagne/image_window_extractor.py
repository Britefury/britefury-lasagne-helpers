import six
import numpy as np
import skimage.util
import skimage.transform
from . import tiling_scheme



class ImageWindowExtractor (object):
    """
    Extracts windows from a list of images according to a tiling scheme.

    Is index-able and has a `batch_iterator` method so can be passed as a dataset to
    `batch.batch_iterator`, `Trainer.train`, etc.
    """
    def __init__(self, images, image_read_fn, tiling, pad_mode='reflect', downsample=None, postprocess_fn=None):
        """

        :param images: a list of images to read; these can be paths, IDs, objects
        :param image_read_fn: an image reader function of the form `fn(image) -> np.array[H,W,C]`
        :param tiling: a `tiling_scheme.TilingScheme` instance that describes how windows are to be extracted
        `from the data
        :param postprocess_fn: [optional] a post processing function of the form
        `postprocess_fn(extracted_windows) -> transformed_extracted_windows` that applies some sort of transformation
        to the extracted data
        """
        if tiling.ndim != 2:
            raise ValueError('tiling should have 2 dimensions, not {}'.format(tiling.ndim))

        self.images = images
        self.image_read_fn = image_read_fn
        self.postprocess_fn = postprocess_fn
        self.N_images = len(images)
        img0 = self.image_read_fn(images[0])
        self.input_img_shape = img0.shape[:2]

        self.tiling_scheme = tiling

        if isinstance(tiling, tiling_scheme.TilingScheme):
            tiling = tiling.apply(self.input_img_shape)

        self.downsample = downsample
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

        self.shape = (self.N, self.n_channels) + self.tiling.tile_shape


    def assembler(self, image_n_channels=None, n_images=None, upsample_order=0, pad_mode='reflect', img_dtype=None):
        image_n_channels = image_n_channels or self.n_channels
        n_images = n_images or self.N_images
        img_dtype = img_dtype or self.dtype
        return ImageWindowAssembler(image_shape=self.input_img_shape, image_n_channels=image_n_channels,
                                    n_images=n_images, tiling=self.tiling_scheme, upsample=self.downsample,
                                    upsample_order=upsample_order, pad_mode=pad_mode, img_dtype=img_dtype)


    def window_indices_to_coords(self, indices):
        window_x = indices % self.img_windows[1]
        window_y = (indices // self.img_windows[1]) % self.img_windows[0]
        img_i = (indices // (self.img_windows[0] * self.img_windows[1]))
        return img_i, window_y, window_x

    def get_window(self, index):
        img_i, window_y, window_x = self.window_indices_to_coords(index)
        windows = self.get_windows_by_separate_coords(np.array([img_i]), np.array([window_y]), np.array([window_x]))
        return windows[0,...]

    def get_windows(self, indices):
        if not isinstance(indices, np.ndarray):
            raise TypeError('indices must be a NumPy integer array, not a {}'.format(type(indices)))
        img_i, window_y, window_x = self.window_indices_to_coords(indices)
        windows = self.get_windows_by_separate_coords(img_i, window_y, window_x)
        return windows

    def get_windows_by_separate_coords(self, img_i, window_y, window_x):
        """
        img_i - array of shape (N) providing image indices
        block_y - array of shape (N) providing block y-co-ordinate
        block_x - array of shape (N) providing block x-co-ordinate
        """
        block_y = window_y * self.tiling.step_shape[0]
        block_x = window_x * self.tiling.step_shape[1]
        windows = np.zeros((img_i.shape[0], self.n_channels) + self.tiling.tile_shape, dtype=self.dtype)
        for i in xrange(img_i.shape[0]):
            win = self.X[img_i[i], :, block_y[i]:block_y[i]+self.tiling.tile_shape[0],
                  block_x[i]:block_x[i]+self.tiling.tile_shape[1]]
            windows[i,:,:,:] = win
        if self.postprocess_fn is not None:
            windows = self.postprocess_fn(windows)
        return windows

    def get_windows_by_coords(self, coords):
        """
        coords - array of shape (N,3) where each row is (image_index, block_y, block_x)
        """
        window_x = coords[:,2]
        window_y = coords[:,1]
        img_i = coords[:,0]
        return self.get_windows_by_separate_coords(img_i, window_y, window_x)


    def __len__(self):
        return self.N

    def __getitem__(self, i):
        if isinstance(i, (int, long)):
            return self.get_window(i)
        elif isinstance(i, slice):
            indices = np.arange(*i.indices(self.N))
            return self.get_windows(indices)
        elif isinstance(i, tuple) and len(i) == 3 and \
                isinstance(i[0], slice) and isinstance(i[1], slice) and isinstance(i[2], slice):
            img_i = np.arange(*i[0].indices(self.N_images))
            window_y = np.arange(*i[1].indices(self.img_windows[0]))
            window_x = np.arange(*i[2].indices(self.img_windows[1]))
            n_x = len(window_x)
            n_y = len(window_y)
            n_i = len(img_i)
            window_x = np.tile(window_x, (n_x * n_i,))
            window_y = np.tile(np.repeat(window_y, n_x, axis=0), (n_i,))
            img_i = np.repeat(img_i, n_x * n_y, axis=0)
            return self.get_windows_by_separate_coords(img_i, window_y, window_x)
        elif isinstance(i, np.ndarray):
            if len(i.shape) == 1:
                return self.get_windows(i)
            elif len(i.shape) == 2 and i.shape[1] == 3:
                return self.get_windows_by_coords(i)
            else:
                raise TypeError('if i is a NumPy array, its shape must be (N,) or (N,3), not {}'.format(i.shape))
        else:
            raise TypeError('i must be an int/long, a slice, a tuple of 3 slices or a (N,) NumPy integer array, '
                            'or a (N,3) NumPy array, not a {}'.format(type(i)))



    def batch_iterator(self, batchsize, shuffle_rng=None):
        """
        Please note that this method will extract windows from one set of images. This is often not too useful
        as you frequently need more than one e.g. an input set and a target set. For this, see the
        `ImageWindowExtractor.multiplexed_minibatch_iterator` method.

        :param batchsize: the mini-batch size
        :param shuffle_rng: [optional] a random number generator used to shuffle the order in which image windows
        are extracted
        :return: an iterator that yields mini-batch lists of the form `[batch_of_windows]`
        """
        indices = np.arange(self.N)
        if shuffle_rng is not None:
            shuffle_rng.shuffle(indices)
        for start_idx in range(0, self.N, batchsize):
            yield [self.get_windows(indices[start_idx:start_idx + batchsize])]

    def __repr__(self):
        return 'ImageWindowExtractor(n_images={}, downsample={}, N={}, tiling={}, dtype={})'.format(
            self.N_images, self.downsample, self.N, self.tiling, self.dtype
        )


class NonUniformImageWindowExtractor (object):
    def __init__(self, images, image_read_fn, tiling, pad_mode='reflect', downsample=None, postprocess_fn=None):
        """

        :param images: a list of images to read; these can be paths, IDs, objects
        :param image_read_fn: an image reader function of the form `fn(image) -> np.array[H,W,C]`
        :param tiling: a `tiling_scheme.TilingScheme` instance that describes how windows are to be extracted
        `from the data
        :param postprocess_fn: [optional] a post processing function of the form
        `postprocess_fn(extracted_windows) -> transformed_extracted_windows` that applies some sort of transformation
        to the extracted data
        """
        if tiling.ndim != 2:
            raise ValueError('tiling should have 2 dimensions, not {}'.format(tiling.ndim))

        self.tiling_scheme = tiling
        self.N_images = len(images)

        image_data = [image_read_fn(img) for img in images]

        self.extractors = []
        self.extractor_image_offsets = [0]

        images_by_shape = []
        shape = image_data[0].shape[:2]

        for img in image_data:
            img_shape = img.shape[:2]
            if img_shape != shape:
                self.__new_extractor(images_by_shape, shape, tiling, pad_mode, downsample, postprocess_fn)
                images_by_shape = []
                shape = img_shape
            images_by_shape.append(img)
        if len(images_by_shape) > 0:
            self.__new_extractor(images_by_shape, shape, tiling, pad_mode, downsample, postprocess_fn)
            images_by_shape = []

        self.extractor_offsets = np.append(np.array([0]),
                                           np.cumsum([len(ext) for ext in self.extractors]))
        self.extractor_image_offsets = np.array(self.extractor_image_offsets)

        self.N = self.extractor_offsets[-1]


    def __new_extractor(self, images, shape, tiling, pad_mode, downsample, postprocess_fn):
        extractor = ImageWindowExtractor(images, lambda x: x, tiling, pad_mode=pad_mode, downsample=downsample,
                                         postprocess_fn=postprocess_fn)
        self.extractors.append(extractor)
        pos = self.extractor_image_offsets[-1]
        self.extractor_image_offsets.append(pos + len(images))

    def get_window(self, index):
        return self.get_windows(np.array([index]))[0,...]

    def get_windows(self, indices):
        if not isinstance(indices, np.ndarray):
            raise TypeError('indices must be a NumPy integer array, not a {}'.format(type(indices)))

        # Get the indices of the extractors that cover these samples
        extractor_indices = np.searchsorted(self.extractor_offsets, indices, side='right') - 1
        # Get the offsets indices within the relevant extractors
        sample_offsets = indices - self.extractor_offsets[extractor_indices]
        # Use RLE to get runs of extractor indices
        ext_runs = rle(extractor_indices)

        # For each run:
        windows = []
        for run_i in six.moves.range(ext_runs.shape[0]):
            start = ext_runs[run_i, 0]
            end = ext_runs[run_i, 1]
            extractor_i = ext_runs[run_i, 2]
            # Get the extractor
            extractor = self.extractors[extractor_i]
            # Extract a run of windows
            wins = extractor.get_windows(sample_offsets[start:end])
            windows.append(wins)

        return np.concatenate(windows, axis=0)

    def get_windows_by_separate_coords(self, img_i, window_y, window_x):
        """
        img_i - array of shape (N) providing image indices
        block_y - array of shape (N) providing block y-co-ordinate
        block_x - array of shape (N) providing block x-co-ordinate
        """
        # Get the indices of the extractors that cover these samples
        extractor_indices = np.searchsorted(self.extractor_image_offsets, img_i, side='right') - 1
        # Get the offsets indices within the relevant extractors
        image_offsets = img_i - self.extractor_image_offsets[extractor_indices]
        # Use RLE to get runs of extractor indices
        ext_runs = rle(extractor_indices)

        # For each run:
        windows = []
        for run_i in six.moves.range(ext_runs.shape[0]):
            start = ext_runs[run_i, 0]
            end = ext_runs[run_i, 1]
            extractor_i = ext_runs[run_i, 2]
            # Get the extractor
            extractor = self.extractors[extractor_i]
            # Extract a run of windows
            wins = extractor.get_windows_by_separate_coords(image_offsets[start:end],
                                                            window_y[start:end], window_x[start:end])
            windows.append(wins)

        return np.concatenate(windows, axis=0)

    def get_windows_by_coords(self, coords):
        """
        coords - array of shape (N,3) where each row is (image_index, block_y, block_x)
        """
        window_x = coords[:,2]
        window_y = coords[:,1]
        img_i = coords[:,0]
        return self.get_windows_by_separate_coords(img_i, window_y, window_x)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if isinstance(index, (int, long)):
            return self.get_window(index)
        elif isinstance(index, slice):
            indices = np.arange(*index.indices(self.N))
            return self.get_windows(indices)
        elif isinstance(index, np.ndarray):
            return self.get_windows(index)
        else:
            raise TypeError('index must be an int, a slice or a numpy array')

    def batch_iterator(self, batchsize, shuffle_rng=None):
        """
        `batch_iterator` method for support batch iterator protocol. Returns an iterator that
        generates mini-batches extracted from `self`

        :param batchsize: the mini-batch size
        :param shuffle_rng: [optional] a random number generator used to shuffle the order in which image windows
        are extracted
        :return: an iterator that yields mini-batch lists of the form `[batch_of_windows]`
        """
        indices = np.arange(self.N)
        if shuffle_rng is not None:
            shuffle_rng.shuffle(indices)
        for start_idx in range(0, self.N, batchsize):
            yield [self[indices[start_idx:start_idx + batchsize]]]


class ImageWindowAssembler (object):
    def __init__(self, image_shape, image_n_channels, n_images, tiling, upsample=None, upsample_order=0,
                 pad_mode='reflect', img_dtype=np.float32):
        """
        :param image_shape: the shape of the images that are to be generated
        :param image_n_channels: the number of channels
        :param n_images: the number of images to be generated
        :param tiling: the tiling scheme used to generate the tiles
        :param upsample: [default=`None`] the upsampling factor
        :param upsample_order: [default=`0`] the interpolation order used for upsampling
        :param pad_mode: [default=`'reflect'`] the padding mode used to invert the effect of cropping
        :param img_dtype: [default=`np.float32`] the data type used for storing the images
        """
        self.N_images = n_images
        self.output_img_shape = image_shape

        if tiling.ndim != 2:
            raise ValueError('tiling should have 2 dimensions, not {}'.format(tiling.ndim))

        if isinstance(tiling, tiling_scheme.TilingScheme):
            tiling = tiling.apply(self.output_img_shape)

        if upsample is not None:
            if len(upsample) != 2:
                raise ValueError('dimensionality of downsample ({}) != 2'.format(len(upsample)))
            ds_tiling = tiling.downsample(upsample)
        else:
            ds_tiling = tiling
        self.upsampled_tiling = tiling
        self.tiling = ds_tiling
        self.upsample = upsample
        self.upsample_order = upsample_order
        self.pad_mode = pad_mode

        self.img_shape = ds_tiling.req_data_shape
        self.n_channels = image_n_channels
        self.dtype = img_dtype

        self.X = np.zeros((self.N_images, self.n_channels) + self.img_shape, dtype=self.dtype)

        self.img_windows = ds_tiling.tiles

        self.N = self.N_images * self.img_windows[0] * self.img_windows[1]

    def window_indices_to_coords(self, indices):
        block_x = indices % self.img_windows[1]
        block_y = (indices // self.img_windows[1]) % self.img_windows[0]
        img_i = (indices // (self.img_windows[0] * self.img_windows[1]))
        return img_i, block_y, block_x

    def set_windows(self, indices, X):
        img_i, block_y, block_x = self.window_indices_to_coords(indices)
        self.set_windows_by_coords(np.concatenate([img_i[:, None], block_y[:, None], block_x[:, None]], axis=1), X)

    def set_windows_by_coords(self, coords, X):
        """
        coords - array of shape (N,3) where each row is (image_index, block_y, block_x)
        """
        block_x = coords[:,2] * self.tiling.step_shape[1]
        block_y = coords[:,1] * self.tiling.step_shape[0]
        img_i = coords[:,0]
        self.set_windows_by_separate_coords(img_i, block_y, block_x, X)

    def set_windows_by_separate_coords(self, img_i, block_y, block_x, X):
        for i in xrange(img_i.shape[0]):
            self.X[img_i[i], :,
                   block_y[i]:block_y[i]+self.tiling.tile_shape[0],
                   block_x[i]:block_x[i]+self.tiling.tile_shape[1]] = X[i,:,:,:]

    def get_image(self, i):
        x = self.X[i,:,:,:]
        # Move channel axis to the back
        x = x.transpose(1,2,0)

        if self.upsample is not None:
            x = skimage.transform.rescale(x, tuple([float(x) for x in self.upsample]), order=self.upsample_order)

        # Apply padding and cropping
        cropping = self.upsampled_tiling.cropping
        if cropping is not None:
            cropping.append((0, 0))
            x = skimage.util.pad(x, cropping, mode=self.pad_mode)
        padding = self.upsampled_tiling.inv_padding_as_slices
        if padding is not None:
            x = x[padding[0], padding[1], :]

        return x

    def __repr__(self):
        return 'ImageWindowAssembler(n_images={}, output_img_shape={}, upsample={}, N={}, tiling={}, n_channels={}, dtype={})'.format(
            self.N_images, self.output_img_shape, self.upsample, self.N, self.tiling, self.n_channels, self.dtype
        )



def rle(x):
    """
    Run length encoding of an array `x`
    :param x: the array to encode
    :return: a `(N,3)` array `y` where `y[:,0]` are the start indices of the runs, `y[:,1]` are the end indices
        and `y[:,2]` are the values from `x` at the start indices
    """
    if len(x.shape) != 1:
        raise ValueError('x should be 1-dimensional, not {}'.format(len(x.shape)))
    # Get the indices of value changes
    pos, = np.where(np.diff(x) != 0)
    # Prepend 0, append length
    pos = np.concatenate(([0], pos + 1, [len(x)]))
    # Get the start, end, length and values
    start = pos[:-1]
    end = pos[1:]
    values = x[pos[:-1]]
    # Join
    return np.concatenate([start[:, None], end[:, None], values[:, None]], axis=1)


import unittest

class Test_rle (unittest.TestCase):
    def test_rle(self):
        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
        y = rle(x)
        expected = np.array([
            [0, 10, 1],
            [10, 12, 0],
            [12, 14, 2],
            [14, 24, 9],
        ])
        self.assertTrue((y == expected).all())


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
        self.assertEqual(len(wins), 85*85*4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23]]))[0,:,:,:] ==
                         img1b[:,34:50,23:39]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23], [2, 9, 61]]))[:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())

        a_i = 1*85*85+34*85+23
        b_i = 2*85*85+9*85+61
        # Single index; get_window() method
        self.assertTrue((wins.get_window(a_i) ==
                         img1b[:,34:50,23:39]).all())
        # Single index; index operator
        self.assertTrue((wins[a_i] ==
                         img1b[:,34:50,23:39]).all())
        # Slice; index operator
        self.assertTrue((wins[a_i:a_i+2] ==
                         np.append(img1b[None,:,34:50,23:39], img1b[None,:,34:50,24:40], axis=0)).all())
        # Single index in an array; get_windows() method
        self.assertTrue((wins.get_windows(np.array([a_i]))[0,:,:,:] ==
                         img1b[:,34:50,23:39]).all())
        # Multiple indices in an array; get_windows() method
        self.assertTrue((wins.get_windows(np.array([a_i, b_i]))[:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())
        # Multiple indices in an array; index operator
        self.assertTrue((wins[np.array([a_i, b_i])][:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())
        # Multiple indices in a (N,3) array; index operator
        self.assertTrue((wins[np.array([[1, 34, 23], [2, 9, 61]])][:,:,:,:] ==
                         np.append(img1b[None,:,34:50,23:39], img2b[None,:,9:25, 61:77], axis=0)).all())

        # Tuple of slices; index operator
        tos_wins = wins[1:3, 5:20, 15:40]
        tos_imgs = [img1b, img2b]
        for i in range(2):
            for y in range(15):
                for x in range(25):
                    self.assertTrue(
                        (tos_wins[i * 15 * 25 + y * 25 + x, :, :, :] ==
                         tos_imgs[i][:, 5 + y:21 + y, 15 + x:31 + x]).all())


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


    def test_pad_crop_and_reassemble(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0b = np.rollaxis(img0, 2, 0)
        img1b = np.rollaxis(img1, 2, 0)
        img2b = np.rollaxis(img2, 2, 0)
        img3b = np.rollaxis(img3, 2, 0)

        tiling = tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(1,1), data_pad_or_crop=[(4,4), (-3,-3)])
        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x, tiling=tiling)

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

        # Reassemble
        assembler = ImageWindowAssembler(image_shape=(100,100), image_n_channels=3, n_images=4, tiling=tiling,
                                         img_dtype=img0.dtype)
        all_win_indices = np.arange(wins.N)
        assembler.set_windows(all_win_indices, wins.get_windows(all_win_indices))

        self.assertTrue((assembler.get_image(0)[:,3:-3,:] == img0[:,3:-3,:]).all())
        self.assertTrue((assembler.get_image(1)[:,3:-3,:] == img1[:,3:-3,:]).all())
        self.assertTrue((assembler.get_image(2)[:,3:-3,:] == img2[:,3:-3,:]).all())
        self.assertTrue((assembler.get_image(3)[:,3:-3,:] == img3[:,3:-3,:]).all())


    def test_stepped_downsampled(self):
        img0 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img1 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img2 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img3 = np.random.uniform(0.0, 1.0, size=(100,100,3))
        img0_ds = skimage.transform.downscale_local_mean(img0, (2,4,1))
        img1_ds = skimage.transform.downscale_local_mean(img1, (2,4,1))
        img2_ds = skimage.transform.downscale_local_mean(img2, (2,4,1))
        img3_ds = skimage.transform.downscale_local_mean(img3, (2,4,1))
        img0_ds_us = skimage.transform.rescale(img0_ds, (2.0, 4.0), order=0)
        img1_ds_us = skimage.transform.rescale(img1_ds, (2.0, 4.0), order=0)
        img2_ds_us = skimage.transform.rescale(img2_ds, (2.0, 4.0), order=0)
        img3_ds_us = skimage.transform.rescale(img3_ds, (2.0, 4.0), order=0)
        img0b = np.rollaxis(img0_ds, 2, 0)
        img1b = np.rollaxis(img1_ds, 2, 0)
        img2b = np.rollaxis(img2_ds, 2, 0)
        img3b = np.rollaxis(img3_ds, 2, 0)

        tiling = tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(8,8))
        wins = ImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                    tiling=tiling, downsample=(2,4))

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


        # Reassemble
        assembler = ImageWindowAssembler(image_shape=(100,100), image_n_channels=3, n_images=4, tiling=tiling,
                                         upsample=(2,4), img_dtype=img0.dtype)
        all_win_indices = np.arange(wins.N)
        assembler.set_windows(all_win_indices, wins.get_windows(all_win_indices))

        self.assertTrue(np.isclose(assembler.get_image(0)[8:-8,8:-8,:], img0_ds_us[8:-8,8:-8,:]).all())
        self.assertTrue(np.isclose(assembler.get_image(1)[8:-8,8:-8,:], img1_ds_us[8:-8,8:-8,:]).all())
        self.assertTrue(np.isclose(assembler.get_image(2)[8:-8,8:-8,:], img2_ds_us[8:-8,8:-8,:]).all())
        self.assertTrue(np.isclose(assembler.get_image(3)[8:-8,8:-8,:], img3_ds_us[8:-8,8:-8,:]).all())


class Test_NonUniformImageWindowExtractor (unittest.TestCase):
    def test_simple(self):
        # Images of non-uniform size
        img0 = np.random.uniform(0.0, 1.0, size=(100, 100, 3))
        img1 = np.random.uniform(0.0, 1.0, size=(90, 110, 3))
        img2 = np.random.uniform(0.0, 1.0, size=(95, 105, 3))
        img3 = np.random.uniform(0.0, 1.0, size=(105, 95, 3))
        img0b = np.rollaxis(img0, 2, 0)
        img1b = np.rollaxis(img1, 2, 0)
        img2b = np.rollaxis(img2, 2, 0)
        img3b = np.rollaxis(img3, 2, 0)

        wins = NonUniformImageWindowExtractor(images=[img0, img1, img2, img3], image_read_fn=lambda x: x,
                                              tiling=tiling_scheme.TilingScheme(tile_shape=(16, 16), step_shape=(1, 1)))

        N0 = 0
        N1 = N0 + 85 * 85
        N2 = N1 + 75 * 95
        N3 = N2 + 80 * 90
        N4 = N3 + 90 * 80

        self.assertEqual(wins.tiling_scheme.tile_shape, (16, 16))
        self.assertEqual(wins.N_images, 4)
        self.assertEqual(wins.extractors[0].img_windows, (85, 85))
        self.assertEqual(wins.extractors[1].img_windows, (75, 95))
        self.assertEqual(wins.extractors[2].img_windows, (80, 90))
        self.assertEqual(wins.extractors[3].img_windows, (90, 80))
        self.assertEqual(wins.N, N4)
        self.assertEqual(len(wins), N4)

        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23]]))[0, :, :, :] ==
                         img1b[:, 34:50, 23:39]).all())
        self.assertTrue((wins.get_windows_by_coords(np.array([[1, 34, 23], [2, 9, 61]]))[:, :, :, :] ==
                         np.append(img1b[None, :, 34:50, 23:39], img2b[None, :, 9:25, 61:77], axis=0)).all())

        a_i = N1 + 34 * 95 + 23
        b_i = N2 + 9 * 90 + 61
        # Single index; get_window() method
        self.assertTrue((wins.get_window(a_i) ==
                         img1b[:, 34:50, 23:39]).all())
        # Single index; index operator
        self.assertTrue((wins[a_i] ==
                         img1b[:, 34:50, 23:39]).all())
        # Slice; index operator
        self.assertTrue((wins[a_i:a_i + 2] ==
                         np.append(img1b[None, :, 34:50, 23:39], img1b[None, :, 34:50, 24:40], axis=0)).all())
        # Single index in an array; get_windows() method
        self.assertTrue((wins.get_windows(np.array([a_i]))[0, :, :, :] ==
                         img1b[:, 34:50, 23:39]).all())
        # Multiple indices in an array; get_windows() method
        self.assertTrue((wins.get_windows(np.array([a_i, b_i]))[:, :, :, :] ==
                         np.append(img1b[None, :, 34:50, 23:39], img2b[None, :, 9:25, 61:77], axis=0)).all())
        # Multiple indices in an array; index operator
        self.assertTrue((wins[np.array([a_i, b_i])][:, :, :, :] ==
                         np.append(img1b[None, :, 34:50, 23:39], img2b[None, :, 9:25, 61:77], axis=0)).all())
