from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tifffile import tifffile

from . import OmeXml as bioformats
from .base_class import base


class BioReader(base):
    """BioReader Read supported image formats using Bioformats

    This class handles reading data from any of the formats supported by the
    OME Bioformats tool. It handles some of the bugs that are commonly
    encountered when handling larger images, such as the indexing issue
    encountered when an image plan is larger than 2GB.

    Note: The javabridge is not handled by the BioReader class. It must be
          initialized prior to using the BioReader class, and must be closed
          before the program terminates. An example is provided in read_image().

    For for information, visit the Bioformats page:
    https://www.openmicroscopy.org/bio-formats/

    Methods:
        BioReader(file_path): Initialize the BioReader for image in file_path
        channel_names(): Retrieve the names of each channel in the image
        num_x(): Number of pixels in the x-dimension (width)
        num_y(): Number of pixels in the y-dimension (height)
        num_z(): Number of pixels in the z-dimension (depth)
        num_c(): Number of channels in the image
        num_t(): Number of timepoints in the image
        physical_size_x(): tuple indicating physical size and units of x-dimension
        physical_size_y(): tuple indicating physical size and units of y-dimension
        physical_size_z(): tuple indicating physical size and units of z-dimension
        read_metadata(update): Returns an OMEXML class containing metadata for the image
        read_image(X,Y,Z,C,T,series): Returns a part or all of the image as numpy array
    """


    def __init__(self, file_path):
        """__init__ Initialize the a file for reading

        Prior to initializing the class, it is important to remember that
        the javabridge must be initialized. See the read_image() method
        for an example.

        Args:
            file_path (str): Path to file to read
        """
        base._file_path = file_path
        self._rdr = tifffile.TiffFile(file_path)
        self._metadata = self.read_metadata()

        # Information about image dimensions
        self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                       'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                       'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                       'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                       'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints

        # Information about data type and loading
        self._pix = {'type': self._metadata.image().Pixels.get_PixelType(),  # string indicating pixel type
                     'bpp': self._BPP[self._metadata.image().Pixels.get_PixelType()],  # bytes per pixel
                     'spp': self._metadata.image().Pixels.Channel().SamplesPerPixel}  # samples per pixel

        # number of pixels to load at a time
        self._pix['chunk'] = self._MAX_BYTES / \
                             (self._pix['spp'] * self._pix['bpp'])

        # determine if channels are interleaved
        self._pix['interleaved'] = self._pix['spp'] > 1



    def read_metadata(self, update=False):
        """read_metadata Get the metadata for the image

        This function calls the Bioformats metadata parser, which extracts metadata
        from an image. This returns the python-bioformats OMEXML class, which is a
        convenient handler for the complex xml metadata created by Bioformats.

        Most basic metadata information have their own BioReader methods, such as
        image dimensions(i.e. num_x(), num_y(), etc). However, in some cases it may
        be necessary to access the underlying metadata class.

        For information on the OMEXML class:
        https://github.com/CellProfiler/python-bioformats/blob/master/bioformats/omexml.py

        Args:
            update (bool, optional): Whether to force update of metadata. Defaults to False.
                Only needs to be used if the metadata is updated while the image is open.

        Returns:
            OMEXML: Class that
        """

        # Return cached data if it exists
        if self._metadata and not update:
            return self._metadata

        omexml = self._rdr.ome_metadata

        # Parse it using the OMEXML class
        self._metadata = bioformats.OMEXML(omexml)
        return self._metadata

    def _val_xyz(self, xyz, axis):
        """_val_xyz Utility function for validating image dimensions

        Args:
            xyz (int): Pixel value of x, y, or z dimension.
                If None, returns the maximum range of the dimension
            axis (str): Must be 'x', 'y', or 'z'

        Returns:
            list: list of ints indicating the first and last index in the dimension
        """
        assert axis in 'XYZ'
        if not xyz:
            xyz = [0, self._xyzct[axis]]
        else:
            assert len(xyz) == 2, \
                '{} must be a list or tuple of length 2.'.format(axis)
            assert xyz[0] >= 0, \
                '{}[0] must be greater than or equal to 0.'.format(axis)
            assert xyz[1] <= self._xyzct[axis], \
                '{}[1] cannot be greater than the maximum of the dimension ({}).'.format(axis, self._xyzct[axis])
        return xyz

    def _val_ct(self, ct, axis):
        """_val_ct Utility function for validating image dimensions

        Args:
            ct (int,list): List of ints indicating the channels or timepoints to load
                If None, returns a list of ints
            axis (str): Must be 'c', 't'

        Returns:
            list: list of ints indicating the first and last index in the dimension
        """
        assert axis in 'CT'
        if not ct:
            # number of timepoints
            ct = list(range(0, self._xyzct[axis]))
        else:
            assert np.any(np.greater(self._xyzct[axis], ct)), \
                'At least one of the {}-indices was larger than largest index ({}).'.format(axis, self._xyzct[axis] - 1)
            assert np.any(np.less_equal(0, ct)), \
                'At least one of the {}-indices was less than 0.'.format(axis)
            assert len(ct) != 0, \
                'At least one {}-index must be selected.'.format(axis)
        return ct

    def pixel_type(self):
        """pixel_type Get the pixel type

        One of the following strings will be returned:

        'uint8':  Unsigned 8-bit pixel type
        'int8':   Signed 8-bit pixel type
        'uint16': Unsigned 8-bit pixel type
        'int16':  Signed 16-bit pixel type
        'uint32': Unsigned 32-bit pixel type
        'int32':  Signed 32-bit pixel type
        'float':  IEEE single-precision pixel type
        'double': IEEE double precision pixel type

        Returns:
            str: One of the above data types.
        """

        return self._metadata.image(0).Pixels.PixelType

    def _chunk_indices(self, X, Y, Z):
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2

        offsets = []
        bytecounts = []

        x_tiles = np.arange(X[0] // 1024, np.ceil(X[1] / 1024), dtype=int)
        y_tile_stride = np.ceil(self.num_x() / 1024)

        for z in range(Z[0], Z[1]):
            for y in range(Y[0] // 1024, int(np.ceil(Y[1] / 1024))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()

                offsets.extend([self._rdr.pages[z].dataoffsets[i] for i in ind])
                bytecounts.extend([self._rdr.pages[z].databytecounts[i] for i in ind])

        return offsets, bytecounts

    def _decode_tile(self, args):

        keyframe = self._keyframe
        out = self._out
        tile_shape = self._tile_shape

        w, l, d = self._tile_indices[args[1]]

        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)

        if segment is None:
            segment = keyframe.nodata

        out[l: l + shape[1],
        w: w + shape[2],
        d, 0, 0] = segment.squeeze()

    def read_image(self, X=None, Y=None, Z=None, C=None, T=None, series=None):
        """read_image Read the image

        Read the image. A 5-dimmensional numpy.ndarray is always returned.

        Args:
            X ([tuple,list], optional): 2-tuple indicating the (min,max) range of
                pixels to load. If None, loads the full range.
                Defaults to None.
            Y ([tuple,list], optional): 2-tuple indicating the (min,max) range of
                pixels to load. If None, loads the full range.
                Defaults to None.
            Z ([tuple,list], optional): 2-tuple indicating the (min,max) range of
                pixels to load. If None, loads the full range.
                Defaults to None.
            C ([tuple,list], optional): tuple or list of values indicating channel
                indices to load. If None, loads the full range.
                Defaults to None.
            T ([tuple,list], optional): tuple or list of values indicating timepoints
                to load. If None, loads the full range.
                Defaults to None.
            series (tuple, optional): Placeholder. Currently does nothing.

        Returns:
            numpy.ndarray: A 5-dimensional numpy array.

        Example:
            # Import javabridge and start the vm
            import javabridge
            javabridge.start_vm(class_path=bioformats.JARS)

            # Path to bioformats supported image
            image_path = 'path/to/image'

            # Create the BioReader object
            bf = BioReader(image_path)

            # Load the full image
            image = bf.read_image()

            # Only load the first 256x256 pixels, will still load all Z,C,T dimensions
            image = bf.read_image(X=(0,256),Y=(0,256))

            # Only load the second channel
            image = bf.read_image(C=[1])

            # Done executing program, so kill the vm. If the program needs to be run
            # again, a new interpreter will need to be spawned to start the vm.
            javabridge.kill_vm()
        """

        # Validate inputs
        X = self._val_xyz(X, 'X')
        Y = self._val_xyz(Y, 'Y')
        Z = self._val_xyz(Z, 'Z')
        C = self._val_ct(C, 'C')
        T = self._val_ct(T, 'T')

        x_range = X[1] - X[0]
        y_range = Y[1] - Y[0]
        x_tile_shape = int(np.ceil(X[1] / 1024) - X[0] // 1024)
        y_tile_shape = int(np.ceil(Y[1] / 1024) - Y[0] // 1024)
        z_tile_shape = Z[1] - Z[0]

        self._tile_shape = (x_tile_shape, y_tile_shape, z_tile_shape)

        self._out = np.zeros([y_tile_shape * 1024, x_tile_shape * 1024, z_tile_shape,
                              1, 1], self._pix['type'])

        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Do the work
        offsets, bytecounts = self._chunk_indices(X, Y, Z)
        self._tile_indices = []
        for z in range(0, Z[1] - Z[0]):
            for y in range(0, y_tile_shape):
                for x in range(0, x_tile_shape):
                    self._tile_indices.append((x * 1024, y * 1024, z))

        with ThreadPoolExecutor(self._keyframe.maxworkers) as executor:
            executor.map(self._decode_tile, fh.read_segments(offsets, bytecounts))

        xi = X[0] - 1024 * (X[0] // 1024)
        yi = Y[0] - 1024 * (Y[0] // 1024)

        return self._out[yi:yi + y_range, xi:xi + x_range, ...]

    def _fetch(self):
        """_fetch Method for fetching image supertiles
        This method is intended to be run within a thread, and grabs a
        chunk of the image according to the coordinates in a queue.

        Currently, this function will only grab the first Z, C, and T
        positions regardless of what Z, C, and T coordinate are provided
        to the function. This function will need to be changed in then
        future to account for this.

        If the first value in X or Y is negative, then the image is
        pre-padded with the number of pixels equal to the absolute value
        of the negative number.

        If the last value in X or Y is larger than the size of the
        image, then the image is post-padded with the difference between
        the number and the size of the image.
        Input coordinate are read from the _supertile_index Queue object.
        Output data is stored in the _raw_buffer Queue object.

        As soon as the method is executed, a boolean value is put into the
        _data_in_buffer Queue to indicate that data is either in the buffer
        or will be put into the buffer.
        """

        self._data_in_buffer.put(True)
        X, Y, Z, C, T = self._supertile_index.get()

        # Attach the jvm to the thread
        jutil.attach()

        # Determine padding if needed
        reflect_x = False
        x_min = X[0]
        x_max = X[1]
        y_min = Y[0]
        y_max = Y[1]
        prepad_x = 0
        postpad_x = 0
        prepad_y = 0
        postpad_y = 0
        if x_min < 0:
            prepad_x = abs(x_min)
            x_min = 0
        if y_min < 0:
            prepad_y = abs(y_min)
            y_min = 0
        if x_max > self.num_x():
            if x_min >= self.num_x():
                x_min = 1024 * ((self.num_x() - 1) // 1024)
                reflect_x = True
            x_max = self.num_x()
            postpad_x = x_max - self.num_x()
        if y_max > self.num_y():
            y_max = self.num_y()
            postpad_y = y_max - self.num_y()

        # Read the image
        I = self.read_image([x_min, x_max], [y_min, y_max], [0, 1], [0], [0]).squeeze()
        if reflect_x:
            I = np.fliplr(I)

        # Pad the image if needed
        if sum(1 for p in [prepad_x, prepad_y, postpad_x, postpad_y] if p != 0) > 0:
            I = np.pad(I, ((prepad_y, postpad_y), (prepad_x, postpad_x)), mode='symmetric')

        # Store the data in the buffer
        self._raw_buffer.put(I)

        # Detach the jvm
        jutil.detach()

        return I

    def _buffer_supertile(self, column_start, column_end):
        """_buffer_supertile Process the pixel buffer
        Give the column indices of the data to process, and determine if
        the buffer needs to be processed. This method performs two operations
        on the buffer. First, it checks to see if data in the buffer can be
        shifted out of the buffer if it's already been processed, where data
        before column_start is assumed to have been processed. Second, this
        function loads data into the buffer if the image reader has made some
        available and there is room in _pixel_buffer for it.
        Args:
            column_start ([int]): First column index of data to be loaded
            column_end ([int]): Last column index of data to be loaded
        """

        # If the column indices are outside what is available in the buffer,
        # shift the buffer so more data can be loaded.
        if column_end - self._tile_x_offset >= 1024:
            x_min = column_start - self._tile_x_offset
            x_max = self._pixel_buffer.shape[1] - x_min
            self._pixel_buffer[:, 0:x_max] = self._pixel_buffer[:, x_min:]
            self._pixel_buffer[:, x_max:] = 0
            self._tile_x_offset = column_start
            self._tile_last_column = np.argwhere((self._pixel_buffer == 0).all(axis=0))[0, 0]

        # Is there data in the buffer?
        if (self._supertile_index.qsize() > 0 or self._data_in_buffer.qsize() > 0):

            # If there is data in the _raw_buffer, return if there isn't room to load
            # it into the _pixel_buffer
            if self._pixel_buffer.shape[1] - self._tile_last_column < 1024:
                return

            I = self._raw_buffer.get()
            if self._tile_last_column == 0:
                self._pixel_buffer[:I.shape[0], :I.shape[1]] = I
                self._tile_last_column = I.shape[1]
                self._tile_x_offset = column_start
            else:
                self._pixel_buffer[:I.shape[0], self._tile_last_column:self._tile_last_column + I.shape[1]] = I
                self._tile_last_column += I.shape[1]

            self._data_in_buffer.get()

    def _get_tiles(self, X, Y, Z, C, T):
        """_get_tiles Handle data buffering and tiling
        This function returns tiles of data according to the input
        coordinates. The X, Y, Z, C, and T are lists of lists, where
        each internal list indicates a set of coordinates specifying
        the range of pixel values to grab from an image.
        Args:
            X (list): List of 2-tuples indicating the (min,max)
                range of pixels to load within a tile.
            Y (list): List of 2-tuples indicating the (min,max)
                range of pixels to load within a tile.
            Z (None): Placeholder, to be implemented.
            C (None): Placeholder, to be implemented.
            T (None): Placeholder, to be implemented.
        Returns:
            numpy.ndarray: 2-dimensional ndarray.
        """

        self._buffer_supertile(X[0][0], X[0][1])

        if X[-1][0] - self._tile_x_offset > 1024:
            shift_buffer = True
            split_ind = 0
            while X[split_ind][0] - self._tile_x_offset < 1024:
                split_ind += 1
        else:
            shift_buffer = False
            split_ind = len(X)

        # Tile the data
        num_rows = Y[0][1] - Y[0][0]
        num_cols = X[0][1] - X[0][0]
        num_tiles = len(X)
        images = np.zeros((num_tiles, num_rows, num_cols, 1), dtype=self.pixel_type())

        for ind in range(split_ind):
            images[ind, :, :, 0] = self._pixel_buffer[Y[ind][0] - self._tile_y_offset:Y[ind][1] - self._tile_y_offset,
                                   X[ind][0] - self._tile_x_offset:X[ind][1] - self._tile_x_offset]

        if split_ind != num_tiles:
            self._buffer_supertile(X[-1][0], X[-1][1])
            for ind in range(split_ind, num_tiles):
                images[ind, :, :, 0] = self._pixel_buffer[
                                       Y[ind][0] - self._tile_y_offset:Y[ind][1] - self._tile_y_offset,
                                       X[ind][0] - self._tile_x_offset:X[ind][1] - self._tile_x_offset]

        return images

    def maximum_batch_size(self, tile_size, tile_stride=None):
        """maximum_batch_size Maximum allowable batch size for tiling
        The pixel buffer only loads at most two supertiles at a time. If the batch
        size is too large, then the tiling function will attempt to create more
        tiles than what the buffer holds. To prevent the tiling function from doing
        this, there is a limit on the number of tiles that can be retrieved in a
        single call. This function determines what the largest number of retreivable
        batches is.
        Args:
            tile_size (list): The height and width of the tiles to retrieve
            tile_stride (list, optional): If None, defaults to tile_size.
                Defaults to None.
        Returns:
            int: Maximum allowed number of batches that can be retrieved by the
                iterate method.
        """
        if tile_stride == None:
            tile_stride = tile_size

        xyoffset = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]

        num_tile_rows = int(np.ceil(self.num_y() / tile_stride[0]))
        num_tile_cols = (1024 - xyoffset[1]) // tile_stride[1]
        if num_tile_cols == 0:
            num_tile_cols = 1

        return int(num_tile_cols * num_tile_rows)

    def iterate(self, tile_size, tile_stride=None, batch_size=None, channels=[0]):
        """iterate Iterate through tiles of an image
        This method is an iterator to load tiles of an image. This method
        buffers the loading of pixels asynchronously to quickly deliver
        images of the appropriate size.
        Args:
            tile_size ([list,tuple]): A list/tuple of length 2, indicating
                the height and width of the tiles to return.
            tile_stride ([list,tuple], optional): A list/tuple of length 2,
                indicating the row and column stride size. If None, then
                tile_stride = tile_size.
                Defaults to None.
            batch_size (int, optional): Number of tiles to return on each
                iteration. Defaults to 32.
            channels (list, optional): A placeholder. Only the first channel
                is ever loaded. Defaults to [0].
        Yields:
            numpy.ndarray: A 4-d array where the dimensions are
                [tile_num,tile_size[0],tile_size[1],channels]
            tuple: A tuple containing lists of X,Y,Z,C,T indices

        Example:
            from bfio import BioReader
            import matplotlib.pyplot as plt

            br = BioReader('/path/to/file')

            for tiles,ind in br.iterate(tile_size=[256,256],tile_stride=[200,200]):
                for i in tiles.shape[0]:
                    print('Displaying tile with X,Y coords: {},{}'.format(ind[i][0],ind[i][1]))
                    plt.figure()
                    plt.imshow(tiles[ind,:,:,0].squeeze())
                    plt.show()

        """

        # Enure that the number of tiles does not exceed the width of a supertile
        if batch_size == None:
            batch_size = min([32, self.maximum_batch_size(tile_size, tile_stride)])
        else:
            assert batch_size <= self.maximum_batch_size(tile_size, tile_stride), \
                'batch_size must be less than or equal to {}.'.format(self.maximum_batch_size(tile_size, tile_stride))

        # input error checking
        assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
        if tile_stride != None:
            assert len(tile_stride) == 2, "stride must be a list with 2 elements"
        else:
            stride = tile_size

        # calculate padding if needed
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]
            xypad = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]
            xypad[0] = xyoffset[0] + (tile_stride[0] - np.mod(self.num_y(), tile_stride[0])) / 2
            xypad[1] = xyoffset[1] + (tile_stride[1] - np.mod(self.num_x(), tile_stride[1])) / 2
            xypad = ((int(xyoffset[0]), int(2 * xypad[0] - xyoffset[0])),
                     (int(xyoffset[1]), int(2 * xypad[1] - xyoffset[1])))
        else:
            xyoffset = [0, 0]
            xypad = ((0, max([tile_size[0] - tile_stride[0], 0])),
                     (0, max([tile_size[1] - tile_stride[1], 0])))

        # determine supertile sizes
        y_tile_dim = int(np.ceil((self.num_y() - 1) / 1024))
        x_tile_dim = 1

        # Initialize the pixel buffer
        self._pixel_buffer = np.zeros((y_tile_dim * 1024 + tile_size[0], 2 * x_tile_dim * 1024 + tile_size[1]),
                                      dtype=self.pixel_type())
        self._tile_x_offset = -xypad[1][0]
        self._tile_y_offset = -xypad[0][0]

        # Generate the supertile loading order
        tiles = []
        y_tile_list = list(range(0, self.num_y() + xypad[0][1], 1024 * y_tile_dim))
        if y_tile_list[-1] != 1024 * y_tile_dim:
            y_tile_list.append(1024 * y_tile_dim)
        if y_tile_list[0] != xypad[0][0]:
            y_tile_list[0] = -xypad[0][0]
        x_tile_list = list(range(0, self.num_x() + xypad[1][1], 1024 * x_tile_dim))
        if x_tile_list[-1] < self.num_x() + xypad[1][1]:
            x_tile_list.append(x_tile_list[-1] + 1024)
        if x_tile_list[0] != xypad[1][0]:
            x_tile_list[0] = -xypad[1][0]
        for yi in range(len(y_tile_list) - 1):
            for xi in range(len(x_tile_list) - 1):
                y_range = [y_tile_list[yi], y_tile_list[yi + 1]]
                x_range = [x_tile_list[xi], x_tile_list[xi + 1]]
                tiles.append([x_range, y_range])
                self._supertile_index.put((x_range, y_range, [0, 1], [0], [0]))

        # Start the thread pool and start loading the first supertile
        thread_pool = ThreadPoolExecutor(max_workers=2)
        self._fetch_thread = thread_pool.submit(self._fetch)

        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = np.array(np.arange(-xypad[1][0], self.num_x(), tile_stride[1]))
        y_list = np.array(np.arange(-xypad[0][0], self.num_y(), tile_stride[0]))
        for x in x_list:
            for y in y_list:
                X.append([x, x + tile_size[1]])
                Y.append([y, y + tile_size[0]])
                Z.append([0, 1])
                C.append(channels)
                T.append([0])

        # Set up batches
        batches = list(range(0, len(X), batch_size))

        # get the first batch
        b = min([batch_size, len(X)])
        index = (X[0:b], Y[0:b], Z[0:b], C[0:b], T[0:b])
        images = self._get_tiles(*index)

        # start looping through batches
        for bn in batches[1:]:
            # start the thread to get the next batch
            b = min([bn + batch_size, len(X)])
            self._tile_thread = thread_pool.submit(self._get_tiles, X[bn:b], Y[bn:b], Z[bn:b], C[bn:b], T[bn:b])

            # Load another supertile if possible
            if self._supertile_index.qsize() > 0 and not self._fetch_thread.running():
                self._fetch_thread = thread_pool.submit(self._fetch)

            # return the curent set of images
            yield images, index

            # get the images from the thread
            index = (X[bn:b], Y[bn:b], Z[bn:b], C[bn:b], T[bn:b])
            images = self._tile_thread.result()

        thread_pool.shutdown()

        # return the last set of images
        yield images, index
