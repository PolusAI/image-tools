import threading, typing, numpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from bfio import backends
from bfio.base_classes import BioBase

class BioReader(BioBase):
    """BioReader Read supported image formats using Bioformats

    This class handles the reading of data from any Bioformats supported file
    formats, but is specially optimized for handling the OME tiled tiff format.
    There is a Java backend version of this tool that directly interacts with
    the Bioformats codebase written in Java through the javabridge package,
    permitting reading of any file format supported by Bioformats. There is also
    a Python backend, which is significantly faster but is limited to reading
    only OME tiled tiff images.
    
    One of the features of this class is that it handles some of the potential
    issues that are not caught by the python-bioformats package. For example,
    this class can automatically read images larger than 2GB in size, whereas
    the python-bioformats class will throw an error due to the way that Java
    handles file reading.

    For for information, visit the Bioformats page:
    https://www.openmicroscopy.org/bio-formats/

    Note:
        The javabridge is not handled by the BioReader class. It must be
        initialized prior to using the BioReader class, and must be closed
        before the program terminates. An example is provided in read_image().
    """

    def __init__(self,
                 file_path: typing.Union[str,Path],
                 max_workers: typing.Union[int,None] = None,
                 backend: str = 'python'):
        """
        Parameters:
            file_path: Path to file to read
            max_workers: Number of threads used to read and image. *Default is
                half the number of detected cores.*
            backend: Must be ``python`` or ``java``. *Default is python.*
        """

        # Initialize BioBase
        super(BioReader, self).__init__(file_path,
                                        max_workers=max_workers,
                                        backend=backend)
        
        # Ensure backend is supported
        if self._backend_name == 'python':
            self._backend = backends.PythonReader(self)
        elif self._backend_name == 'java':
            self._backend = backends.JavaReader(self)
        else:
            raise ValueError('backend must be "python" or "java"')
        
        # Preload the metadata
        self._metadata = self._backend.read_metadata()
                           
    def __getitem__(self,keys: typing.Union[tuple,slice]) -> numpy.ndarray:
        """Image loading using numpy-like indexing
        
        This is an abbreviated method of accessing the :attr:`~.read` method,
        where a portion of the image will be loaded using numpy-like slicing
        syntax. Up to 5 dimensions can be designated depending on the number of
        available dimensions in the image array (Y, X, Z, C, T).
        
        Note:
            Not all methods of indexing can be used, and some indexing will lead
            to unexpected results. For example, logical indexing cannot be used,
            and step sizes in slice objects is ignored for the first three
            indices. This means and index such as ``[0:100:2,0:100:2,0,0,0]``
            will return a 100x100x1x1x1 numpy array.
        
        Parameters:
            keys: numpy-like slicing used to load a section of an image.
        
        Returns:
            A numpy.ndarray where trailing empty dimensions are removed.
            
        Example:

            .. code-block:: python

                import bfio
                
                # Initialize the bioreader
                br = bfio.BioReader('Path/To/File.ome.tif')
                
                # Load and copy a 100x100 array of pixels
                a = br[:100,:100,0,0,0]
                
                # Slice steps sizes are ignored for the first 3 indices, so this
                # returns the same as above
                a = br[0:100:2,0:100:2,0,0,0]
                
                # The last two dimensions can receive a tuple or list as input
                # Load the first and third channel
                a = br[:100,100,0,(0,2),0]
                
                # If the file is 3d, load the first 10 z-slices
                b = br[...,:10,0,0]
        """
        
        # Dimension ordering and index initialization
        dims = 'YXZCT'
        ind = {d:None for d in dims}
        
        # If an empty slice, load the whole image
        if not isinstance(keys,tuple):
            if isinstance(keys,slice) and keys.start == None and keys.stop == None and keys.step==None:
                pass
            else:
                raise ValueError('If only a single index is supplied, it must be an empty slice ([:]).')
            
        # If not an empty slice, parse the key tuple
        else:
            
            # At most, 5 indices can be indicated
            if len(keys) > 5:
                raise ValueError('Found {} indices, but at most 5 indices may be supplied.'.format(len(keys)))
            
            # If the first key is an ellipsis, read backwards
            if keys[0] == Ellipsis:
                
                # If the last key is an ellipsis, throw an error
                if keys[-1]==Ellipsis:
                    raise ValueError('Ellipsis (...) may be used in either the first or last index, not both.')
                
                dims = [d for d in reversed(dims)]
                keys = [k for k in reversed(keys)]
            
            # Get key values
            for dim,key in zip(dims,keys):
                
                if isinstance(key,slice):
                    # For CT dimensions, generate a list from slices
                    if dim in 'CT':
                        ind[dim] = [v for v in range(key)]
                    
                    # For XYZ dimensions, get start and stop of slice, ignore step
                    else:
                        start = 0 if key.start == None else key.start
                        stop = 0 if key.stop == None else key.stop
                        ind[dim] = [start,stop]
                        
                elif isinstance(key,(int,tuple,list)):
                    # Only the last two dimensions can use int, tuple, or list indexing
                    if dim in 'CT':
                        if isinstance(key,int):
                            ind[dim] = [key]
                        else:
                            ind[dim] = key
                    else:
                        raise ValueError('The index in position {} must be a slice type.'.format(dims.find(key)))
                elif key==Ellipsis:
                    if dims.find(dim)+1 < len(keys):
                        raise ValueError('Ellipsis may only be used in the first or last index.')
                else:
                    raise ValueError('Did not recognize indexing value of type: {}'.format(dims.find(key)))
        
        return self.read(**ind)

    def read(self,
             X: typing.Union[list,tuple,None] = None,
             Y: typing.Union[list,tuple,None] = None,
             Z: typing.Union[list,tuple,None] = None,
             C: typing.Union[list,tuple,None] = None,
             T: typing.Union[list,tuple,None] = None) -> numpy.ndarray:
        """Read the image

        Read the all or part of the image. A n-dimmensional numpy.ndarray is
        returned such that all trailing empty dimensions will be removed.
        
        For example, if an image is read and it represents an xz plane, then the
        shape will be [1,m,n].

        Args:
            X: The (min,max) range of pixels to load along the x-axis (columns).
                If None, loads the full range. *Defaults to None.*
            Y: The (min,max) range of pixels to load along the y-axis (rows). If
                None, loads the full range. *Defaults to None.*
            Z: The (min,max) range of pixels to load along the z-axis (depth).
                If None, loads the full range. *Defaults to None.*
            C: Values indicating channel indices to load. If None, loads the
                full range. *Defaults to None.*
            T: Values indicating timepoints to load. If None, loads the full
                range. *Defaults to None.*

        Returns:
            A 5-dimensional numpy array.
        """
        # Validate inputs
        X = self._val_xyz(X, 'X')
        Y = self._val_xyz(Y, 'Y')
        Z = self._val_xyz(Z, 'Z')
        C = self._val_ct(C, 'C')
        T = self._val_ct(T, 'T')
        
        # Define tile bounds
        X_tile_start = (X[0]//self._TILE_SIZE) * self._TILE_SIZE
        Y_tile_start = (Y[0]//self._TILE_SIZE) * self._TILE_SIZE
        X_tile_end = numpy.ceil(X[1]/self._TILE_SIZE).astype(int) * self._TILE_SIZE
        Y_tile_end = numpy.ceil(Y[1]/self._TILE_SIZE).astype(int) * self._TILE_SIZE
        X_tile_shape = X_tile_end - X_tile_start
        Y_tile_shape = Y_tile_end - Y_tile_start
        Z_tile_shape = Z[1]-Z[0]

        # Initialize the output
        output = numpy.zeros([Y_tile_shape,
                              X_tile_shape,
                              Z_tile_shape,
                              len(C),
                              len(T)],
                             dtype=self.dtype)
        
        # Read the image
        self._backend.read_image([X_tile_start,X_tile_end],
                                 [Y_tile_start,Y_tile_end],
                                 Z,
                                 C,T,output)

        return output[Y[0]-Y_tile_start:Y[1]-Y_tile_start,X[0]-X_tile_start:X[1]-X_tile_start,...]

    def _fetch(self) -> numpy.ndarray:
        """Method for fetching image supertiles
        
        This method is intended to be run within a thread, and grabs a chunk of
        the image according to the coordinates in a queue.

        Currently, this function will only grab the first Z, C, and T positions
        regardless of what Z, C, and T coordinate are provided to the function.
        This function will need to be changed in then future to account for
        this.

        If the first value in X or Y is negative, then the image is pre-padded
        with the number of pixels equal to the absolute value of the negative
        number.

        If the last value in X or Y is larger than the size of the image, then
        the image is post-padded with the difference between the number and the
        size of the image. Input coordinate are read from the _supertile_index
        Queue object. Output data is stored in the _raw_buffer Queue object.

        As soon as the method is executed, a boolean value is put into the
        _data_in_buffer Queue to indicate that data is either in the buffer or
        will be put into the buffer.
        
        Returns:
            An image supertile
        """

        self._data_in_buffer.put(True)
        X, Y, Z, C, T = self._supertile_index.get()

        # Attach the jvm to the thread if present
        self._backend.attach()

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
                x_min = self._TILE_SIZE * ((self.num_x() - 1) // self._TILE_SIZE)
                reflect_x = True
            x_max = self.num_x()
            postpad_x = x_max - self.num_x()
        if y_max > self.num_y():
            y_max = self.num_y()
            postpad_y = y_max - self.num_y()

        # Read the image
        I = self.read([x_min, x_max], [y_min, y_max], [0, 1], [0], [0]).squeeze()
        if reflect_x:
            I = numpy.fliplr(I)

        # Pad the image if needed
        if sum(1 for p in [prepad_x, prepad_y, postpad_x, postpad_y] if p != 0) > 0:
            I = numpy.pad(I, ((prepad_y, postpad_y), (prepad_x, postpad_x)), mode='symmetric')

        # Store the data in the buffer
        self._raw_buffer.put(I)

        # Detach the jvm
        self._backend.detach()

        return I

    def _buffer_supertile(self, column_start: int, column_end: int):
        """_buffer_supertile Process the pixel buffer
        
        Give the column indices of the data to process, and determine if the
        buffer needs to be processed. This method performs two operations on the
        buffer. First, it checks to see if data in the buffer can be shifted out
        of the buffer if it's already been processed, where data before
        column_start is assumed to have been processed. Second, this function
        loads data into the buffer if the image reader has made some available
        and there is room in _pixel_buffer for it.
        
        Args:
            column_start: First column index of data to be loaded
            column_end: Last column index of data to be loaded
        """

        # If the column indices are outside what is available in the buffer,
        # shift the buffer so more data can be loaded.
        if column_end - self._tile_x_offset >= self._TILE_SIZE:
            x_min = column_start - self._tile_x_offset
            x_max = self._pixel_buffer.shape[1] - x_min
            self._pixel_buffer[:, 0:x_max] = self._pixel_buffer[:, x_min:]
            self._pixel_buffer[:, x_max:] = 0
            self._tile_x_offset = column_start
            self._tile_last_column = numpy.argwhere((self._pixel_buffer == 0).all(axis=0))[0, 0]

        # Is there data in the buffer?
        if (self._supertile_index.qsize() > 0 or self._data_in_buffer.qsize() > 0):

            # If there is data in the _raw_buffer, return if there isn't room to load
            # it into the _pixel_buffer
            if self._pixel_buffer.shape[1] - self._tile_last_column < self._TILE_SIZE:
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

    def _get_tiles(self,
                   X:typing.List[typing.List[int]],
                   Y:typing.List[typing.List[int]],
                   Z:typing.List[typing.List[int]],
                   C:typing.List[typing.List[int]],
                   T:typing.List[typing.List[int]]) -> numpy.ndarray:
        """_get_tiles Handle data buffering and tiling
        
        This function returns tiles of data according to the input coordinates.
        The X, Y, Z, C, and T are lists of lists, where each internal list
        indicates a set of coordinates specifying the range of pixel values to
        grab from an image.
        
        Args:
            X: List of 2-tuples indicating the (min,max) range of pixels to load
                within a tile.
            Y: List of 2-tuples indicating the (min,max) range of pixels to load
                within a tile.
            Z: Placeholder, to be implemented.
            C: Placeholder, to be implemented.
            T: Placeholder, to be implemented.
            
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
        images = numpy.zeros((num_tiles, num_rows, num_cols, 1), dtype=self.pixel_type())

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

    def __call__(self,
                 tile_size: typing.Union[list,tuple],
                 tile_stride: typing.Union[list,tuple,None] = None,
                 batch_size: typing.Union[int,None] = None,
                 channels: typing.Union[list] = [0]) -> typing.Iterable[typing.Tuple[numpy.ndarray,tuple]]:
        """Iterate through tiles of an image
        
        The BioReader object can be called, and will act as an iterator to load
        tiles of an image. The iterator buffers the loading of pixels
        asynchronously to quickly deliver images of the appropriate size.
        
        Args:
            tile_size: A list/tuple of length 2, indicating the height and width
                of the tiles to return.
            tile_stride: A list/tuple of length 2, indicating the row and column
                stride size. If None, then tile_stride = tile_size. *Defaults to
                None.*
            batch_size: Number of tiles to return on each iteration. *Defaults
                to None, which is the smaller of 32 or the*
                :attr:`~.maximum_batch_size`
            channels: A placeholder. Only the first channel
                is ever loaded. *Defaults to [0].*
                
        Returns:
            A tuple containing a 4-d numpy array and a tuple containing a list
            of X,Y,Z,C,T indices. The numpy array has dimensions 
            ``[tile_num,tile_size[0],tile_size[1],channels]``

        Example:
            .. code:: python
            
                from bfio import BioReader
                import matplotlib.pyplot as plt

                br = BioReader('/path/to/file')

                for tiles,ind in br(tile_size=[256,256],tile_stride=[200,200]):
                    for i in tiles.shape[0]:
                        print('Displaying tile with X,Y coords: {},{}'.format(ind[i][0],ind[i][1]))
                        plt.figure()
                        plt.imshow(tiles[ind,:,:,0].squeeze())
                        plt.show()

        """
        
        self._iter_tile_size = tile_size
        self._iter_tile_stride = tile_stride
        self._iter_batch_size = batch_size
        self._iter_channels = channels
        
        return self

    def __iter__(self):
        
        tile_size = self._iter_tile_size
        tile_stride = self._iter_tile_stride
        batch_size = self._iter_batch_size
        channels = self._iter_channels
        
        if tile_size == None:
            raise SyntaxError('Cannot directly iterate over a BioReader object. Call it (i.e. for i in bioreader(256,256))')

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
            xypad[0] = xyoffset[0] + (tile_stride[0] - numpy.mod(self.Y, tile_stride[0])) / 2
            xypad[1] = xyoffset[1] + (tile_stride[1] - numpy.mod(self.X, tile_stride[1])) / 2
            xypad = ((int(xyoffset[0]), int(2 * xypad[0] - xyoffset[0])),
                     (int(xyoffset[1]), int(2 * xypad[1] - xyoffset[1])))
        else:
            xyoffset = [0, 0]
            xypad = ((0, max([tile_size[0] - tile_stride[0], 0])),
                     (0, max([tile_size[1] - tile_stride[1], 0])))

        # determine supertile sizes
        y_tile_dim = int(numpy.ceil((self.Y - 1) / 1024))
        x_tile_dim = 1

        # Initialize the pixel buffer
        self._pixel_buffer = numpy.zeros((y_tile_dim * 1024 + tile_size[0], 2 * x_tile_dim * 1024 + tile_size[1]),
                                         dtype=self.dtype)
        self._tile_x_offset = -xypad[1][0]
        self._tile_y_offset = -xypad[0][0]

        # Generate the supertile loading order
        tiles = []
        y_tile_list = list(range(0, self.Y + xypad[0][1], 1024 * y_tile_dim))
        if y_tile_list[-1] != 1024 * y_tile_dim:
            y_tile_list.append(1024 * y_tile_dim)
        if y_tile_list[0] != xypad[0][0]:
            y_tile_list[0] = -xypad[0][0]
        x_tile_list = list(range(0, self.X + xypad[1][1], 1024 * x_tile_dim))
        if x_tile_list[-1] < self.X + xypad[1][1]:
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
        thread_pool = ThreadPoolExecutor(self._max_workers)
        self._fetch_thread = thread_pool.submit(self._fetch)

        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = numpy.array(numpy.arange(-xypad[1][0], self.X, tile_stride[1]))
        y_list = numpy.array(numpy.arange(-xypad[0][0], self.Y, tile_stride[0]))
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
        
    def close(self):
        """Closes the image reader"""
        self._backend.close()
        
    def __enter__(self):
        """Handle entrance to a context manager
        
        This code is called when a `with` statement is used. This allows the
        BioReader object to be used like this:
        
        with bfio.BioReader('Path/To/File.ome.tif') as reader:
            ...
        """
        return self
    
    def __exit__(self, type_class, value, traceback):
        """Handle exit from the context manager

        This code runs when exiting a `with` statement.
        """
        self.close()

class BioWriter(BioBase):
        """BioWriter Write OME tiled tiffs using Bioformats

        This class handles writing data to OME tiled tiff format using the
        OME Bioformats tool. Like the BioReader class, it handles writing
        large images (>2GB).

        Once the class is initialized, and once the write_image() function
        is called at least once, most of the methods to set the metadata
        information will throw an error. Therefore, make sure to set all
        metadata before image writing is started.

        Note: The javabridge is not handled by the BioReader class. It must be
              initialized prior to using the BioReader class, and must be closed
              before the program terminates. An example is provided in read_image().

        For for information, visit the Bioformats page:
        https://www.openmicroscopy.org/bio-formats/
        """

        __writer = None
        # Variables for saving multipage tiffs
        _current_page = None
        _ifds = None
        _databytecounts = []
        _tags = []  # list of (code, ifdentry, ifdvalue, writeonce)
        _page_open = False

        @property
        def set_dim(self):
            # Information about image dimensions
            return super().set_dim

        @set_dim.setter
        def set_dim(self,metadata):
            super(BioWriter, type(self)).set_dim.fset(self,metadata)

        @property
        def set_read(self):
            return super().set_read

        @set_read.setter
        def set_read(self, new_value):
            #super(BioWriter, self.__class__).set_read.fset(self, False)
            super(BioWriter, type(self)).set_read.fset(self, False)

        # self._backend = backends.BACKEND
        # self._writer = PythonWriterBackend()
        # if self._backend != 'PythonBackend':
        #     self._writer = JavaWriterBackend()

        def __init__(self, file_path, image=None,
                     X=None, Y=None, Z=None, C=None, T=None,
                     metadata=None, max_workers=None):
            """__init__ Initialize an output OME tiled tiff file

            Prior to initializing the class, it is important to remember that
            the javabridge must be initialized. See the write_image() method
            for an example.

            Args:
                file_path (str): Path to output file
                image (numpy.ndarray, optional): If image is defined, this will
                    be used to determine data type. This input must be a
                    5-dimensional numpy.ndarray. If none of the positional
                    variables are used (i.e. X,Y,Z,C,T), then the size of the
                    output file be be inferred from this array.
                    Defaults to None.
                X (int, optional): Number of pixels in x-dimension (width). If
                    this is None, then the width of the image is set to
                    image.shape[0] if image is defined, and is set to 1 otherwise.
                    Defaults to None.
                Y (int, optional): Number of pixels in y-dimension (height). If
                    this is None, then the height of the image is set to
                    image.shape[1] if image is defined, and is set to 1 otherwise.
                    Defaults to None.
                Z (int, optional): Number of pixels in z-dimension (depth). If
                    this is None, then the depth of the image is set to
                    image.shape[2] if image is defined, and is set to 1 otherwise.
                    Defaults to None.
                C (int, optional): Number of image channels. If this is None, then
                    the number of channels is set to image.shape[4] if image is
                    defined, and is set to 1 otherwise.
                    Defaults to None.
                T (int, optional): Number of image timepoints. If this is None,
                    then the number of timepoints is set to image.shape[5] if image
                    is defined, and is set to 1 otherwise.
                    Defaults to None.
                metadata (bioformats.omexml.OMEXML, optional): If defined, all other
                    inputs except file_path are ignored. This directly sets the
                    ome tiff metadata using the OMEXML class.
                    Defaults to None.
            """
            super(BioWriter, self).__init__(file_path, _metadata=metadata, max_workers=max_workers)

            #BioBase.set_read.fset(self,False)
            self._lock = threading.Lock()
            if metadata:
                assert metadata.__class__.__name__ == "OMEXML"
                self._metadata = self.xml_metadata(str(metadata))
                self._metadata.image(0).Name = file_path
                self._metadata.image().Pixels.channel_count = self._xyzct['C']
                self._metadata.image().Pixels.DimensionOrder = self.xml_metadata(var=1).DO_XYZCT
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
            elif isinstance(image, numpy.ndarray):
                assert len(image.shape) == 5, "Image must be 5-dimensional (x,y,z,c,t)."
                x = X if X else image.shape[1]
                y = Y if Y else image.shape[0]
                z = Z if Z else image.shape[2]
                c = C if C else image.shape[3]
                t = T if T else image.shape[4]
                self._xyzct = {'X': x,  # image width
                               'Y': y,  # image height
                               'Z': z,  # image depth
                               'C': c,  # number of channels
                               'T': t}  # number of timepoints
                self._pix = {'type': str(image.dtype),  # string indicating pixel type
                             'bpp': self._BPP[str(image.dtype)],  # bytes per pixel
                             'spp': 1}  # samples per pixel
                # number of pixels to load at a time
                self._pix['chunk'] = self._MAX_BYTES / \
                                     (self._pix['spp'] * self._pix['bpp'])
                self._pix['interleaved'] = False
                self._metadata = self._minimal_xml()
            else:
                x = X if X else 1
                y = Y if Y else 1
                z = Z if Z else 1
                c = C if C else 1
                t = T if T else 1
                self._xyzct = {'X': x,  # image width
                               'Y': y,  # image height
                               'Z': z,  # image depth
                               'C': c,  # number of channels
                               'T': t}  # number of timepoints
                self._pix = {'type': 'uint8',  # string indicating pixel type
                             'bpp': self._BPP['uint8'],  # bytes per pixel
                             'spp': 1}  # samples per pixel
                # number of pixels to load at a time
                self._pix['chunk'] = self._MAX_BYTES / \
                                     (self._pix['spp'] * self._pix['bpp'])
                self._pix['interleaved'] = False
                self._metadata = self._minimal_xml()

            if file_path.endswith('.ome.tif'):
                ValueError(
                    "The file name that will be saved to must have extension .ome.tif")

        def _minimal_xml(self):
            """_minimal_xml Generates minimal xml for ome tiff initialization

            Returns:
                bioformats.omexml.OMEXML
            """
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            omexml = self.xml_metadata()
            omexml.image(0).Name = Path(self._file_path).name
            p = omexml.image(0).Pixels
            # assert isinstance(p, bioformats.OMEXML.Pixels)
            assert isinstance(p, self.xml_metadata().Pixels)
            p.SizeX = self._xyzct['X']
            p.SizeY = self._xyzct['Y']
            p.SizeZ = self._xyzct['Z']
            p.SizeC = self._xyzct['C']
            p.SizeT = self._xyzct['T']
            p.DimensionOrder = self.xml_metadata(var=1).DO_XYZCT
            p.PixelType = self._pix['type']
            if self._xyzct['C'] > 1:
                p.channel_count = self._xyzct['C']
            return omexml

        def _pack(self, fmt, *val):
            return struct.pack(self._byteorder + fmt, *val)



        def _init_writer(self):
            """_init_writer Initializes file writing.

            This method is called exactly once per object. Once it is
            called, all other methods of setting metadata will throw an
            error.

            """
            self._tags = []

            self._metadata.image().set_ID(Path(self._file_path).name)

            self.__writer = tifffile.TiffWriter(self._file_path, bigtiff=True, append=False)

            self._byteorder = self.__writer._byteorder

            self._datashape = (1, 1, 1) + (self.num_y(), self.num_x()) + (1,)
            self._datadtype = numpy.dtype(self._pix['type']).newbyteorder(self._byteorder)

            tagnoformat = self.__writer._tagnoformat
            valueformat = self.__writer._valueformat
            offsetformat = self.__writer._offsetformat
            offsetsize = self.__writer._offsetsize
            tagsize = self.__writer._tagsize

            # self._compresstag = tifffile.TIFF.COMPRESSION.NONE
            self._compresstag = tifffile.TIFF.COMPRESSION.ADOBE_DEFLATE

            # normalize data shape to 5D or 6D, depending on volume:
            #   (pages, planar_samples, height, width, contig_samples)
            self._samplesperpixel = 1
            self._bitspersample = self._datadtype.itemsize * 8

            self._tagbytecounts = 325  # TileByteCounts
            self._tagoffsets = 324  # TileOffsets

            def rational(arg, max_denominator=1000000):
                # return nominator and denominator from float or two integers
                from fractions import Fraction  # delayed import
                try:
                    f = Fraction.from_float(arg)
                except TypeError:
                    f = Fraction(arg[0], arg[1])
                f = f.limit_denominator(max_denominator)
                return f.numerator, f.denominator

            description = ''.join(['<?xml version="1.0" encoding="UTF-8"?>',
                                   '<!-- Warning: this comment is an OME-XML metadata block, which contains crucial dimensional parameters and other important metadata. ',
                                   'Please edit cautiously (if at all), and back up the original data before doing so. '
                                   'For more information, see the OME-TIFF web site: https://docs.openmicroscopy.org/latest/ome-model/ome-tiff/. -->',
                                   str(self._metadata).replace('ome:', '').replace(':ome', '')])
            self._addtag(270, 's', 0, description, writeonce=True)  # Description
            self._addtag(305, 's', 0, 'bfio 2.4.1')  # Software
            # addtag(306, 's', 0, datetime, writeonce=True)
            self._addtag(259, 'H', 1, self._compresstag)  # Compression
            self._addtag(256, 'I', 1, self._datashape[-2])  # ImageWidth
            self._addtag(257, 'I', 1, self._datashape[-3])  # ImageLength
            self._addtag(322, 'I', 1, self._TILE_SIZE)  # TileWidth
            self._addtag(323, 'I', 1, self._TILE_SIZE)  # TileLength

            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[self._datadtype.kind]
            self._addtag(339, 'H', self._samplesperpixel,
                         (sampleformat,) * self._samplesperpixel)

            self._addtag(277, 'H', 1, self._samplesperpixel)
            self._addtag(258, 'H', 1, self._bitspersample)

            subsampling = None
            maxsampling = 1
            # PhotometricInterpretation
            self._addtag(262, 'H', 1, tifffile.TIFF.PHOTOMETRIC.MINISBLACK.value)

            if self.physical_size_x()[0] is not None:
                self._addtag(282, '2I', 1,
                             rational(10000 / self.physical_size_x()[0] / 10000))  # XResolution in pixels/cm
                self._addtag(283, '2I', 1, rational(10000 / self.physical_size_y()[0]))  # YResolution in pixels/cm
                self._addtag(296, 'H', 1, 3)  # ResolutionUnit = cm
            else:
                self._addtag(282, '2I', 1, (1, 1))  # XResolution
                self._addtag(283, '2I', 1, (1, 1))  # YResolution
                self._addtag(296, 'H', 1, 1)  # ResolutionUnit

            def bytecount_format(bytecounts, size=offsetsize):
                # return small bytecount format
                if len(bytecounts) == 1:
                    return {4: 'I', 8: 'Q'}[size]
                bytecount = bytecounts[0] * 10
                if bytecount < 2 ** 16:
                    return 'H'
                if bytecount < 2 ** 32:
                    return 'I'
                if size == 4:
                    return 'I'
                return 'Q'

            # can save data array contiguous
            contiguous = False

            # one chunk per tile per plane
            self._tiles = (
                (self._datashape[3] + self._TILE_SIZE - 1) // self._TILE_SIZE,
                (self._datashape[4] + self._TILE_SIZE - 1) // self._TILE_SIZE,
            )

            self._numtiles = tifffile.product(self._tiles)
            self._databytecounts = [
                                       self._TILE_SIZE ** 2 * self._datadtype.itemsize] * self._numtiles
            self._bytecountformat = bytecount_format(self._databytecounts)
            self._addtag(self._tagbytecounts, self._bytecountformat, self._numtiles, self._databytecounts)
            self._addtag(self._tagoffsets, self.__writer._offsetformat, self._numtiles, [0] * self._numtiles)
            self._bytecountformat = self._bytecountformat * self._numtiles

            # the entries in an IFD must be sorted in ascending order by tag code
            self._tags = sorted(self._tags, key=lambda x: x[0])

        def _open_next_page(self):
            if self._current_page == None:
                self._current_page = 0
            else:
                self._current_page += 1

            if self._current_page == 1:
                for ind, tag in enumerate(self._tags):
                    if tag[0] == 270:
                        del self._tags[ind]
                        break
                description = "ImageJ=\nhyperstack=true\nimages={}\nchannels={}\nslices={}\nframes={}".format(
                    self.num_z(), self.num_c(), self.num_z(), self.num_t())
                self._addtag(270, 's', 0, description)  # Description
                self._tags = sorted(self._tags, key=lambda x: x[0])

            fh = self.__writer._fh

            self._ifdpos = fh.tell()

            tagnoformat = self.__writer._tagnoformat
            valueformat = self.__writer._valueformat
            offsetformat = self.__writer._offsetformat
            offsetsize = self.__writer._offsetsize
            tagsize = self.__writer._tagsize
            tagbytecounts = self._tagbytecounts
            tagoffsets = self._tagoffsets
            tags = self._tags

            if self._ifdpos % 2:
                # location of IFD must begin on a word boundary
                fh.write(b'\0')
                self._ifdpos += 1

            # update pointer at ifdoffset
            fh.seek(self.__writer._ifdoffset)
            fh.write(self._pack(offsetformat, self._ifdpos))
            fh.seek(self._ifdpos)

            # create IFD in memory, do not write to disk
            if self._current_page < 2:
                self._ifd = io.BytesIO()
                self._ifd.write(self._pack(tagnoformat, len(tags)))
                tagoffset = self._ifd.tell()
                self._ifd.write(b''.join(t[1] for t in tags))
                self._ifdoffset = self._ifd.tell()
                self._ifd.write(self._pack(offsetformat, 0))  # offset to next IFD
                # write tag values and patch offsets in ifdentries
                for tagindex, tag in enumerate(tags):
                    offset = tagoffset + tagindex * tagsize + offsetsize + 4
                    code = tag[0]
                    value = tag[2]
                    if value:
                        pos = self._ifd.tell()
                        if pos % 2:
                            # tag value is expected to begin on word boundary
                            self._ifd.write(b'\0')
                            pos += 1
                        self._ifd.seek(offset)
                        self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
                        self._ifd.seek(pos)
                        self._ifd.write(value)
                        if code == tagoffsets:
                            self._dataoffsetsoffset = offset, pos
                        elif code == tagbytecounts:
                            self._databytecountsoffset = offset, pos
                        elif code == 270 and value.endswith(b'\0\0\0\0'):
                            # image description buffer
                            self._descriptionoffset = self._ifdpos + pos
                            self._descriptionlenoffset = (
                                    self._ifdpos + tagoffset + tagindex * tagsize + 4)
                    elif code == tagoffsets:
                        self._dataoffsetsoffset = offset, None
                    elif code == tagbytecounts:
                        databytecountsoffset = offset, None
                self._ifdsize = self._ifd.tell()
                if self._ifdsize % 2:
                    self._ifd.write(b'\0')
                    self._ifdsize += 1

            self._databytecounts = [0 for _ in self._databytecounts]
            self._databyteoffsets = [0 for _ in self._databytecounts]

            # move to file position where data writing will begin
            # will write the tags later when the tile offsets are known
            fh.seek(self._ifdsize, 1)

            # write image data
            dataoffset = fh.tell()
            skip = (16 - (dataoffset % 16)) % 16
            fh.seek(skip, 1)
            dataoffset += skip

            self._page_open = True

        def _close_page(self):

            offsetformat = self.__writer._offsetformat
            bytecountformat = self._bytecountformat

            fh = self.__writer._fh

            # update strip/tile offsets
            offset, pos = self._dataoffsetsoffset
            self._ifd.seek(offset)

            self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
            self._ifd.seek(pos)
            for size in self._databyteoffsets:
                self._ifd.write(self._pack(offsetformat, size))

            # update strip/tile bytecounts
            offset, pos = self._databytecountsoffset
            self._ifd.seek(offset)
            if pos:
                self._ifd.write(self._pack(offsetformat, self._ifdpos + pos))
                self._ifd.seek(pos)
            self._ifd.write(self._pack(bytecountformat, *self._databytecounts))

            self._fhpos = fh.tell()
            fh.seek(self._ifdpos)
            fh.write(self._ifd.getvalue())
            fh.flush()
            fh.seek(self._fhpos)

            self.__writer._ifdoffset = self._ifdpos + self._ifdoffset

            # remove tags that should be written only once
            if self._current_page == 0:
                self._tags = [tag for tag in self._tags if not tag[-1]]

            self._page_open = False

        def _write_tiles(self, data, X, Y):

            assert len(X) == 2 and len(Y) == 2

            if X[0] % self._TILE_SIZE != 0 or Y[0] % self._TILE_SIZE != 0:
                print('X or Y positions are not on tile boundary, tile may save incorrectly')

            fh = self.__writer._fh

            x_tiles = list(range(X[0] // self._TILE_SIZE, 1 + (X[1] - 1) // self._TILE_SIZE))
            tiles = []
            for y in range(Y[0] // self._TILE_SIZE, 1 + (Y[1] - 1) // self._TILE_SIZE):
                tiles.extend([y * self._tiles[1] + x for x in x_tiles])

            tile_shape = ((Y[1] - Y[0] - 1 + self._TILE_SIZE) // self._TILE_SIZE,
                          (X[1] - X[0] - 1 + self._TILE_SIZE) // self._TILE_SIZE)

            data = data.reshape(1, 1, 1, data.shape[0], data.shape[1], 1)
            tileiter = tifffile.iter_tiles(data,
                                           (self._TILE_SIZE, self._TILE_SIZE), tile_shape)

            # define compress function
            compressor = tifffile.TIFF.COMPESSORS[self._compresstag]

            def compress(data, level=1):
                data = memoryview(data)
                cpr = zlib.compressobj(level,
                                       memLevel=9,
                                       wbits=15)
                output = b''.join([cpr.compress(data), cpr.flush()])
                return output

            offsetformat = self.__writer._offsetformat
            tagnoformat = self.__writer._tagnoformat

            # tileiter = [tile for tile in tileiter]
            tileiter = [copy.deepcopy(tile) for tile in tileiter]
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # with ThreadPoolExecutor(1) as executor:
                compressed_tiles = iter(executor.map(compress, tileiter))
            for tileindex in tiles:
                t = next(compressed_tiles)
                self._databyteoffsets[tileindex] = fh.tell()
                fh.write(t)
                self._databytecounts[tileindex] = len(t)

            return None


        def write_image(self, image, X=None, Y=None, Z=None, C=None, T=None):
            """write_image Write the image

            [extended_summary]

            Args:
                image (numpy.ndarray): Must be a 5-dimensional array, where coordinates
                    are [Y,X,Z,C,T]
                X (list, optional): A list of length 1 indicating the first x-pixel.
                    index that the image should be written to.
                    If None, writes the image starting at index 0. Defaults to None.
                Y (list, optional): A list of length 1 indicating the first y-pixel.
                    index that the image should be written to.
                    If None, writes the image starting at index 0. Defaults to None.
                Z (list, optional): A list of length 1 indicating the first z-pixel.
                    index that the image should be written to.
                    If None, writes the image starting at index 0. Defaults to None.
                C ([tuple,list], optional): tuple or list of values indicating channel
                    indices to write to. If None, writes to the full range.
                    Defaults to None.
                T ([tuple,list], optional): tuple or list of values indicating timepoint
                    indices to write to. If None, writes to the full range.
                    Defaults to None.

            Example:
                # Import javabridge and start the vm
                jutil.start_vm(class_path=bioformats.JARS)

                # Path to bioformats supported image
                image_path = 'AMD-CD_1_Maximumintensityprojection_c2.ome.tif'

                # Create the BioReader object
                br = BioReader(image_path)

                # Load the full image
                image = br.read_image()

                # Save the image, rename the channels
                bw = BioWriter("New_" + image_path,image=image)
                bw.channel_names(["Empty","ZO1","Empty"])
                bw.write_image(image)
                bw.close_image()

                # Only save one channel
                bw = BioWriter("ZO1_" + image_path,image=image)
                bw.num_c(1)
                bw.write_image(image[:,:,0,1,0].reshape((image.shape[0],image.shape[1],1,1,1)))
                bw.close_image()

                # List the channel names
                print(bw.channel_names())

                # Done executing program, so kill the vm. If the program needs to be run
                # again, a new interpreter will need to be spawned to start the vm.
                jutil.kill_vm()
            """

            assert len(image.shape) == 5, "Image must be 5-dimensional (x,y,z,c,t)."

            # Set pixel bounds
            if not X:
                X = [0]
            if not Y:
                Y = [0]
            if not Z:
                Z = [0]
            X.append(image.shape[1] + X[0])
            Y.append(image.shape[0] + Y[0])
            Z.append(image.shape[2] + Z[0])

            C = C if C else [c for c in range(0, self._xyzct['C'])]
            T = T if T else [t for t in range(0, self._xyzct['T'])]

            # Validate inputs
            self._val_xyz(X, 'X')
            self._val_xyz(Y, 'Y')
            self._val_xyz(Z, 'Z')
            self._val_ct(C, 'C')
            self._val_ct(T, 'T')

            if self._current_page != None and Z[0] < self._current_page:
                raise ValueError('Cannot write z layers below the current open page. (current page={},Z[0]={})'.format(
                    self._current_page, Z[0]))

            with self._lock:
                # Initialize the writer if it hasn't already been initialized
                if not self.__writer:
                    self._init_writer()

                # Do the work
                for zi, z in zip(range(0, Z[1] - Z[0]), range(Z[0], Z[1])):
                    while z != self._current_page:
                        if self._page_open:
                            self._close_page()
                        self._open_next_page()
                    self._write_tiles(image[..., zi, 0, 0], X, Y)

        def close_image(self):
            """close_image Close the image

            This function should be called when an image will no longer be written
            to. This allows for proper closing and organization of metadata.
            """
            if self._page_open:
                self._close_page()
            self._ifd.close()
            self.__writer._fh.close()

        def _put(self):
            """_put Method for saving image supertiles

            This method is intended to be run within a thread, and writes a
            chunk of the image according to the coordinates in a queue.

            Currently, this function will only write the first Z, C, and T
            positions regardless of what Z, C, and T coordinate are provided
            to the function. This function will need to be changed in then
            future to account for this.

            If the last value in X or Y is larger than the size of the
            image, then the image is cropped to the appropriate size.

            Input coordinates are read from the _supertile_index Queue object.

            Input data is stored in the _raw_buffer Queue object.

            A boolean value is returned to indicate the processed has finished.
            """

            I = self._raw_buffer.get()
            X, Y, Z, C, T = self._supertile_index.get()

            # Attach the jvm to the thread
            self._backend.attach()

            # Write the image
            self.write_image(I[:self.num_y(), :, numpy.newaxis, numpy.newaxis, numpy.newaxis],
                             X=[X[0]],
                             Y=[Y[0]])

            # Detach the jvm
            self._backend.detach()

            return True

        def _buffer_supertile(self, column_start, column_end):
            """_buffer_supertile Process the pixel buffer

            Give the column indices of the data to process, and determine if
            the buffer needs to be processed. This method checks to see if
            data in the buffer can be shifted into the _raw_buffer for writing.

            Args:
                column_start ([int]): First column index of data to be loaded
                column_end ([int]): Last column index of data to be loaded

            """

            # If the start column index is outside of the width of the supertile,
            # write the data and shift the pixels
            if column_start - self._tile_x_offset >= 1024:
                self._raw_buffer.put(numpy.copy(self._pixel_buffer[:, 0:1024]))
                self._pixel_buffer[:, 0:1024] = self._pixel_buffer[:, 1024:2048]
                self._pixel_buffer[:, 1024:] = 0
                self._tile_x_offset += 1024
                self._tile_last_column = numpy.argwhere((self._pixel_buffer == 0).all(axis=0))[0, 0]

        def _assemble_tiles(self, images, X, Y, Z, C, T):
            """_assemble_tiles Handle data untiling

            This function puts tiles into the _pixel_buffer, effectively
            untiling them.

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
                split_ind = 0
                while X[split_ind][0] - self._tile_x_offset < 1024:
                    split_ind += 1
            else:
                split_ind = len(X)

            # Untile the data
            num_rows = Y[0][1] - Y[0][0]
            num_cols = X[0][1] - X[0][0]
            num_tiles = len(X)

            for ind in range(split_ind):
                r_min = Y[ind][0] - self._tile_y_offset
                r_max = Y[ind][1] - self._tile_y_offset
                c_min = X[ind][0] - self._tile_x_offset
                c_max = X[ind][1] - self._tile_x_offset
                self._pixel_buffer[r_min:r_max, c_min:c_max] = images[ind, :, :, 0]

            if split_ind != num_tiles:
                self._buffer_supertile(X[-1][0], X[-1][1])
                for ind in range(split_ind, num_tiles):
                    r_min = Y[ind][0] - self._tile_y_offset
                    r_max = Y[ind][1] - self._tile_y_offset
                    c_min = X[ind][0] - self._tile_x_offset
                    c_max = X[ind][1] - self._tile_x_offset
                    self._pixel_buffer[r_min:r_max, c_min:c_max] = images[ind, :, :, 0]

            self._tile_last_column = c_max

            return True



        def writerate(self, tile_size, tile_stride=None, batch_size=None, channels=[0]):
            """writerate Image saving iterator

            This method is an iterator to save tiles of an image. This method
            buffers the saving of pixels asynchronously to quickly save
            images to disk. It is designed to work in complement to the
            BioReader.iterate method, and expects images to be fed into it in
            the exact same order as they would come out of that method.

            Data is sent to this iterator using the send() method once the
            iterator has been created. See the example for more information.

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
                Nothing

            Example:
                from bfio import BioReader, BioWriter
                import numpy as np

                # Create the BioReader
                br = bfio.BioReader('/path/to/file')

                # Create the BioWriter
                out_path = '/path/to/output'
                bw = bfio.BioWriter(out_path,metadata=br.read_metadata())

                # Get the batch size
                batch_size = br.maximum_batch_size(tile_size=[256,256],tile_stride=[256,256])
                readerator = br.iterate(tile_size=[256,256],tile_stride=[256,256],batch_size=batch_size)
                writerator = bw.writerate(tile_size=[256,256],tile_stride=[256,256],batch_size=batch_size)

                # Initialize the writerator
                next(writerator)

                # Load tiles of the imgae and save them
                for images,indices in readerator:
                    writerator.send(images)
                bw.close_image()

                # Verify images are the same
                original_image = br.read_image()
                bw = bfio.BioReader(out_path)
                saved_image = bw.read_image()

                print('Original and saved images are the same: {}'.format(numpy.array_equal(original_image,saved_image)))

            """

            # Enure that the number of tiles does not exceed the width of a supertile
            if batch_size == None:
                batch_size = min([32, self.maximum_batch_size(tile_size, tile_stride)])
            else:
                assert batch_size <= self.maximum_batch_size(tile_size, tile_stride), \
                    'batch_size must be less than or equal to {}.'.format(
                        self.maximum_batch_size(tile_size, tile_stride))

            # input error checking
            assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
            if tile_stride != None:
                assert len(tile_stride) == 2, "stride must be a list with 2 elements"
            else:
                stride = tile_size

            # calculate unpadding
            if not (set(tile_size) & set(tile_stride)):
                xyoffset = [int((tile_size[0] - tile_stride[0]) / 2), int((tile_size[1] - tile_stride[1]) / 2)]
                xypad = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]
                xypad[0] = xyoffset[0] + (tile_stride[0] - numpy.mod(self.num_y(), tile_stride[0])) / 2
                xypad[1] = xyoffset[1] + (tile_stride[1] - numpy.mod(self.num_x(), tile_stride[1])) / 2
                xypad = ((int(xyoffset[0]), int(2 * xypad[0] - xyoffset[0])),
                         (int(xyoffset[1]), int(2 * xypad[1] - xyoffset[1])))
            else:
                xyoffset = [0, 0]
                xypad = ((0, max([tile_size[0] - tile_stride[0], 0])),
                         (0, max([tile_size[1] - tile_stride[1], 0])))

            # determine supertile sizes
            y_tile_dim = int(numpy.ceil((self.num_y() - 1) / 1024))
            x_tile_dim = 1

            # Initialize the pixel buffer
            self._pixel_buffer = numpy.zeros((y_tile_dim * 1024 + tile_size[0], 2 * x_tile_dim * 1024 + tile_size[1]),
                                          dtype=self.pixel_type())
            self._tile_x_offset = 0
            self._tile_y_offset = 0

            # Generate the supertile saving order
            tiles = []
            y_tile_list = list(range(0, self.num_y(), 1024 * y_tile_dim))
            if y_tile_list[-1] != 1024 * y_tile_dim:
                y_tile_list.append(1024 * y_tile_dim)
            x_tile_list = list(range(0, self.num_x(), 1024 * x_tile_dim))
            if x_tile_list[-1] < self.num_x() + xypad[1][1]:
                x_tile_list.append(x_tile_list[-1] + 1024)

            for yi in range(len(y_tile_list) - 1):
                for xi in range(len(x_tile_list) - 1):
                    y_range = [y_tile_list[yi], y_tile_list[yi + 1]]
                    x_range = [x_tile_list[xi], x_tile_list[xi + 1]]
                    tiles.append([x_range, y_range])
                    self._supertile_index.put((x_range, y_range, [0, 1], [0], [0]))

            # Start the thread pool and start loading the first supertile
            thread_pool = ThreadPoolExecutor(self._max_workers)

            # generate the indices for each tile
            # TODO: modify this to grab more than just the first z-index
            X = []
            Y = []
            Z = []
            C = []
            T = []
            x_list = numpy.array(numpy.arange(0, self.num_x(), tile_stride[1]))
            y_list = numpy.array(numpy.arange(0, self.num_y(), tile_stride[0]))
            for x in x_list:
                for y in y_list:
                    X.append([x, x + tile_stride[1]])
                    Y.append([y, y + tile_stride[0]])
                    Z.append([0, 1])
                    C.append(channels)
                    T.append([0])

            # start looping through batches
            bn = 0
            while bn < len(X):
                # Wait for tiles to be sent
                images = yield

                # Wait for the last untiling thread to finish
                if self._tile_thread != None:
                    self._tile_thread.result()

                # start a thread to untile the data
                b = bn + images.shape[0]
                self._tile_thread = thread_pool.submit(self._assemble_tiles, images, X[bn:b], Y[bn:b], Z[bn:b], C[bn:b],
                                                       T[bn:b])
                bn = b

                # Save a supertile if a thread is available
                if self._raw_buffer.qsize() > 0:
                    if self._put_thread != None:
                        self._put_thread.result()
                    self._put_thread = thread_pool.submit(self._put)

            # Wait for the final untiling thread to finish
            self._tile_thread.result()

            # Put the remaining pixels in the buffer into the _raw_buffer
            self._raw_buffer.put(self._pixel_buffer[:, 0:self.num_x() - self._tile_x_offset])

            # Save the last supertile
            if self._put_thread != None:
                self._put_thread.result()  # wait for the previous thread to finish
            self._put()  # no need to use a thread for final save

            thread_pool.shutdown()

            yield