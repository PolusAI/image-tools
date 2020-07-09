"""
The two primary classes this code uses are bfio.BioReader and bfio.BioWriter:
bfio.BioReader will read any image that the Bioformats tool can read.
bfio.BioWriter will only save images as an ome tiled tiff.

Example usage is provided in the comments to each class.

Required packages:
javabridge (also requires jdk > 8)
python-bioformats
numpy

Note: Prior to reading or writing using these classes, the javabridge session
      must be started. This may be automated in the future.
"""
from tifffile import tifffile
from . import OmeXml as bioformats
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class BioReader():
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
    _file_path = None
    _metadata = None
    _xyzct = None
    _pix = None

    # Set constants for opening images
    _MAX_BYTES = 2**30
    _BPP = {'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float': 4,
            'double': 8}
    _TILE_SIZE = 2**10
    
    # Buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1) # only preload one supertile at a time
    _data_in_buffer = Queue(maxsize=1)
    _supertile_index = Queue()
    _pixel_buffer = None
    _fetch_thread = None
    _tile_thread= None
    _tile_x_offset = 0
    _tile_last_column = 0

    def __init__(self, file_path):
        """__init__ Initialize the a file for reading

        Prior to initializing the class, it is important to remember that
        the javabridge must be initialized. See the read_image() method
        for an example.

        Args:
            file_path (str): Path to file to read
        """
        self._file_path = file_path
        self._rdr = tifffile.TiffFile(file_path)
        self._metadata = self.read_metadata()

        # Information about image dimensions
        self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                       'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                       'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                       'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                       'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints

        # Information about data type and loading
        self._pix = {'type': self._metadata.image().Pixels.get_PixelType(),            # string indicating pixel type
                     'bpp': self._BPP[self._metadata.image().Pixels.get_PixelType()],  # bytes per pixel
                     'spp': self._metadata.image().Pixels.Channel().SamplesPerPixel}   # samples per pixel
        
        # number of pixels to load at a time
        self._pix['chunk'] = self._MAX_BYTES / \
            (self._pix['spp']*self._pix['bpp'])
            
        # determine if channels are interleaved
        self._pix['interleaved'] = self._pix['spp'] > 1

    def channel_names(self):
        """channel_names

        Returns:
            list: Strings indicating channel names
        """
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self._xyzct['C'])]

    def num_x(self):
        """num_x Width of image in pixels

        Returns:
            int: Width of image in pixels
        """
        return self._xyzct['X']

    def num_y(self):
        """num_y Height of image in pixels

        Returns:
            int: Height of image in pixels
        """
        return self._xyzct['Y']

    def num_z(self):
        """num_z Depth of image in pixels

        Returns:
            int: Depth of image in pixels
        """
        return self._xyzct['Z']

    def num_c(self):
        """num_c Number of channels in the image

        Returns:
            int: Number of channels
        """
        return self._xyzct['C']

    def num_t(self):
        """num_x Number of timepoints in an image

        Returns:
            int: Number of timepoints
        """
        return self._xyzct['T']

    def physical_size_x(self):
        """num_x Size of pixels in x-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeX, self._metadata.image(0).Pixels.PhysicalSizeXUnit)

    def physical_size_y(self):
        """num_y Size of pixels in y-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeY, self._metadata.image(0).Pixels.PhysicalSizeYUnit)

    def physical_size_z(self):
        """num_z Size of pixels in z-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeZ, self._metadata.image(0).Pixels.PhysicalSizeZUnit)

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
            assert len(xyz) == 2,\
                '{} must be a list or tuple of length 2.'.format(axis)
            assert xyz[0] >= 0,\
                '{}[0] must be greater than or equal to 0.'.format(axis)
            assert xyz[1] <= self._xyzct[axis],\
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
            assert np.any(np.greater(self._xyzct[axis], ct)),\
                'At least one of the {}-indices was larger than largest index ({}).'.format(axis, self._xyzct[axis]-1)
            assert np.any(np.less_equal(0, ct)),\
                'At least one of the {}-indices was less than 0.'.format(axis)
            assert len(ct) != 0,\
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
    
    def _chunk_indices(self,X,Y,Z):
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2
        
        offsets = []
        bytecounts = []
        
        x_tiles = np.arange(X[0]//1024,np.ceil(X[1]/1024),dtype=int)
        y_tile_stride = np.ceil(self.num_x()/1024)
        
        for z in range(Z[0],Z[1]):
            for y in range(Y[0],Y[1],1024):
                y_offset = int(y//1024 * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()
                
                offsets.extend([self._rdr.pages[z].dataoffsets[i] for i in ind])
                bytecounts.extend([self._rdr.pages[z].databytecounts[i] for i in ind])
                
        return offsets,bytecounts
    
    def _decode_tile(self, args):
        
        keyframe = self._keyframe
        out = self._out
        tile_shape = self._tile_shape
        
        w,l,d = self._tile_indices[args[1]]
        
        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)
        
        if segment is None:
            segment = keyframe.nodata
        
        out[l: l + shape[1],
            w: w + shape[2],
            d,0,0] = segment.squeeze()

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
        
        x_range = X[1]-X[0]
        y_range = Y[1]-Y[0]
        x_tile_shape = int(np.ceil((x_range)/1024))
        y_tile_shape = int(np.ceil((y_range)/1024))
        z_tile_shape = Z[1]-Z[0]
        
        self._tile_shape = (x_tile_shape,y_tile_shape,z_tile_shape)

        self._out = np.zeros([y_tile_shape*1024, x_tile_shape*1024, z_tile_shape,
                              1, 1], self._pix['type'])
        
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Do the work
        offsets,bytecounts = self._chunk_indices(X,Y,Z)
        self._tile_indices = []
        for z in range(0,Z[1]-Z[0]):
            for y in range(0,Y[1]-Y[0],1024):
                for x in range(0,X[1]-X[0],1024):
                    self._tile_indices.append((x,y,z))
        
        with ThreadPoolExecutor(self._keyframe.maxworkers) as executor:
            executor.map(self._decode_tile,fh.read_segments(offsets,bytecounts))
            
        xi = X[0] - 1024*X[0]//1024
        yi = Y[0] - 1024*Y[0]//1024 

        return self._out[yi:yi+y_range,xi:xi+x_range,...]
    
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
        X,Y,Z,C,T = self._supertile_index.get()
        
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
                x_min = 1024 * ((self.num_x()-1)//1024)
                reflect_x = True
            x_max = self.num_x()
            postpad_x = x_max - self.num_x()
        if y_max > self.num_y():
            y_max = self.num_y()
            postpad_y = y_max - self.num_y()
            
        # Read the image
        I = self.read_image([x_min,x_max],[y_min,y_max],[0,1],[0],[0]).squeeze()
        if reflect_x:
            I = np.fliplr(I)
        
        # Pad the image if needed
        if sum(1 for p in [prepad_x,prepad_y,postpad_x,postpad_y] if p != 0) > 0:
            I = np.pad(I,((prepad_y,postpad_y),(prepad_x,postpad_x)),mode='symmetric')

        # Store the data in the buffer
        self._raw_buffer.put(I)
        
        # Detach the jvm
        jutil.detach()
        
        return I
    
    def _buffer_supertile(self,column_start,column_end):
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
            self._pixel_buffer[:,0:x_max] = self._pixel_buffer[:,x_min:]
            self._pixel_buffer[:,x_max:] = 0
            self._tile_x_offset = column_start
            self._tile_last_column = np.argwhere((self._pixel_buffer==0).all(axis=0))[0,0]
        
        # Is there data in the buffer?
        if (self._supertile_index.qsize() > 0 or self._data_in_buffer.qsize() > 0):
            
            # If there is data in the _raw_buffer, return if there isn't room to load
            # it into the _pixel_buffer
            if self._pixel_buffer.shape[1] - self._tile_last_column < 1024:
                return
    
            I = self._raw_buffer.get()
            if self._tile_last_column == 0:
                self._pixel_buffer[:I.shape[0],:I.shape[1]] = I
                self._tile_last_column = I.shape[1]
                self._tile_x_offset = column_start
            else:
                self._pixel_buffer[:I.shape[0],self._tile_last_column:self._tile_last_column+I.shape[1]] = I
                self._tile_last_column += I.shape[1]
            
            self._data_in_buffer.get()

    def _get_tiles(self,X,Y,Z,C,T):
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
            
        self._buffer_supertile(X[0][0],X[0][1])
            
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
        images = np.zeros((num_tiles,num_rows,num_cols,1),dtype=self.pixel_type())
        
        for ind in range(split_ind):
            images[ind,:,:,0] = self._pixel_buffer[Y[ind][0]-self._tile_y_offset:Y[ind][1]-self._tile_y_offset,
                                                   X[ind][0]-self._tile_x_offset:X[ind][1]-self._tile_x_offset]
        
        if split_ind != num_tiles:
            self._buffer_supertile(X[-1][0],X[-1][1])
            for ind in range(split_ind,num_tiles):
                images[ind,:,:,0] = self._pixel_buffer[Y[ind][0]-self._tile_y_offset:Y[ind][1]-self._tile_y_offset,
                                                       X[ind][0]-self._tile_x_offset:X[ind][1]-self._tile_x_offset]
        
        return images
    
    def maximum_batch_size(self,tile_size,tile_stride=None):
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
            
        xyoffset = [(tile_size[0]-tile_stride[0])/2,(tile_size[1]-tile_stride[1])/2]
        
        num_tile_rows = int(np.ceil(self.num_y()/tile_stride[0]))
        num_tile_cols = (1024 - xyoffset[1])//tile_stride[1]
        if num_tile_cols == 0:
            num_tile_cols = 1
        
        return int(num_tile_cols*num_tile_rows)

    def iterate(self,tile_size,tile_stride=None,batch_size=None,channels=[0]):
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
            batch_size = min([32,self.maximum_batch_size(tile_size,tile_stride)])
        else:
            assert batch_size <= self.maximum_batch_size(tile_size,tile_stride),\
                'batch_size must be less than or equal to {}.'.format(self.maximum_batch_size(tile_size,tile_stride))
        
        # input error checking
        assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
        if tile_stride != None:
            assert len(tile_stride) == 2, "stride must be a list with 2 elements"
        else:
            stride = tile_size
            
        # calculate padding if needed
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [(tile_size[0]-tile_stride[0])/2,(tile_size[1]-tile_stride[1])/2]
            xypad = [(tile_size[0]-tile_stride[0])/2,(tile_size[1]-tile_stride[1])/2]
            xypad[0] = xyoffset[0] + (tile_stride[0] - np.mod(self.num_y(),tile_stride[0]))/2
            xypad[1] = xyoffset[1] + (tile_stride[1] - np.mod(self.num_x(),tile_stride[1]))/2
            xypad = ((int(xyoffset[0]),int(2*xypad[0] - xyoffset[0])),
                    (int(xyoffset[1]),int(2*xypad[1] - xyoffset[1])))
        else:
            xyoffset = [0,0]
            xypad = ((0,max([tile_size[0] - tile_stride[0],0])),
                     (0,max([tile_size[1] - tile_stride[1],0])))
            
        # determine supertile sizes
        y_tile_dim = int(np.ceil((self.num_y()-1)/1024))
        x_tile_dim = 1
            
        # Initialize the pixel buffer
        self._pixel_buffer = np.zeros((y_tile_dim*1024 + tile_size[0],2*x_tile_dim*1024 + tile_size[1]),dtype=self.pixel_type())
        self._tile_x_offset = -xypad[1][0]
        self._tile_y_offset = -xypad[0][0]
        
        # Generate the supertile loading order
        tiles = []
        y_tile_list = list(range(0,self.num_y()+xypad[0][1],1024*y_tile_dim))
        if y_tile_list[-1] != 1024*y_tile_dim:
            y_tile_list.append(1024*y_tile_dim)
        if y_tile_list[0] != xypad[0][0]:
            y_tile_list[0] = -xypad[0][0]
        x_tile_list = list(range(0,self.num_x()+xypad[1][1],1024*x_tile_dim))
        if x_tile_list[-1] < self.num_x()+xypad[1][1]:
            x_tile_list.append(x_tile_list[-1]+1024)
        if x_tile_list[0] != xypad[1][0]:
            x_tile_list[0] = -xypad[1][0]
        for yi in range(len(y_tile_list)-1):
            for xi in range(len(x_tile_list)-1):
                y_range = [y_tile_list[yi],y_tile_list[yi+1]]
                x_range = [x_tile_list[xi],x_tile_list[xi+1]]
                tiles.append([x_range,y_range])
                self._supertile_index.put((x_range,y_range,[0, 1],[0],[0]))
                
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
        x_list = np.array(np.arange(-xypad[1][0],self.num_x(),tile_stride[1]))
        y_list = np.array(np.arange(-xypad[0][0],self.num_y(),tile_stride[0]))
        for x in x_list:
            for y in y_list:
                X.append([x,x+tile_size[1]])
                Y.append([y,y+tile_size[0]])
                Z.append([0,1])
                C.append(channels)
                T.append([0])
        
        # Set up batches
        batches = list(range(0,len(X),batch_size))
        
        # get the first batch
        b = min([batch_size,len(X)])
        index = (X[0:b],Y[0:b],Z[0:b],C[0:b],T[0:b])
        images = self._get_tiles(*index)
        
        # start looping through batches
        for bn in batches[1:]:
            # start the thread to get the next batch
            b = min([bn+batch_size,len(X)])
            self._tile_thread = thread_pool.submit(self._get_tiles,X[bn:b],Y[bn:b],Z[bn:b],C[bn:b],T[bn:b])
            
            # Load another supertile if possible
            if self._supertile_index.qsize() > 0 and not self._fetch_thread.running():
                self._fetch_thread = thread_pool.submit(self._fetch)
            
            # return the curent set of images
            yield images, index
            
            # get the images from the thread
            index = (X[bn:b],Y[bn:b],Z[bn:b],C[bn:b],T[bn:b])
            images = self._tile_thread.result()
        
        thread_pool.shutdown()
        
        # return the last set of images
        yield images, index

class BioWriter():
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

    Methods:
        BioReader(file_path,image,X,Y,Z,C,T,metadata): See __init__ for details
        pixel_type(dtype): Gets/sets the pixel type (e.g. uint8)
        channel_names(cnames): Gets/sets the names of each channel
        num_x(X): Get/set number of pixels in the x-dimension (width)
        num_y(Y): Get/set number of pixels in the y-dimension (height)
        num_z(Z): Get/set number of pixels in the z-dimension (depth)
        num_c(C): Get/set number of channels in the image
        num_t(T): Get/set number of timepoints in the image
        physical_size_x(psize,units): Get/set the physical size of the x-dimension
        physical_size_y(psize,units): Get/set the physical size of the y-dimension
        physical_size_z(psize,units): Get/set the physical size of the z-dimension
        write_image(image,X,Y,Z,C,T): Writes the 5d image array to file
    """
    _file_path = None
    _metadata = None
    _xyzct = None
    _pix = None

    # Set constants for opening images
    _MAX_BYTES = 2**30
    _BPP = {'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float': 4,
            'double': 8}
    _TILE_SIZE = 2**10

    __writer = None
    
    # Buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1) # only preload one supertile at a time
    _supertile_index = Queue()
    _pixel_buffer = None
    _put_thread = None
    _tile_thread= None
    _tile_x_offset = 0
    _tile_last_column = 0

    def __init__(self, file_path, image=None,
                 X=None, Y=None, Z=None, C=None, T=None,
                 metadata=None):
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

        self._file_path = file_path

        if metadata:
            assert isinstance(metadata, bioformats.OMEXML)
            self._metadata = bioformats.OMEXML(str(metadata))
            self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                           'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                           'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                           'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                           'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints
            self._pix = {'type': self._metadata.image().Pixels.get_PixelType(),            # string indicating pixel type
                         # bytes per pixel
                         'bpp': self._BPP[self._metadata.image().Pixels.get_PixelType()],
                         'spp': 1}   # samples per pixel
            # number of pixels to load at a time
            self._pix['chunk'] = self._MAX_BYTES / \
                (self._pix['spp']*self._pix['bpp'])
            self._pix['interleaved'] = False
            self._metadata.image(0).Name = file_path
            self._metadata.image().Pixels.channel_count = self._xyzct['C']
            self._metadata.image().Pixels.DimensionOrder = bioformats.omexml.DO_XYZCT
        elif image != None:
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
            self._pix = {'type': str(image.dtype),            # string indicating pixel type
                         'bpp': self._BPP[str(image.dtype)],  # bytes per pixel
                         'spp': 1}                            # samples per pixel
            # number of pixels to load at a time
            self._pix['chunk'] = self._MAX_BYTES / \
                (self._pix['spp']*self._pix['bpp'])
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
            self._pix = {'type': 'uint8',            # string indicating pixel type
                         'bpp': self._BPP['uint8'],  # bytes per pixel
                         'spp': 1}                   # samples per pixel
            # number of pixels to load at a time
            self._pix['chunk'] = self._MAX_BYTES / \
                (self._pix['spp']*self._pix['bpp'])
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
        omexml = bioformats.OMEXML()
        omexml.image(0).Name = os.path.split(self._file_path)[1]
        p = omexml.image(0).Pixels
        assert isinstance(p, bioformats.OMEXML.Pixels)
        p.SizeX = self._xyzct['X']
        p.SizeY = self._xyzct['Y']
        p.SizeZ = self._xyzct['Z']
        p.SizeC = self._xyzct['C']
        p.SizeT = self._xyzct['T']
        p.DimensionOrder = bioformats.DO_XYZCT
        p.PixelType = self._pix['type']
        if self._xyzct['C'] > 1:
            p.channel_count = self._xyzct['C']
        return omexml

    def _init_writer(self):
        """_init_writer Initializes file writing.

        This method is called exactly once per object. Once it is
        called, all other methods of setting metadata will throw an
        error.
        
        """        
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

        w_klass = make_ome_tiff_writer_class()
        w_klass.setId = jutil.make_method('setId', '(Ljava/lang/String;)V',
                                          'Sets the current file name.')
        w_klass.saveBytesXYWH = jutil.make_method('saveBytes', '(I[BIIII)V',
                                                  'Saves the given byte array to the current file')
        w_klass.close = jutil.make_method('close', '()V',
                                          'Closes currently open file(s) and frees allocated memory.')
        w_klass.setTileSizeX = jutil.make_method('setTileSizeX', '(I)I',
                                                 'Set tile size width in pixels.')
        w_klass.setTileSizeY = jutil.make_method('setTileSizeY', '(I)I',
                                                 'Set tile size height in pixels.')
        w_klass.getTileSizeX = jutil.make_method('getTileSizeX', '()I',
                                                 'Set tile size width in pixels.')
        w_klass.getTileSizeY = jutil.make_method('getTileSizeY', '()I',
                                                 'Set tile size height in pixels.')
        w_klass.setBigTiff = jutil.make_method('setBigTiff', '(Z)V',
                                               'Set the BigTiff flag.')
        writer = w_klass()

        # Always set bigtiff flag. There have been some instances where bioformats does not
        # properly write images that are less than 2^32 if bigtiff is not set. So, just always
        # set it since it doesn't drastically alter file size.
        writer.setBigTiff(True)

        script = """
        importClass(Packages.loci.formats.services.OMEXMLService,
                    Packages.loci.common.services.ServiceFactory);
        var service = new ServiceFactory().getInstance(OMEXMLService);
        var metadata = service.createOMEXMLMetadata(xml);
        var writer = writer
        writer.setMetadataRetrieve(metadata);
        """
        jutil.run_script(script,
                         dict(path=self._file_path,
                              xml=self._metadata.to_xml().replace('<ome:', '<').replace('</ome:', '</'),
                              writer=writer))
        writer.setId(self._file_path)
        writer.setInterleaved(False)
        writer.setCompression("LZW")
        x = writer.setTileSizeX(self._TILE_SIZE)
        y = writer.setTileSizeY(self._TILE_SIZE)

        # number of pixels to load at a time
        self._pix['chunk'] = self._MAX_BYTES / \
            (self._pix['spp']*self._pix['bpp'])

        self.__writer = writer
        
    def _write_tiles(self, data=None, shape=None, dtype=None, returnoffset=False,
                     photometric=None, planarconfig=None, extrasamples=None, tile=None,
                     contiguous=True, align=16, truncate=False, rowsperstrip=None,
                     bitspersample=None, compress=None, predictor=False,
                     subsampling=None, colormap=None, description=None, datetime=None,
                     resolution=None, subfiletype=0, software='tifffile.py',
                     metadata={}, ijmetadata=None, extratags=()):
        """Write numpy array and tags to TIFF file.

        The data shape's last dimensions are assumed to be image depth,
        height (length), width, and samples.
        If a colormap is provided, the data's dtype must be uint8 or uint16
        and the data values are indices into the last dimension of the
        colormap.
        If 'shape' and 'dtype' are specified instead of 'data', an empty array
        is saved. This option cannot be used with compression or multiple
        tiles.
        If 'shape', 'dtype', and 'tile' are specified, 'data' must be a
        sequence or generator of all tiles in the image.
        Image data are written uncompressed in one strip per plane by default.
        Dimensions larger than 2 to 4 (depending on photometric mode, planar
        configuration, and SGI mode) are flattened and saved as separate pages.
        If the data size is zero, a single page with shape (0, 0) is saved.
        The SampleFormat tag is derived from the data type.

        Parameters
        ----------
        data : numpy.ndarray, sequence of numpy.ndarray, or None
            Input image or tiles.
            Tiles must match the shape specified in 'tile'.
        shape : tuple or None
            Shape of the empty or tiled array to save.
            Used only if 'data' is None or a sequence of tiles.
        dtype : numpy.dtype or None
            Datatype of the empty or tiled array to save.
            Used only if 'data' is None or a sequence of tiles.
        returnoffset : bool
            If True and the image data in the file is memory-mappable, return
            the offset and number of bytes of the image data in the file.
        photometric : {'MINISBLACK', 'MINISWHITE', 'RGB', 'PALETTE', 'CFA'}
            The color space of the image data according to TIFF.PHOTOMETRIC.
            By default, this setting is inferred from the data shape and the
            value of colormap.
            For CFA images, the CFARepeatPatternDim, CFAPattern, and other
            DNG or TIFF/EP tags must be specified in 'extratags' to produce a
            valid file.
        planarconfig : {'CONTIG', 'SEPARATE'}
            Specifies if samples are stored interleaved or in separate planes.
            By default, this setting is inferred from the data shape.
            If this parameter is set, extra samples are used to store grayscale
            images.
            'CONTIG': last dimension contains samples.
            'SEPARATE': third last dimension contains samples.
        extrasamples : tuple of {'UNSPECIFIED', 'ASSOCALPHA', 'UNASSALPHA'}
            Defines the interpretation of extra components in pixels.
            'UNSPECIFIED': no transparency information (default).
            'ASSOCALPHA': single, true transparency with pre-multiplied color.
            'UNASSALPHA': independent transparency masks.
        tile : tuple of int
            The shape ([depth,] length, width) of image tiles to write.
            If None (default), image data are written in strips.
            The tile length and width must be a multiple of 16.
            If a tile depth is provided, the SGI ImageDepth and TileDepth
            tags are used to save volume data.
            Tiles cannot be used to write contiguous series, except if tile
            matches the data shape.
            Few software can read the SGI format, e.g. MeVisLab.
        contiguous : bool
            If True (default) and the data and parameters are compatible with
            previous saved ones, the image data are stored contiguously after
            the previous one. In that case, 'photometric',
            'planarconfig', and 'rowsperstrip' are ignored. Metadata such as
            'description', 'metadata', 'datetime', and 'extratags' are written
            to the first page of a contiguous series only.
            If False, start a new contiguous series.
        align : int
            Byte boundary on which to align the image data in the file.
            Default 16. Use mmap.ALLOCATIONGRANULARITY for memory-mapped data.
            Following contiguous writes are not aligned.
        truncate : bool
            If True, only write the first page of a contiguous series if
            possible (uncompressed, contiguous, not tiled).
            Other TIFF readers will only be able to read part of the data.
        rowsperstrip : int
            The number of rows per strip. By default, strips will be ~64 KB
            if compression is enabled, else rowsperstrip is set to the image
            length. Bilevel images are always stored in one strip per plane.
        bitspersample : int
            Number of bits per sample. By default this is the number of
            bits of the data dtype. Different values for different samples
            are not supported. Unsigned integer data are packed into bytes
            as tightly as possible. Valid values are 1-8 for uint8, 9-16 for
            uint16 and 17-32 for uint32. Cannot be used with compression.
        compress : int, str, or (str, int)
            If 0 or None (default), data are written uncompressed.
            If 0-9, the level of ADOBE_DEFLATE compression.
            If a str, one of TIFF.COMPESSORS, e.g. 'LZMA' or 'ZSTD'.
            If a tuple, the first item is one of TIFF.COMPESSORS and the
            second item is the compression level.
            Compression cannot be used to write contiguous series.
            Compressors may require certain data shapes, types or value ranges.
            For example, JPEG requires grayscale or RGB(A), uint8 or 12-bit
            uint16. JPEG compression is experimental. JPEG markers and TIFF
            tags may not match.
        predictor : bool
            If True, apply horizontal differencing or floating-point predictor
            before compression. Predictors are disabled for 64-bit integers.
        subsampling : {(1, 1), (2, 1), (2, 2), (4, 1)}
            The horizontal and vertical subsampling factors used for the
            chrominance components of images. The default is (2, 2).
            Currently applies to JPEG compression of RGB images only.
            Images will be stored in YCbCr color space.
            Segment widths must be a multiple of 8 times the horizontal factor.
            Segment lengths and rowsperstrip must be a multiple of 8 times the
            vertical factor.
        colormap : numpy.ndarray
            RGB color values for the corresponding data value.
            Must be of shape (3, 2**(data.itemsize*8)) and dtype uint16.
        description : str
            The subject of the image. Must be 7-bit ASCII. Cannot be used with
            the ImageJ format.
            Saved with the first page of a contiguous series only.
        datetime : datetime, str, or bool
            Date and time of image creation in '%Y:%m:%d %H:%M:%S' format or
            datetime object. Else if True, the current date and time is used.
            Saved with the first page of a contiguous series only.
        resolution : (float, float[, str]) or ((int, int), (int, int)[, str])
            X and Y resolutions in pixels per resolution unit as float or
            rational numbers. A third, optional parameter specifies the
            resolution unit, which must be None (default for ImageJ),
            'INCH' (default), or 'CENTIMETER'.
        subfiletype : int
            Bitfield to indicate the kind of data. Set bit 0 if the image
            is a reduced-resolution version of another image. Set bit 1 if
            the image is part of a multi-page image. Set bit 2 if the image
            is transparency mask for another image (photometric must be
            MASK, SamplesPerPixel and BitsPerSample must be 1).
        software : str
            Name of the software used to create the file. Must be 7-bit ASCII.
            Saved with the first page of a contiguous series only.
        metadata : dict
            Additional metadata to be saved along with shape information
            in JSON or ImageJ formats in ImageDescription or IJMetadata tags.
            If None, do not write a ImageDescription tag with shape in JSON
            format.
            If ImageJ format, values for keys 'Info', 'Labels', 'Ranges',
            'LUTs', 'Plot', 'ROI', and 'Overlays' are saved in IJMetadata and
            IJMetadataByteCounts tags. Refer to the imagej_metadata_tag
            function for valid values.
            Strings must be 7-bit ASCII.
            Saved with the first page of a contiguous series only.
        extratags : sequence of tuples
            Additional tags as [(code, dtype, count, value, writeonce)].

            code : int
                The TIFF tag Id.
            dtype : str
                Data type of items in 'value' in Python struct format.
                One of B, s, H, I, 2I, b, h, i, 2i, f, d, Q, or q.
            count : int
                Number of data values. Not used for string or bytes values.
            value : sequence
                'Count' values compatible with 'dtype'.
                Bytes must contain count values of dtype packed as binary data.
            writeonce : bool
                If True, the tag is written to the first page of a contiguous
                series only.

        """
        # TODO: refactor this function
        fh = self.__writer._fh
        byteorder = self.__writer._byteorder
        compress = 'LZW'
        
        # iterable tiles
        if not hasattr(data, '__iter__'):
            raise ValueError('data is not iterable')
        tileiter = data
        datashape = shape
        datadtype = np.dtype(dtype).newbyteorder(byteorder)
        datadtypechar = datadtype.char

        datasize = tifffile.product(datashape) * datadtype.itemsize
        input_shape = datashape

        self.__writer._truncate = False

        tagnoformat = self.__writer._tagnoformat
        valueformat = self.__writer._valueformat
        offsetformat = self.__writer._offsetformat
        offsetsize = self.__writer._offsetsize
        tagsize = self.__writer._tagsize

        photometric = self.__writer.TIFF.PHOTOMETRIC.MINISBLACK
        planarconfig = self.__writer.TIFF.PLANARCONFIG.CONTIG
        extrasamples = self.__writer.TIFF.EXTRASAMPLES.UNSPECIFIED
        compress = 'LZW'
        compresstag = self.__writer.TIFF.COMPRESSION.LZW
        colormap = None
        tile = (1024,1024)
        volume = False

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, planar_samples, height, width, contig_samples)
        datashape = tifffile.reshape_nd(datashape, 2)
        shape = datashape
        ndim = len(datashape)
        samplesperpixel = 1
        extrasamples = 0

        packints = False
        if bilevel:
            if bitspersample is not None and bitspersample != 1:
                raise ValueError('bitspersample must be 1 for bilevel')
            bitspersample = 1
        elif compresstag == 7 and datadtype == 'uint16':
            if bitspersample is not None and bitspersample != 12:
                raise ValueError(
                    'bitspersample must be 12 for JPEG compressed uint16'
                )
            bitspersample = 12  # use 12-bit JPEG compression
        elif bitspersample is None:
            bitspersample = datadtype.itemsize * 8
        elif (
            (datadtype.kind != 'u' or datadtype.itemsize > 4) and
            bitspersample != datadtype.itemsize * 8
        ):
            raise ValueError('bitspersample does not match dtype')
        elif not (
            bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize] and
            bitspersample <= datadtype.itemsize * 8
        ):
            raise ValueError('bitspersample out of range of dtype')
        elif compress:
            if bitspersample != datadtype.itemsize * 8:
                raise ValueError(
                    'bitspersample cannot be used with compression'
                )
        elif bitspersample != datadtype.itemsize * 8:
            packints = True

        # normalize shape to 6D
        if len(datashape) not in (5, 6):
            raise RuntimeError('len(datashape) not in (5, 6)')
        if len(datashape) == 5:
            datashape = datashape[:2] + (1,) + datashape[2:]
        if datashape[0] == -1:
            s0 = product(datashape[1:])
            s0 = 1 if s0 == 0 else product(input_shape) // s0
            datashape = (s0,) + datashape[1:]
        shape = datashape
        try:
            data = data.reshape(shape)
        except AttributeError:
            pass  # data is None or iterator

        if photometric == PALETTE:
            if (
                samplesperpixel != 1
                or extrasamples
                or shape[1] != 1
                or shape[-1] != 1
            ):
                raise ValueError('invalid data shape for palette mode')

        if photometric == RGB and samplesperpixel == 2:
            raise ValueError('not a RGB image (samplesperpixel=2)')

        if bilevel:
            if compresstag not in (1, 32773):
                raise ValueError('cannot compress bilevel image')
            if tile:
                raise ValueError('cannot save tiled bilevel image')
            if photometric not in (0, 1, 4):
                raise ValueError(f'cannot save bilevel image as {photometric}')
            datashape = list(datashape)
            if datashape[-2] % 8:
                datashape[-2] = datashape[-2] // 8 + 1
            else:
                datashape[-2] = datashape[-2] // 8
            datashape = tuple(datashape)
            if datasize != product(datashape):
                raise RuntimeError('datasize != product(datashape)')
            if data is not None:
                data = numpy.packbits(data, axis=-2)
                if datashape[-2] != data.shape[-2]:
                    raise RuntimeError('datashape[-2] != data.shape[-2]')

        tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

        if tile:
            tagbytecounts = 325  # TileByteCounts
            tagoffsets = 324  # TileOffsets
        else:
            tagbytecounts = 279  # StripByteCounts
            tagoffsets = 273  # StripOffsets
        self._tagoffsets = tagoffsets

        def pack(fmt, *val):
            return struct.pack(byteorder + fmt, *val)

        def addtag(code, dtype, count, value, writeonce=False):
            # compute ifdentry & ifdvalue bytes from code, dtype, count, value
            # append (code, ifdentry, ifdvalue, writeonce) to tags list
            if not isinstance(code, int):
                code = TIFF.TAGS[code]
            try:
                tifftype = TIFF.DATA_DTYPES[dtype]
            except KeyError as exc:
                raise ValueError(f'unknown dtype {dtype}') from exc
            rawcount = count

            if dtype == 's':
                # strings; enforce 7-bit ASCII on unicode strings
                value = bytestr(value, 'ascii') + b'\0'
                count = rawcount = len(value)
                rawcount = value.find(b'\0\0')
                if rawcount < 0:
                    rawcount = count
                else:
                    rawcount += 1  # length of string without buffer
                value = (value,)
            elif isinstance(value, bytes):
                # packed binary data
                dtsize = struct.calcsize(dtype)
                if len(value) % dtsize:
                    raise ValueError('invalid packed binary data')
                count = len(value) // dtsize
            if len(dtype) > 1:
                count *= int(dtype[:-1])
                dtype = dtype[-1]
            ifdentry = [pack('HH', code, tifftype),
                        pack(offsetformat, rawcount)]
            ifdvalue = None
            if struct.calcsize(dtype) * count <= offsetsize:
                # value(s) can be written directly
                if isinstance(value, bytes):
                    ifdentry.append(pack(valueformat, value))
                elif count == 1:
                    if isinstance(value, (tuple, list, numpy.ndarray)):
                        value = value[0]
                    ifdentry.append(pack(valueformat, pack(dtype, value)))
                else:
                    ifdentry.append(pack(valueformat,
                                         pack(str(count) + dtype, *value)))
            else:
                # use offset to value(s)
                ifdentry.append(pack(offsetformat, 0))
                if isinstance(value, bytes):
                    ifdvalue = value
                elif isinstance(value, numpy.ndarray):
                    if value.size != count:
                        raise RuntimeError('value.size != count')
                    if value.dtype.char != dtype:
                        raise RuntimeError('value.dtype.char != dtype')
                    ifdvalue = value.tobytes()
                elif isinstance(value, (tuple, list)):
                    ifdvalue = pack(str(count) + dtype, *value)
                else:
                    ifdvalue = pack(dtype, value)
            tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

        def rational(arg, max_denominator=1000000):
            # return nominator and denominator from float or two integers
            from fractions import Fraction  # delayed import
            try:
                f = Fraction.from_float(arg)
            except TypeError:
                f = Fraction(arg[0], arg[1])
            f = f.limit_denominator(max_denominator)
            return f.numerator, f.denominator

        if description:
            # ImageDescription: user provided description
            addtag(270, 's', 0, description, writeonce=True)

        # write shape and metadata to ImageDescription
        self._metadata = {} if not metadata else metadata.copy()
        if self._imagej:
            if ijmetadata is None:
                ijmetadata = parse_kwargs(
                    self._metadata,
                    'Info', 'Labels', 'Ranges', 'LUTs', 'Plot', 'ROI',
                    'Overlays',
                    'info', 'labels', 'ranges', 'luts', 'plot', 'roi',
                    'overlays'
                )
            # TODO: activate DeprecationWarning and update tests
            # else:
            #     warnings.warn(
            #         "TiffWriter: the 'ijmetadata' argument is deprecated",
            #         DeprecationWarning
            #     )
            for t in imagej_metadata_tag(ijmetadata, byteorder):
                addtag(*t)
            description = imagej_description(
                input_shape,
                shape[-1] in (3, 4),
                self._colormap is not None,
                **self._metadata
            )
        elif metadata or metadata == {}:
            if self._truncate:
                self._metadata.update(truncated=True)
            description = json_description(input_shape, **self._metadata)
        # elif metadata is None and self._truncate:
        #     raise ValueError('cannot truncate without writing metadata')
        else:
            description = None
        if description:
            # add 64 bytes buffer
            # the image description might be updated later with the final shape
            description = description.encode('ascii')
            description += b'\0' * 64
            self._descriptionlen = len(description)
            addtag(270, 's', 0, description, writeonce=True)

        if software:
            addtag(305, 's', 0, software, writeonce=True)
        if datetime:
            if isinstance(datetime, str):
                if len(datetime) != 19 or datetime[16] != ':':
                    raise ValueError('invalid datetime string')
            else:
                try:
                    datetime = datetime.strftime('%Y:%m:%d %H:%M:%S')
                except AttributeError:
                    datetime = self._now().strftime('%Y:%m:%d %H:%M:%S')
            addtag(306, 's', 0, datetime, writeonce=True)
        addtag(259, 'H', 1, compresstag)  # Compression
        if compresstag == 34887:
            # LERC without additional compression
            addtag(50674, 'I', 2, (4, 0))
        if predictor:
            addtag(317, 'H', 1, predictortag)
        addtag(256, 'I', 1, shape[-2])  # ImageWidth
        addtag(257, 'I', 1, shape[-3])  # ImageLength
        if tile:
            addtag(322, 'I', 1, tile[-1])  # TileWidth
            addtag(323, 'I', 1, tile[-2])  # TileLength
            if volume:
                addtag(32997, 'I', 1, shape[-4])  # ImageDepth
                addtag(32998, 'I', 1, tile[0])  # TileDepth
        if subfiletype:
            addtag(254, 'I', 1, subfiletype)  # NewSubfileType
        if not bilevel and not datadtype.kind == 'u':
            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[datadtype.kind]
            addtag(339, 'H', samplesperpixel,
                   (sampleformat,) * samplesperpixel)
        if colormap is not None:
            addtag(320, 'H', colormap.size, colormap)
        addtag(277, 'H', 1, samplesperpixel)
        if bilevel:
            pass
        elif planarconfig and samplesperpixel > 1:
            addtag(284, 'H', 1, planarconfig.value)  # PlanarConfiguration
            addtag(258, 'H', samplesperpixel,
                   (bitspersample,) * samplesperpixel)  # BitsPerSample
        else:
            addtag(258, 'H', 1, bitspersample)
        if extrasamples:
            if extrasamples_ is not None:
                if extrasamples != len(extrasamples_):
                    raise ValueError('wrong number of extrasamples specified')
                addtag(338, 'H', extrasamples, extrasamples_)
            elif photometric == RGB and extrasamples == 1:
                # Unassociated alpha channel
                addtag(338, 'H', 1, 2)
            else:
                # Unspecified alpha channel
                addtag(338, 'H', extrasamples, (0,) * extrasamples)

        if compresstag == 7 and photometric == RGB and planarconfig == 1:
            # JPEG compression with subsampling. Store as YCbCr
            # TODO: use JPEGTables for multiple tiles or strips
            if subsampling is None:
                subsampling = (2, 2)
            elif subsampling not in ((1, 1), (2, 1), (2, 2), (4, 1)):
                raise ValueError('invalid subsampling factors')
            maxsampling = max(subsampling) * 8
            if tile and (tile[-1] % maxsampling or tile[-2] % maxsampling):
                raise ValueError(f'tile shape not a multiple of {maxsampling}')
            if extrasamples > 1:
                raise ValueError('JPEG subsampling requires RGB(A) images')
            addtag(530, 'H', 2, subsampling)  # YCbCrSubSampling
            addtag(262, 'H', 1, 6)  # PhotometricInterpretation YCBCR
            # ReferenceBlackWhite is required for YCBCR
            addtag(532, '2I', 6,
                   (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1))
        else:
            if subsampling not in (None, (1, 1)):
                log_warning('TiffWriter: cannot apply subsampling')
            subsampling = None
            maxsampling = 1
            # PhotometricInterpretation
            addtag(262, 'H', 1, photometric.value)
            # if compresstag == 7:
            #     addtag(530, 'H', 2, (1, 1))  # YCbCrSubSampling

        if resolution is not None:
            addtag(282, '2I', 1, rational(resolution[0]))  # XResolution
            addtag(283, '2I', 1, rational(resolution[1]))  # YResolution
            if len(resolution) > 2:
                unit = resolution[2]
                unit = 1 if unit is None else enumarg(TIFF.RESUNIT, unit)
            elif self._imagej:
                unit = 1
            else:
                unit = 2
            addtag(296, 'H', 1, unit)  # ResolutionUnit
        elif not self._imagej:
            addtag(282, '2I', 1, (1, 1))  # XResolution
            addtag(283, '2I', 1, (1, 1))  # YResolution
            addtag(296, 'H', 1, 1)  # ResolutionUnit

        def bytecount_format(bytecounts, compress=compress, size=offsetsize):
            # return small bytecount format
            if len(bytecounts) == 1:
                return {4: 'I', 8: 'Q'}[size]
            bytecount = bytecounts[0]
            if compress:
                bytecount = bytecount * 10
            if bytecount < 2**16:
                return 'H'
            if bytecount < 2**32:
                return 'I'
            if size == 4:
                return 'I'
            return 'Q'

        # can save data array contiguous
        contiguous = not compress and not packints
        if tile:
            # one chunk per tile per plane
            if len(tile) == 2:
                tiles = (
                    (shape[3] + tile[0] - 1) // tile[0],
                    (shape[4] + tile[1] - 1) // tile[1],
                )
                contiguous = (
                    contiguous and
                    shape[3] == tile[0] and
                    shape[4] == tile[1]
                )
            else:
                tiles = (
                    (shape[2] + tile[0] - 1) // tile[0],
                    (shape[3] + tile[1] - 1) // tile[1],
                    (shape[4] + tile[2] - 1) // tile[2],
                )
                contiguous = (
                    contiguous and
                    shape[2] == tile[0] and
                    shape[3] == tile[1] and
                    shape[4] == tile[2]
                )
            numtiles = product(tiles) * shape[1]
            databytecounts = [
                product(tile) * shape[-1] * datadtype.itemsize] * numtiles
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, numtiles, databytecounts)
            addtag(tagoffsets, offsetformat, numtiles, [0] * numtiles)
            bytecountformat = bytecountformat * numtiles
            if contiguous or tileiter is not None:
                pass
            else:
                tileiter = iter_tiles(data, tile, tiles)

        elif contiguous and (bilevel or rowsperstrip is None):
            # one strip per plane
            if bilevel:
                databytecounts = [product(datashape[2:])] * shape[1]
            else:
                databytecounts = [
                    product(datashape[2:]) * datadtype.itemsize] * shape[1]
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, shape[1], databytecounts)
            addtag(tagoffsets, offsetformat, shape[1], [0] * shape[1])
            addtag(278, 'I', 1, shape[-3])  # RowsPerStrip
            bytecountformat = bytecountformat * shape[1]
        else:
            # use rowsperstrip
            rowsize = product(shape[-2:]) * datadtype.itemsize
            if rowsperstrip is None:
                # compress ~64 KB chunks by default
                rowsperstrip = 65536 // rowsize if compress else shape[-3]
            if rowsperstrip < 1:
                rowsperstrip = maxsampling
            elif rowsperstrip > shape[-3]:
                rowsperstrip = shape[-3]
            elif subsampling and rowsperstrip % maxsampling:
                rowsperstrip = (math.ceil(rowsperstrip / maxsampling) *
                                maxsampling)
            addtag(278, 'I', 1, rowsperstrip)  # RowsPerStrip

            numstrips1 = (shape[-3] + rowsperstrip - 1) // rowsperstrip
            numstrips = numstrips1 * shape[1]
            # TODO: save bilevel data with rowsperstrip
            stripsize = rowsperstrip * rowsize
            databytecounts = [stripsize] * numstrips
            stripsize -= rowsize * (numstrips1 * rowsperstrip - shape[-3])
            for i in range(numstrips1 - 1, numstrips, numstrips1):
                databytecounts[i] = stripsize
            bytecountformat = bytecount_format(databytecounts)
            addtag(tagbytecounts, bytecountformat, numstrips, databytecounts)
            addtag(tagoffsets, offsetformat, numstrips, [0] * numstrips)
            bytecountformat = bytecountformat * numstrips

        if data is None and not contiguous:
            raise ValueError('cannot write non-contiguous empty file')

        # add extra tags from user
        for t in extratags:
            addtag(*t)

        # define compress function
        if compress:
            compressor = TIFF.COMPESSORS[compresstag]

            if subsampling:
                # JPEG with subsampling. Store RGB as YCbCr
                def compress(data, compressor=compressor, level=compresslevel,
                             subsampling=subsampling):
                    return compressor(data, level, subsampling=subsampling,
                                      colorspace=2, outcolorspace=3)

            elif predictor:
                def compress(data, predictor=predictor, compressor=compressor,
                             level=compresslevel):
                    data = predictor(data, axis=-2)
                    return compressor(data, level)

            elif compresslevel is not None:
                def compress(data, compressor=compressor, level=compresslevel):
                    return compressor(data, level)

            else:
                compress = compressor

        elif packints:

            def compress(data, bps=bitspersample):
                return packints_encode(data, bps, axis=-2)

        # TODO: check TIFFReadDirectoryCheckOrder warning in files containing
        #   multiple tags of same code
        # the entries in an IFD must be sorted in ascending order by tag code
        tags = sorted(tags, key=lambda x: x[0])

        fhpos = fh.tell()
        if (
            not (self._bigtiff or self._imagej or compress)
            and fhpos + datasize > 2**32 - 1
        ):
            raise ValueError('data too large for standard TIFF file')

        # if not compressed or multi-tiled, write the first IFD and then
        # all data contiguously; else, write all IFDs and data interleaved
        for pageindex in range(1 if contiguous else shape[0]):

            ifdpos = fhpos
            if ifdpos % 2:
                # location of IFD must begin on a word boundary
                fh.write(b'\0')
                ifdpos += 1

            # update pointer at ifdoffset
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, ifdpos))
            fh.seek(ifdpos)

            # create IFD in memory
            if pageindex < 2:
                ifd = io.BytesIO()
                ifd.write(pack(tagnoformat, len(tags)))
                tagoffset = ifd.tell()
                ifd.write(b''.join(t[1] for t in tags))
                ifdoffset = ifd.tell()
                ifd.write(pack(offsetformat, 0))  # offset to next IFD
                # write tag values and patch offsets in ifdentries
                for tagindex, tag in enumerate(tags):
                    offset = tagoffset + tagindex * tagsize + offsetsize + 4
                    code = tag[0]
                    value = tag[2]
                    if value:
                        pos = ifd.tell()
                        if pos % 2:
                            # tag value is expected to begin on word boundary
                            ifd.write(b'\0')
                            pos += 1
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))
                        ifd.seek(pos)
                        ifd.write(value)
                        if code == tagoffsets:
                            dataoffsetsoffset = offset, pos
                        elif code == tagbytecounts:
                            databytecountsoffset = offset, pos
                        elif code == 270 and value.endswith(b'\0\0\0\0'):
                            # image description buffer
                            self._descriptionoffset = ifdpos + pos
                            self._descriptionlenoffset = (
                                ifdpos + tagoffset + tagindex * tagsize + 4)
                    elif code == tagoffsets:
                        dataoffsetsoffset = offset, None
                    elif code == tagbytecounts:
                        databytecountsoffset = offset, None
                ifdsize = ifd.tell()
                if ifdsize % 2:
                    ifd.write(b'\0')
                    ifdsize += 1

            # write IFD later when strip/tile bytecounts and offsets are known
            fh.seek(ifdsize, 1)

            # write image data
            dataoffset = fh.tell()
            skip = (align - (dataoffset % align)) % align
            fh.seek(skip, 1)
            dataoffset += skip
            if contiguous:
                if data is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(data)
            elif tile:
                if data is None:
                    fh.write_empty(numtiles * databytecounts[0])
                else:
                    tilesize = product(tile) * shape[-1]
                    for tileindex in range(datashape[1] * product(tiles)):
                        chunk = next(tileiter)
                        if chunk is None:
                            if compress:
                                databytecounts[tileindex] = 0
                            else:
                                fh.write_empty(databytecounts[0])
                            continue
                        if chunk.size != tilesize:
                            raise ValueError(
                                f'invalid tile {chunk.shape!r} != {tile!r}')
                        if compress:
                            t = compress(chunk)
                            fh.write(t)
                            databytecounts[tileindex] = len(t)
                        else:
                            fh.write_array(chunk)
                            # fh.flush()
            elif compress:
                # write one strip per rowsperstrip
                if data.shape[2] != 1:
                    # not handling depth
                    raise RuntimeError('data.shape[2] != 1')
                numstrips = (shape[-3] + rowsperstrip - 1) // rowsperstrip
                stripindex = 0
                for plane in data[pageindex]:
                    for i in range(numstrips):
                        strip = plane[
                            0,
                            i * rowsperstrip: (i + 1) * rowsperstrip
                        ]
                        strip = compress(strip)
                        fh.write(strip)
                        databytecounts[stripindex] = len(strip)
                        stripindex += 1
            else:
                fh.write_array(data[pageindex])

            # update strip/tile offsets
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in databytecounts:
                    ifd.write(pack(offsetformat, offset))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if compress:
                # update strip/tile bytecounts
                offset, pos = databytecountsoffset
                ifd.seek(offset)
                if pos:
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    ifd.seek(pos)
                ifd.write(pack(bytecountformat, *databytecounts))

            fhpos = fh.tell()
            fh.seek(ifdpos)
            fh.write(ifd.getbuffer())
            fh.flush()
            fh.seek(fhpos)

            self._ifdoffset = ifdpos + ifdoffset

            # remove tags that should be written only once
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]

        self._shape = shape
        self._datashape = (1,) + input_shape
        self._datadtype = datadtype
        self._dataoffset = dataoffset
        self._databytecounts = databytecounts

        if contiguous:
            # write remaining IFDs/tags later
            self._tags = tags
            # return offset and size of image data
            if returnoffset:
                return dataoffset, sum(databytecounts)
        return None

    def pixel_type(self, dtype=None):
        """pixel_type Get/Set the pixel type

        If dtype is not defined, this function returns the current file
        setting. If dtype is defined, it must be one of the following:
        
        'uint8':  Unsigned 8-bit pixel type
        'int8':   Signed 8-bit pixel type
        'uint16': Unsigned 8-bit pixel type
        'int16':  Signed 16-bit pixel type
        'uint32': Unsigned 32-bit pixel type
        'int32':  Signed 32-bit pixel type
        'float':  IEEE single-precision pixel type
        'double': IEEE double precision pixel type
        
        Args:
            dtype (str, optional): Must be one of the above data types.
                If None, returns the value this is set to. Defaults to None.

        Returns:
            str: One of the above data types.
        """
        if dtype:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert dtype in self._BPP.keys(), "Invalid data type."
            self._metadata.image(0).Pixels.PixelType = dtype
            self._pix['type'] = dtype
        
        return self._metadata.image(0).Pixels.PixelType

    def channel_names(self, cnames=None):
        """channel_names Get/Set channel names

        If cnames is None, then this returns a list of channel names.
        If cnames is defined, it must be a list of strings with a list
        length equal to the number of channels.

        Args:
            cnames (list, optional): List of strings indicating channel names.
                If None, returns the value this is set to. Defaults to None.

        Returns:
            list: list of strings
        """
        if cnames:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert len(cnames) == self._xyzct['C'], "Number of names does not match number of channels."
            for i in range(0, len(cnames)):
                self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]
                
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self._xyzct['C'])]

    def num_x(self, X=None):
        """num_x Get/Set the number of pixels in the x-dimension

        If X is defined, it sets the number of pixels in the x-dimension
        (i.e. the image width). If X is None, return the image width.

        Args:
            X (int, optional): Width of image in pixels. Defaults to None.

        Returns:
            int: width of image in pixels
        """
        if X:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert X >= 1
            self._metadata.image(0).Pixels.SizeX = X
            self._xyzct['X'] = X
        
        return self._xyzct['X']

    def num_y(self, Y=None):
        """num_y Get/Set the number of pixels in the y-dimension

        If Y is defined, it sets the number of pixels in the y-dimension
        (i.e. the image height). If Y is None, return the image height.

        Args:
            Y (int, optional): height of image in pixels. Defaults to None.

        Returns:
            int: height of image in pixels
        """
        if Y:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert Y >= 1
            self._metadata.image(0).Pixels.SizeY = Y
            self._xyzct['Y'] = Y
        
        return self._xyzct['Y']

    def num_z(self, Z=None):
        """num_z Get/Set the number of pixels in the z-dimension

        If Z is defined, it sets the number of pixels in the z-dimension
        (i.e. the image depth). If Z is None, return the image depth.

        Args:
            Z (int, optional): Depth of image in pixels. Defaults to None.

        Returns:
            int: depth of image in pixels
        """
        if Z:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert Z >= 1
            self._metadata.image(0).Pixels.SizeZ = Z
            self._xyzct['Z'] = Z
        
        return self._xyzct['Z']

    def num_c(self, C=None):
        """num_c Get/Set the number of channels

        If C is defined, set the number of image channels. If C is
        None, return the number of channels.

        Args:
            C (int, optional): Number of image channels. Defaults to None.

        Returns:
            int: Number of image channels
        """
        if C:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert C >= 1
            self._metadata.image(0).Pixels.SizeC = C
            self._xyzct['C'] = C
        
        return self._xyzct['C']

    def num_t(self, T=None):
        """num_t Get/Set the number of timepoints

        If T is defined, set the number of image timepoints. If C is
        None, return the number of channels.

        Args:
            C (int, optional): Number of image channels. Defaults to None.

        Returns:
            int: Number of image channels
        """
        if T:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert T >= 1
            self._metadata.image(0).Pixels.SizeT = T
            self._xyzct['T'] = T
        
        return self._xyzct['T']

    def physical_size_x(self, psize=None, units=None):
        """physical_size_x Set the physical pixel size in the x-dimension
        
        If both psize and units are not None, then the physical pixel size
        is set in the x-dimension. If both psize and units are None, then
        the pixel size and units are returned.
        
        The availabe unit values are described on the OME website:
        https://docs.openmicroscopy.org/ome-model/6.5.0/developers/ome-units.html#length

        Args:
            psize (float, optional): Width of a pixel. Defaults to None.
            units (str, optional): Width units of a pixel. Defaults to None.

        Returns:
            float: Width of a pixel
            str: Units of the pixel
        """
        if psize != None and units != None:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            self._metadata.image(0).Pixels.PhysicalSizeX = psize
            self._metadata.image(0).Pixels.PhysicalSizeXUnit = units
        elif psize == None and units == None:
            pass
        else:
            raise ValueError('Both psize and units must be defined, or neither should be defined.')
        
        return (self._metadata.image(0).Pixels.PhysicalSizeX, self._metadata.image(0).Pixels.PhysicalSizeXUnit)

    def physical_size_y(self, psize=None, units=None):
        """physical_size_y Set the physical pixel size in the y-dimension
        
        If both psize and units are not None, then the physical pixel size
        is set in the y-dimension. If both psize and units are None, then
        the pixel size and units are returned.
        
        The availabe unit values are described on the OME website:
        https://docs.openmicroscopy.org/ome-model/6.5.0/developers/ome-units.html#length

        Args:
            psize (float, optional): Height of a pixel. Defaults to None.
            units (str, optional): Height units of a pixel. Defaults to None.

        Returns:
            float: Height of a pixel
            str: Units of the pixel
        """
        if psize != None and units != None:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            self._metadata.image(0).Pixels.PhysicalSizeY = psize
            self._metadata.image(0).Pixels.PhysicalSizeYUnit = units
        elif psize == None and units == None:
            pass
        else:
            raise ValueError('Both psize and units must be defined, or neither should be defined.')
        
        return (self._metadata.image(0).Pixels.PhysicalSizeY, self._metadata.image(0).Pixels.PhysicalSizeYUnit)

    def physical_size_z(self, psize=None, units=None):
        """physical_size_z Set the physical pixel size in the z-dimension
        
        If both psize and units are not None, then the physical pixel size
        is set in the z-dimension. If both psize and units are None, then
        the pixel size and units are returned.
        
        The availabe unit values are described on the OME website:
        https://docs.openmicroscopy.org/ome-model/6.5.0/developers/ome-units.html#length

        Args:
            psize (float, optional): Depth of a pixel. Defaults to None.
            units (str, optional): Depth units of a pixel. Defaults to None.

        Returns:
            float: Depth of a pixel
            str: Units of the pixel
        """
        if psize != None and units != None:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            self._metadata.image(0).Pixels.PhysicalSizeZ = psize
            self._metadata.image(0).Pixels.PhysicalSizeZUnit = units
        elif psize == None and units == None:
            pass
        else:
            raise ValueError('Both psize and units must be defined, or neither should be defined.')
        
        return (self._metadata.image(0).Pixels.PhysicalSizeZ, self._metadata.image(0).Pixels.PhysicalSizeZUnit)

    def _val_xyz(self, xyz, axis):
        """_val_xyz Utility function for validating image dimensions

        Args:
            xyz (int): Pixel value of x, y, or z dimension.
                If None, returns the maximum range of the dimension
            axis (str): Must be 'x', 'y', or 'z'

        Returns:
            list: list of ints indicating the first and last index in the dimension
        """
        if len(xyz) != 2:
            ValueError('{} must be a scalar.'.format(axis))
        elif xyz[0] < 0:
            ValueError(
                '{}[0] must be greater than or equal to 0.'.format(axis))
        elif xyz[1] > self._xyzct[axis]:
            ValueError('{}[1] cannot be greater than the maximum of the dimension ({}).'.format(
                axis, self._xyzct[axis]))

    def _val_ct(self, ct, axis):
        """_val_ct Utility function for validating image dimensions

        Args:
            ct (int,list): List of ints indicating the channels or timepoints to load
                If None, returns a list of ints
            axis (str): Must be 'c', 't'

        Returns:
            list: list of ints indicating the first and last index in the dimension
        """
        if np.any(np.greater_equal(self._xyzct[axis], ct)):
            ValueError(
                'At least one of the {}-indices was larger than largest index ({}).'.format(axis, self._xyzct[axis]-1))
        elif np.any(np.less(0, ct)):
            ValueError(
                'At least one of the {}-indices was less than 0.'.format(axis))
        elif len(ct) == 0:
            ValueError('At least one {}-index must be selected.'.format(axis))
        elif isinstance(ct, list):
            TypeError("The values for {} must be a list.".format(axis))

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
        X.append(image.shape[1]+X[0])
        Y.append(image.shape[0]+Y[0])
        Z.append(image.shape[2]+Z[0])

        C = C if C else [c for c in range(0, self._xyzct['C'])]
        T = T if T else [t for t in range(0, self._xyzct['T'])]

        # Validate inputs
        self._val_xyz(X, 'X')
        self._val_xyz(Y, 'Y')
        self._val_xyz(Z, 'Z')
        self._val_ct(C, 'C')
        self._val_ct(T, 'T')

        # Set options for file loading based on metadata
        # open in parts if more than max_bytes
        save_in_parts = (X[1]-X[0])*(Y[1]-Y[0]) > self._pix['chunk']

        # Initialize the writer if it hasn't already been initialized
        if not self.__writer:
            self._init_writer()

        # Do the work
        for ti, t in zip(range(0, len(T)), T):
            for zi, z in zip(range(0, Z[1]-Z[0]), range(Z[0], Z[1])):
                if not save_in_parts:
                    for ci, c in zip(range(0, len(C)), C):
                        index = z + self._xyzct['Z'] * c + \
                            self._xyzct['Z'] * self._xyzct['C'] * t
                        pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(
                            image[:, :, zi, ci, ti], self._pix['type'])
                        self.__writer.saveBytesXYWH(
                            index, pixel_buffer, X[0], Y[0], X[1]-X[0], Y[1]-Y[0])
                else:
                    for ci, c in zip(range(0, len(C)), C):
                        index = z + self._xyzct['Z'] * c + \
                            self._xyzct['Z'] * self._xyzct['C'] * t
                        for x in range(X[0], X[1], self._TILE_SIZE):
                            x_max = np.min([x+self._TILE_SIZE, X[1]])
                            x_range = x_max - x
                            for y in range(Y[0], Y[1], self._TILE_SIZE):
                                y_max = np.min([y+self._TILE_SIZE, Y[1]])
                                y_range = y_max - y

                                pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(
                                    image[y-Y[0]:y_max-Y[0], x-X[0]:x_max-X[0], zi, ci, ti], self._pix['type'])
                                self.__writer.saveBytesXYWH(
                                    index, pixel_buffer, x, y, x_range, y_range)

    def close_image(self):
        """close_image Close the image

        This function should be called when an image will no longer be written
        to. This allows for proper closing and organization of metadata.
        """
        self.__writer.close()

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
        X,Y,Z,C,T = self._supertile_index.get()
        
        # Attach the jvm to the thread
        jutil.attach()
            
        # Write the image
        self.write_image(I[:self.num_y(),:,np.newaxis,np.newaxis,np.newaxis],
                         X=[X[0]],
                         Y=[Y[0]])
        
        # Detach the jvm
        jutil.detach()
        
        return True
    
    def _buffer_supertile(self,column_start,column_end):
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
            self._raw_buffer.put(np.copy(self._pixel_buffer[:,0:1024]))
            self._pixel_buffer[:,0:1024] = self._pixel_buffer[:,1024:2048]
            self._pixel_buffer[:,1024:] = 0
            self._tile_x_offset += 1024
            self._tile_last_column = np.argwhere((self._pixel_buffer==0).all(axis=0))[0,0]

    def _assemble_tiles(self,images,X,Y,Z,C,T):
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
        self._buffer_supertile(X[0][0],X[0][1])
            
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
            r_min = Y[ind][0]-self._tile_y_offset
            r_max = Y[ind][1]-self._tile_y_offset
            c_min = X[ind][0]-self._tile_x_offset
            c_max = X[ind][1]-self._tile_x_offset
            self._pixel_buffer[r_min:r_max,c_min:c_max] = images[ind,:,:,0]
        
        if split_ind != num_tiles:
            self._buffer_supertile(X[-1][0],X[-1][1])
            for ind in range(split_ind,num_tiles):
                r_min = Y[ind][0]-self._tile_y_offset
                r_max = Y[ind][1]-self._tile_y_offset
                c_min = X[ind][0]-self._tile_x_offset
                c_max = X[ind][1]-self._tile_x_offset
                self._pixel_buffer[r_min:r_max,c_min:c_max] = images[ind,:,:,0]
                
        self._tile_last_column = c_max
        
        return True
    
    def maximum_batch_size(self,tile_size,tile_stride=None):
        """maximum_batch_size Maximum allowable batch size for tiling

        The pixel buffer only loads at most two supertiles at a time. If the batch
        size is too large, then the tiling function will attempt to create more
        tiles than what the buffer holds. To prevent the tiling function from doing
        this, there is a limit on the number of tiles that can be retrieved in a
        single call. This function determines what the largest number of saveable
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
            
        xyoffset = [(tile_size[0]-tile_stride[0])/2,(tile_size[1]-tile_stride[1])/2]
        
        num_tile_rows = int(np.ceil(self.num_y()/tile_stride[0]))
        num_tile_cols = (1024 - xyoffset[1])//tile_stride[1]
        if num_tile_cols == 0:
            num_tile_cols = 1
        
        return int(num_tile_cols*num_tile_rows)

    def writerate(self,tile_size,tile_stride=None,batch_size=None,channels=[0]):
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
            
            print('Original and saved images are the same: {}'.format(np.array_equal(original_image,saved_image)))
                
        """
        
        # Enure that the number of tiles does not exceed the width of a supertile
        if batch_size == None:
            batch_size = min([32,self.maximum_batch_size(tile_size,tile_stride)])
        else:
            assert batch_size <= self.maximum_batch_size(tile_size,tile_stride),\
                'batch_size must be less than or equal to {}.'.format(self.maximum_batch_size(tile_size,tile_stride))
        
        # input error checking
        assert len(tile_size) == 2, "tile_size must be a list with 2 elements"
        if tile_stride != None:
            assert len(tile_stride) == 2, "stride must be a list with 2 elements"
        else:
            stride = tile_size
            
        # calculate unpadding
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [int((tile_size[0]-tile_stride[0])/2),int((tile_size[1]-tile_stride[1])/2)]
            xypad = [(tile_size[0]-tile_stride[0])/2,(tile_size[1]-tile_stride[1])/2]
            xypad[0] = xyoffset[0] + (tile_stride[0] - np.mod(self.num_y(),tile_stride[0]))/2
            xypad[1] = xyoffset[1] + (tile_stride[1] - np.mod(self.num_x(),tile_stride[1]))/2
            xypad = ((int(xyoffset[0]),int(2*xypad[0] - xyoffset[0])),
                    (int(xyoffset[1]),int(2*xypad[1] - xyoffset[1])))
        else:
            xyoffset = [0,0]
            xypad = ((0,max([tile_size[0] - tile_stride[0],0])),
                     (0,max([tile_size[1] - tile_stride[1],0])))
            
        # determine supertile sizes
        y_tile_dim = int(np.ceil((self.num_y()-1)/1024))
        x_tile_dim = 1
            
        # Initialize the pixel buffer
        self._pixel_buffer = np.zeros((y_tile_dim*1024 + tile_size[0],2*x_tile_dim*1024 + tile_size[1]),dtype=self.pixel_type())
        self._tile_x_offset = 0
        self._tile_y_offset = 0
        
        # Generate the supertile saving order
        tiles = []
        y_tile_list = list(range(0,self.num_y(),1024*y_tile_dim))
        if y_tile_list[-1] != 1024*y_tile_dim:
            y_tile_list.append(1024*y_tile_dim)
        x_tile_list = list(range(0,self.num_x(),1024*x_tile_dim))
        if x_tile_list[-1] < self.num_x()+xypad[1][1]:
            x_tile_list.append(x_tile_list[-1]+1024)
            
        for yi in range(len(y_tile_list)-1):
            for xi in range(len(x_tile_list)-1):
                y_range = [y_tile_list[yi],y_tile_list[yi+1]]
                x_range = [x_tile_list[xi],x_tile_list[xi+1]]
                tiles.append([x_range,y_range])
                self._supertile_index.put((x_range,y_range,[0, 1],[0],[0]))
                
        # Start the thread pool and start loading the first supertile
        thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = np.array(np.arange(0,self.num_x(),tile_stride[1]))
        y_list = np.array(np.arange(0,self.num_y(),tile_stride[0]))
        for x in x_list:
            for y in y_list:
                X.append([x,x+tile_stride[1]])
                Y.append([y,y+tile_stride[0]])
                Z.append([0,1])
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
            self._tile_thread = thread_pool.submit(self._assemble_tiles,images,X[bn:b],Y[bn:b],Z[bn:b],C[bn:b],T[bn:b])
            bn = b
            
            # Save a supertile if a thread is available
            if self._raw_buffer.qsize() > 0:
                if self._put_thread!=None:
                    self._put_thread.result()
                self._put_thread = thread_pool.submit(self._put)
            
        # Wait for the final untiling thread to finish
        self._tile_thread.result()
        
        # Put the remaining pixels in the buffer into the _raw_buffer
        self._raw_buffer.put(self._pixel_buffer[:,0:self.num_x()-self._tile_x_offset])
        
        # Save the last supertile
        if self._put_thread != None:
            self._put_thread.result() # wait for the previous thread to finish
        self._put() # no need to use a thread for final save
        
        thread_pool.shutdown()
        
        yield