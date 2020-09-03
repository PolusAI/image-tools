import abc, threading
import numpy as np
from queue import Queue
import  multiprocessing

class BioBase(metaclass=abc.ABCMeta) :

    """ Abstract class for reading/writing OME tiled tiff images

    Methods:
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

    """
    # Set constants for reading/writing images
    _MAX_BYTES = 2 ** 30
    _BPP = {'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float': 4,
            'double': 8}
    _TILE_SIZE = 2 ** 10
    
    # protected attribute to hold metadata
    _metadata = None
    
    # Buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1)  # only preload one supertile at a time
    _data_in_buffer = Queue(maxsize=1)
    _supertile_index = Queue()
    _pixel_buffer = None
    _fetch_thread = None
    _tile_thread = None
    _tile_x_offset = 0
    _tile_last_column = 0
    
    # Whether the object is read only
    __read_only = True

    def __init__(self, file_path,max_workers=None):
        """__init__ Initialize BioBase object

         Args:
            file_path (str): Path to output file
            max_workers(int) : Number of threads to be used
        """
        self._file_path = file_path
        self._metadata = _metadata

        self._xyzct = None
        self._pix = None
        self.__read_only = True
        self._max_workers = max_workers if max_workers != None else max([multiprocessing.cpu_count()//2,1])
        
        self._lock =  threading.Lock()
    
    @property
    def read_only(self):
        """read_only Returns true is object is ready only

        Returns:
            bool: True if object is read only
        """
        return self.__read_only

    @read_only.setter
    def read_only(self, bool):
        raise AttributeError('read_only attribute is read-only.')
    
    """ Methods to get/set image dimensions in pixels """
    
    def __xyzct_setter(self,dimension,value):
        assert not self.__read_only, "num_{} is read-only.".format(dimension.lower())
        assert value >= 1, "{} must be >= 0".format(dimension.upper())
        setattr(self._metadata.image(0).Pixels,'Size{}'.format(value.upper()),value)
        self._xyzct[dimension.upper()] = value

    @property
    def X(self):
        """num_x Width of image in pixels"""
        return self._xyzct['X']
    
    @num_x.setter
    def num_x(self,X):
        self.__xyzct_setter('X',X)
        
    @property
    def num_y(self):
        """num_y Height of image in pixels"""
        return self._xyzct['Y']
    
    @num_y.setter
    def num_y(self,Y):
        self.__xyzct_setter('Y',Y)

    @property
    def num_z(self):
        """num_z Depth of image in pixels"""
        return self._xyzct['Z']
    
    @num_z.setter
    def num_z(self,Z):
        self.__xyzct_setter('Z',Z)

    @property
    def num_c(self):
        """num_c Number of channels in the image"""
        return self._xyzct['C']
    
    @num_c.setter
    def num_c(self,C):
        self.__xyzct_setter('C',C)

    @property
    def num_t(self,T=None):
        """num_x Number of timepoints in an image"""
        return self._xyzct['C']
    
    @num_t.setter
    def num_t(self,T):
        self.__xyzct_setter('T',T)
    
    """ Methods to get/set dimensions properties, i.e. pixel size, channels names """
    @property
    def channel_names(self):
        """channel_names getter/setter

        Returns:
            list: Strings indicating channel names
        """
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self._xyzct['C'])]
        
    @channel_names.setter
    def channel_names(self,cnames):
        assert not self.__read_only, "channel_names is read-only."
        assert len(cnames) == self._xyzct['C'], "Number of names does not match number of channels."
        for i in range(0, len(cnames)):
            self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]
            
    def __physical_size(self,dimension,psize,units):
        if psize != None and units != None:
            assert not self.__read_only, "physical_size_{} is read-only.".format(dimension.lower())
            setattr(self._metadata.image(0).Pixels,'PhysicalSize{}'.format(dimension.upper()),psize)
            setattr(self._metadata.image(0).Pixels,'PhysicalSize{}Unit'.format(dimension.upper()),units)

    @property
    def physical_size_x(self):
        """physical_size_x Size of pixels in x-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeX, self._metadata.image(0).Pixels.PhysicalSizeXUnit)

    @physical_size_x.setter
    def physical_size_x(self,psize,units):
        self.__physical_size(self,'X',psize,units)
        
    @property
    def physical_size_y(self):
        """physical_size_y Size of pixels in y-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeY, self._metadata.image(0).Pixels.PhysicalSizeYUnit)

    @physical_size_y.setter
    def physical_size_y(self,psize,units):
        self.__physical_size(self,'Y',psize,units)
        
    @property
    def physical_size_z(self):
        """physical_size_z Size of pixels in z-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        return (self._metadata.image(0).Pixels.PhysicalSizeZ, self._metadata.image(0).Pixels.PhysicalSizeZUnit)

    @physical_size_z.setter
    def physical_size_z(self,psize,units):
        self.__physical_size(self,'Z',psize,units)

    """ Validation methods """
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
            xyz = [0,self._xyzct[axis]]
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

    @property
    def pixel_type(self,dtype=None):
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
    
    @pixel_type.setter
    def pixel_type(self,dtype):
        assert not self.__read_only, "The image has started to be written. To modify the xml again, reinitialize."
        assert dtype in self._BPP.keys(), "Invalid data type."
        self._metadata.image(0).Pixels.PixelType = dtype
        self._pix['type'] = dtype

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
