import abc, threading, logging
import numpy as np
from queue import Queue
import multiprocessing, typing
from pathlib import Path

class BioBase(object,metaclass=abc.ABCMeta) :

    """ Abstract class for reading/writing OME tiled tiff images
    
    Args:
        file_path: Path to the file to be read/written
        max_workers: Number of threads to read/write data
        backend: Backend to use, must be 'python' or 'java'
    
    Attributes:
        dtype: Gets/sets the pixel type (e.g. uint8)
        channel_names: Gets/sets the names of each channel
        samples_per_pixel: Numbers of numbers per pixel location
        bytes_per_pixel: Number of bytes per pixel
        x: Get/set number of pixels in the x-dimension (width)
        y: Get/set number of pixels in the y-dimension (height)
        z: Get/set number of pixels in the z-dimension (depth)
        c: Get/set number of channels in the image
        t: Get/set number of timepoints in the image
        physical_size_x: Get/set the physical size of the x-dimension
        physical_size_y: Get/set the physical size of the y-dimension
        physical_size_z: Get/set the physical size of the z-dimension
        metadata: OmeXml object for the image
        cnames: Same as channel_names
        spp: Same as samples_per_pixel
        bpp: Same as bytes_per_pixel
        X: Same as x attribute
        Y: Same as y attribute
        Z: Same as z attribute
        C: Same as c attribute
        T: same as t attribute
        ps_x: Same as physical_size_x
        ps_y: Same as physical_size_y
        ps_z: Same as physical_size_z

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
    _CHUNK_SIZE = None
    
    # protected backend object for interfacing with the file on disk
    _backend = None
    
    # protected attribute to hold metadata
    _metadata = None
    
    # protected buffering variables for iterating over an image
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
    
    def __init__(self,
                 file_path: typing.Union[str,Path],
                 max_workers: typing.Optional[int] = None,
                 backend: typing.Optional[str] ='python'):
        """__init__ Initialize BioBase object

         Args:
            file_path (str): Path to output file
            max_workers (int,optional): Number of threads to be used.
                Default is None.
            backend (str,optional): Backend to use, must be 'python' or 'java'.
                Default is 'python'.
        """
        if isinstance(file_path,str):
            file_path = Path(file_path)
        self._file_path = file_path

        self.max_workers = max_workers if max_workers != None else max([multiprocessing.cpu_count()//2,1])
        
        if backend.lower() not in ['python','java']:
            raise ValueError('Keyword argument backend must be one of ["python","java"]')
        
        self._backend_name = backend.lower()
        
        self._lock =  threading.Lock()
    
    @property
    def read_only(self):
        """read_only Returns true is object is ready only

        Returns:
            bool: True if object is read only
        """
        return self.__read_only

    @read_only.setter
    def read_only(self):
        raise AttributeError('read_only attribute is read-only.')
    
    def __getattribute__(self,name):
        # Get image dimensions using num_x, x, or X
        if (name.startswith('num_') and name[-1] in 'xyzct'):
            raise PendingDeprecationWarning(('num_{0} will be deprecated in bfio version 2.1.0.\n' + \
                                             'Currently, num_{0} can only be used to get the dimension.' + \
                                             '\tTo get/set the image dimension, ' + \
                                             'use the new get/set attribute BioReader.{0}').format(name[-1]))
            return getattr(self._metadata.image().Pixels,'get_Size{}'.format(name[-1].upper()))
        if (len(name)==1 and name.lower() in 'xyzct'):
            return getattr(self._metadata.image().Pixels,'get_Size{}'.format(name.upper()))()
        else:
            return object.__getattribute__(self,name)
        
    def __setattr__(self,name,args):
        # Set image dimensions, for example, using x or X
        if len(name)==1 and name.lower() in 'xyzct':
            self.__xyzct_setter(self,name,*args)
        else:
            object.__setattr__(self,name,args)
    
    def __xyzct_setter(self,dimension,value):
        assert not self.__read_only, "{} is read-only.".format(dimension.lower())
        assert value >= 1, "{} must be >= 0".format(dimension.upper())
        setattr(self._metadata.image(0).Pixels,'Size{}'.format(value.upper()),value)
    
    """ ------------------------------ """
    """ -Get/Set Dimension Properties- """
    """ ------------------------------ """
    @property
    def channel_names(self):
        """channel_names getter/setter

        Returns:
            list: Strings indicating channel names
        """
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self.C)]
        
    @channel_names.setter
    def channel_names(self,cnames: list):
        assert not self.__read_only, "channel_names is read-only."
        assert len(cnames) == self.C, "Number of names does not match number of channels."
        for i in range(0, len(cnames)):
            self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]
            
    @property
    def cnames(self):
        """cnames Same as channel_names"""
        return self.channel_names
        
    @channel_names.setter
    def cnames(self,cnames: list):
        self.channel_names = cnames
            
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
    def ps_x(self):
        """px_x Same as physical_size_x"""
        return self.physical_size_x

    @ps_x.setter
    def ps_x(self,psize,units):
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
    def ps_y(self):
        """px_y Same as physical_size_y"""
        return self.physical_size_y

    @ps_y.setter
    def ps_y(self,psize,units):
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
        
    @property
    def ps_z(self):
        """px_z Same as physical_size_z"""
        return self.physical_size_x

    @ps_z.setter
    def ps_z(self,psize,units):
        self.__physical_size(self,'Z',psize,units)

    """ -------------------- """
    """ -Validation methods- """
    """ -------------------- """
    
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
        
        if xyz == None:
            xyz = [0,getattr(self,axis)]
        else:
            assert len(xyz) == 2, \
                '{} must be a list or tuple of length 2.'.format(axis)
            assert xyz[0] >= 0, \
                '{}[0] must be greater than or equal to 0.'.format(axis)
            assert xyz[1] <= getattr(self,axis), \
                '{}[1] cannot be greater than the maximum of the dimension ({}).'.format(axis, getattr(self,axis))
                
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
        
        if ct == None:
            # number of timepoints
            ct = list(range(0, getattr(self,axis)))
        else:
            assert np.any(np.greater(getattr(self,axis), ct)), \
            'At least one of the {}-indices was larger than largest index ({}).'.format(axis, getattr(self,axis) - 1)
            assert np.any(np.less_equal(0, ct)), \
            'At least one of the {}-indices was less than 0.'.format(axis)
            assert len(ct) != 0, \
            'At least one {}-index must be selected.'.format(axis)
            
        return ct

    """ ------------------- """
    """ -Pixel information- """
    """ ------------------- """
    
    def pixel_type(self,dtype=None):
        """pixel_type Same as dtype"""
        raise PendingDeprecationWarning(('pixel_type will be deprecated in bfio version 2.1.0.\n' + \
                                         'Switch to new dtype property.'))
        if dtype!=None:
            self.dtype = dtype
        
        return self._metadata.image(0).Pixels.PixelType
    
    @property
    def dtype(self):
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
    
    @dtype.setter
    def dtype(self,dtype):
        assert not self.__read_only, "The dtype attribute is read only. The image is either in read only mode or writing of the image has already begun."
        assert dtype in self._BPP.keys(), "Invalid data type."
        self._metadata.image(0).Pixels.PixelType = dtype
        
    @property
    def samples_per_pixel(self):
        """samples_per_pixel Number of samples per pixel """
        return self._metadata.image().Pixels.Channel().SamplesPerPixel
    
    @samples_per_pixel.setter
    def samples_per_pixel(self,
                          samples_per_pixel: int):
        self._metadata.image().Pixels.Channel().SamplesPerPixel = samples_per_pixel

    @property
    def spp(self):
        """spp Shorthand for samples_per_pixel """
        return self.samples_per_pixel
    
    @spp.setter
    def spp(self,samples_per_pixel):
        self.samples_per_pixel(samples_per_pixel)
        
    @property
    def bytes_per_pixel(self):
        """bytes_per_pixel Number of samples per pixel
        
        Returns:
            int
        """
        return self._BPP[self._metadata.image().Pixels.get_PixelType()]
    
    @bytes_per_pixel.setter
    def bytes_per_pixel(self,
                        bytes_per_pixel: int):
        raise AttributeError('Bytes per pixel cannot be set. Change the dtype instead')
    
    @property
    def bpp(self):
        """bytes_per_pixel Number of samples per pixel
        
        Returns:
            int
        """
        return self.bytes_per_pixel
    
    @bpp.setter
    def bpp(self,
            bytes_per_pixel: int):
        self.bytes_per_pixel = bytes_per_pixel

    """ -------------------------- """
    """ -Other Methods/Properties- """
    """ -------------------------- """
    @property
    def metadata(self):
        """metadata Get the metadata for the image

        This function calls the Bioformats metadata parser, which extracts metadata
        from an image. This returns a reference to an OMEXML class, which is a
        convenient handler for the complex xml metadata created by Bioformats.

        Most basic metadata information have their own BioReader methods, such as
        image dimensions(i.e. x, y, etc). However, in some cases it may
        be necessary to access the underlying metadata class.
        
        Minor changes have been made to the original OMEXML class created for
        python-bioformats, so the original OMEXML documentation should assist those
        interested in directly accessing the metadata. In general, it is best to
        assign data using the object properties to ensure the metadata stays in sync
        with the file.

        For information on the OMEXML class:
        https://github.com/CellProfiler/python-bioformats/blob/master/bioformats/omexml.py

        Returns:
            OMEXML: Class that simplifies editing ome-xml data
        """        
        return self._metadata
    
    @metadata.setter
    def metadata(self,value):
        raise AttributeError('The metadata attribute is read-only.')

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

class AbstractReader(object,metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self,frontend):
        self.frontend = frontend
    
    @abc.abstractmethod
    def read_metadata(self):
        pass
    
    @abc.abstractmethod
    def read_image(self):
        pass
    
