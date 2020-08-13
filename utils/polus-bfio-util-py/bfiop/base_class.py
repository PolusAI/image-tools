import abc
from bfiop import OmeXml as bioformats
import numpy as np
from queue import Queue

import  multiprocessing


# Need to  add handling  of python2.7 compiler
# import six
#
# if six.PY3:
#     from abc import abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod
# else:
#     from abc import abstractmethod, abstractproperty



class BioBase(metaclass=abc.ABCMeta) :
    """
    Base class for reading and writing OME tiled tiff format data. Class initialises  and parses metadata if they are present while reading and writing images.


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
    # Set constants for opening images
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
    # Buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1)  # only preload one supertile at a time
    _data_in_buffer = Queue(maxsize=1)
    _supertile_index = Queue()
    _pixel_buffer = None
    _fetch_thread = None
    _tile_thread = None
    _tile_x_offset = 0
    _tile_last_column = 0
    _metadata=None
    @property
    def set_read(self):
        return self._read_only

    @set_read.setter
    def set_read(self, bool):
        self._read_only = bool

    @property
    def set_dim(self):
        # Information about image dimensions

        return self._xyzct

    @set_dim.setter
    def set_dim(self):
        #print(_metadata)
        self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                       'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                       'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                       'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                       'T': self._metadata.image().Pixels.get_SizeT()}  # n
        print(self._xyzct)


    def __init__(self, file_path,_metadata=None,max_workers=None):
        """__init__ Initialize  filepath,metadata and number of threads

         Args:
            file_path (str): Path to output file
            _metadata(str) : metadata  if present along with image
            max_workers(int) : Number of threads to be used
        """
        self._file_path = file_path
        self._metadata = _metadata

        self._xyzct = None
        self._pix = None
        self._read_only = True
        self._max_workers = max_workers if (max_workers and max_workers!=1)   else  (0 or multiprocessing.cpu_count() // 2)
        if self._read_only:
        # Information about image dimensions
        #     self._xyzct = {'X': _metadata.image().Pixels.get_SizeX(),  # image width
        #                'Y': _metadata.image().Pixels.get_SizeY(),  # image height
        #                'Z': _metadata.image().Pixels.get_SizeZ(),  # image depth
        #                'C': _metadata.image().Pixels.get_SizeC(),  # number of channels
        #                'T': _metadata.image().Pixels.get_SizeT()}  # number of timepoints

        # Information about data type and loading
            self._pix = {'type': _metadata.image().Pixels.get_PixelType(),  # string indicating pixel type
                     'bpp': self._BPP[_metadata.image().Pixels.get_PixelType()],  # bytes per pixel
                     'spp': _metadata.image().Pixels.Channel().SamplesPerPixel}  # samples per pixel

        # number of pixels to load at a time
            self._pix['chunk'] = self._MAX_BYTES / \
                             (self._pix['spp'] * self._pix['bpp'])

        # determine if channels are interleaved
            self._pix['interleaved'] = self._pix['spp'] > 1



    def channel_names(self,cnames=None):
        """channel_names

        Returns:
            list: Strings indicating channel names
        """
        if cnames:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert len(cnames) == self._xyzct['C'], "Number of names does not match number of channels."
                for i in range(0, len(cnames)):
                    self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]

        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0, self._xyzct['C'])]


    def num_x(self,X=None):
        """num_x Width of image in pixels

        Returns:
            int: Width of image in pixels
        """
        if X:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert X >= 1
                self._metadata.image(0).Pixels.SizeX = X
                self._xyzct['X'] = X
        return self._xyzct['X']


    def num_y(self,Y=None):
        """num_y Height of image in pixels

        Returns:
            int: Height of image in pixels
        """
        if Y:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert Y >= 1
                self._metadata.image(0).Pixels.SizeY = Y
                self._xyzct['Y'] = Y
        return self._xyzct['Y']


    def num_z(self,Z=None):
        """num_z Depth of image in pixels

        Returns:
            int: Depth of image in pixels
        """
        if Z:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert Z >= 1
                self._metadata.image(0).Pixels.SizeZ = Z
                self._xyzct['Z'] = Z
        return self._xyzct['Z']


    def num_c(self,C=None):
        """num_c Number of channels in the image

        Returns:
            int: Number of channels
        """
        if C:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert C >= 1
                self._metadata.image(0).Pixels.SizeC = C
                self._xyzct['C'] = C
        return self._xyzct['C']


    def num_t(self,T=None):
        """num_x Number of timepoints in an image

        Returns:psize=None, units=None
            int: Number of timepoints
        """
        if T:
            if not  self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert T >= 1
                self._metadata.image(0).Pixels.SizeT = T
                self._xyzct['T'] = T


    def physical_size_x(self,psize=None, units=None):
        """num_x Size of pixels in x-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        if psize != None and units != None:
            if  not self._read_only :
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                self._metadata.image(0).Pixels.PhysicalSizeX = psize
                self._metadata.image(0).Pixels.PhysicalSizeXUnit = units
        elif psize == None and units == None:
            pass
        else:
            raise ValueError('Both psize and units must be defined, or neither should be defined.')
        return (self._metadata.image(0).Pixels.PhysicalSizeX, self._metadata.image(0).Pixels.PhysicalSizeXUnit)


    def physical_size_y(self,psize=None, units=None):
        """num_y Size of pixels in y-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        if psize != None and units != None:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                self._metadata.image(0).Pixels.PhysicalSizeY = psize
                self._metadata.image(0).Pixels.PhysicalSizeYUnit = units
        elif psize == None and units == None:
            pass
        else:
            raise ValueError('Both psize and units must be defined, or neither should be defined.')
        return (self._metadata.image(0).Pixels.PhysicalSizeY, self._metadata.image(0).Pixels.PhysicalSizeYUnit)


    def physical_size_z(self,psize=None, units=None):
        """num_z Size of pixels in z-dimension

        Returns:
            float: Number of units per pixel
            str: Units (i.e. cm or mm)
        """
        if psize != None and units != None:
            if not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
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
        if self._read_only:
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
        else :
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
        if self._read_only :
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
        else :
            if np.any(np.greater_equal(self._xyzct[axis], ct)):
                ValueError(
                    'At least one of the {}-indices was larger than largest index ({}).'.format(axis,
                                                                                                self._xyzct[axis] - 1))
            elif np.any(np.less(0, ct)):
                ValueError(
                    'At least one of the {}-indices was less than 0.'.format(axis))
            elif len(ct) == 0:
                ValueError('At least one {}-index must be selected.'.format(axis))
            elif isinstance(ct, list):
                TypeError("The values for {} must be a list.".format(axis))


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
        if dtype:
            if  not self._read_only:
                assert not self._read_only, "The image has started to be written. To modify the xml again, reinitialize."
                assert dtype in self._BPP.keys(), "Invalid data type."
                self._metadata.image(0).Pixels.PixelType = dtype
                self._pix['type'] = dtype

        return self._metadata.image(0).Pixels.PixelType

    def xml_metadata(self,metadata=None,var=None):

       if metadata and not var:
           return bioformats.OMEXML(metadata)
       elif not metadata and not var:
           return bioformats.OMEXML()
       else:
           return bioformats

    @abc.abstractmethod
    def _buffer_supertile(self, column_start, column_end):
        pass

    @abc.abstractmethod
    def maximum_batch_size(self, tile_size, tile_stride=None):
        pass