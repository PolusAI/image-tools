import multiprocessing
from queue import Queue


class base :
    _file_path = None
    _metadata = None
    _xyzct = None
    _pix = None
    _read_only = None
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
    _max_workers = multiprocessing.cpu_count() // 2
    def channel_names(self,cnames=None):
        """channel_names

        Returns:
            list: Strings indicating channel names
        """
        if self._read_only:
            if cnames:
               # assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
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
        if self._read_only:
            if X:
            #   assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
                assert X >= 1
                self._metadata.image(0).Pixels.SizeX = X
                self._xyzct['X'] = X
        return self._xyzct['X']

    def num_y(self,Y=None):
        """num_y Height of image in pixels

        Returns:
            int: Height of image in pixels
        """
        if self._read_only:
            if Y:
            # assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
                assert Y >= 1
                self._metadata.image(0).Pixels.SizeY = Y
                self._xyzct['Y'] = Y
        return self._xyzct['Y']

    def num_z(self,Z=None):
        """num_z Depth of image in pixels

        Returns:
            int: Depth of image in pixels
        """
        if self._read_only:
            if Z:
            #assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
                assert Z >= 1
                self._metadata.image(0).Pixels.SizeZ = Z
                self._xyzct['Z'] = Z
        return self._xyzct['Z']

    def num_c(self,C=None):
        """num_c Number of channels in the image

        Returns:
            int: Number of channels
        """
        if self._read_only:
            if C:
              #  assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
                assert C >= 1
                self._metadata.image(0).Pixels.SizeC = C
                self._xyzct['C'] = C
        return self._xyzct['C']

    def num_t(self,T=None):
        """num_x Number of timepoints in an image

        Returns:psize=None, units=None
            int: Number of timepoints
        """
        if self._read_only:
            if T:
               # assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
                assert T >= 1
                self._metadata.image(0).Pixels.SizeT = T
                self._xyzct['T'] = T

    def __init__(self, file_path):
        self._file_path = file_path
        _metadata = None
        _xyzct = None
        _pix = None
        _max_workers = multiprocessing.cpu_count() // 2
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