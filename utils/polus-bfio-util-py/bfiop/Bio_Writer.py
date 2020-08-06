from .base_class import base
from tifffile import tifffile
from . import OmeXml as bioformats
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import struct,io
from pathlib import Path
import zlib, time, copy, multiprocessing
from multiprocessing.dummy import Pool

class BioWriter(base):
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

    __writer = None

    # Buffering variables for iterating over an image
    _raw_buffer = Queue(maxsize=1)  # only preload one supertile at a time
    _supertile_index = Queue()
    _pixel_buffer = None
    _put_thread = None
    _tile_thread = None
    _tile_x_offset = 0
    _tile_last_column = 0

    # Variables for saving multipage tiffs
    _current_page = None
    _ifds = None
    _databytecounts = []
    _tags = []  # list of (code, ifdentry, ifdvalue, writeonce)
    _page_open = False

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
            assert metadata.__class__.__name__ == "OMEXML"
            self._metadata = bioformats.OMEXML(str(metadata))
            self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                           'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                           'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                           'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                           'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints
            self._pix = {'type': self._metadata.image().Pixels.get_PixelType(),  # string indicating pixel type
                         # bytes per pixel
                         'bpp': self._BPP[self._metadata.image().Pixels.get_PixelType()],
                         'spp': 1}  # samples per pixel
            # number of pixels to load at a time
            self._pix['chunk'] = self._MAX_BYTES / \
                                 (self._pix['spp'] * self._pix['bpp'])
            self._pix['interleaved'] = False
            self._metadata.image(0).Name = file_path
            self._metadata.image().Pixels.channel_count = self._xyzct['C']
            self._metadata.image().Pixels.DimensionOrder = bioformats.DO_XYZCT
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

    def _pack(self, fmt, *val):
        return struct.pack(self._byteorder + fmt, *val)

    def _addtag(self, code, dtype, count, value, writeonce=False):
        tags = self._tags

        # compute ifdentry & ifdvalue bytes from code, dtype, count, value
        # append (code, ifdentry, ifdvalue, writeonce) to tags list
        if not isinstance(code, int):
            code = tifffile.TIFF.TAGS[code]
        try:
            tifftype = tifffile.TIFF.DATA_DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f'unknown dtype {dtype}') from exc
        rawcount = count

        if dtype == 's':
            # strings; enforce 7-bit ASCII on unicode strings
            if code == 270:
                value = tifffile.bytestr(value, 'utf-8') + b'\0'
            else:
                value = tifffile.bytestr(value, 'ascii') + b'\0'
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
        ifdentry = [self._pack('HH', code, tifftype),
                    self._pack(self.__writer._offsetformat, rawcount)]
        ifdvalue = None
        if struct.calcsize(dtype) * count <= self.__writer._offsetsize:
            # value(s) can be written directly
            if isinstance(value, bytes):
                ifdentry.append(self._pack(self.__writer._valueformat, value))
            elif count == 1:
                if isinstance(value, (tuple, list, np.ndarray)):
                    value = value[0]
                ifdentry.append(self._pack(self.__writer._valueformat, self._pack(dtype, value)))
            else:
                ifdentry.append(self._pack(self.__writer._valueformat,
                                           self._pack(str(count) + dtype, *value)))
        else:
            # use offset to value(s)
            ifdentry.append(self._pack(self.__writer._offsetformat, 0))
            if isinstance(value, bytes):
                ifdvalue = value
            elif isinstance(value, np.ndarray):
                if value.size != count:
                    raise RuntimeError('value.size != count')
                if value.dtype.char != dtype:
                    raise RuntimeError('value.dtype.char != dtype')
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = self._pack(str(count) + dtype, *value)
            else:
                ifdvalue = self._pack(dtype, value)
        tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

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
        self._datadtype = np.dtype(self._pix['type']).newbyteorder(self._byteorder)

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

        # print(self._metadata)
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

        if self.physical_size_x() is not None:
            self._addtag(282, '2I', 1, rational(10000 / self.physical_size_x()[0] / 10000))  # XResolution in pixels/cm
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
            description = "ImageJ=\nhyperstack=true\nimages={}\nchannels={}\nslices={}\nframes={}".format(self.num_z(),
                                                                                                          self.num_c(),
                                                                                                          self.num_z(),
                                                                                                          self.num_t())
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
        # print('offset,pos: {},{}'.format(offset,pos))
        # print('databyteoffsets: {}'.format(self._databyteoffsets))
        for size in self._databyteoffsets:
            self._ifd.write(self._pack(offsetformat, size))

        # update strip/tile bytecounts
        offset, pos = self._databytecountsoffset
        # print('offset,pos: {},{}'.format(offset,pos))
        # print('databytecounts: {}'.format(self._databytecounts))
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
        # with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()//2) as executor:
        with ThreadPoolExecutor(max_workers=1) as executor:
            compressed_tiles = iter(executor.map(compress, tileiter))
        for tileindex in tiles:
            t = next(compressed_tiles)
            self._databyteoffsets[tileindex] = fh.tell()
            fh.write(t)
            self._databytecounts[tileindex] = len(t)

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
                'At least one of the {}-indices was larger than largest index ({}).'.format(axis,
                                                                                            self._xyzct[axis] - 1))
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
        jutil.attach()

        # Write the image
        self.write_image(I[:self.num_y(), :, np.newaxis, np.newaxis, np.newaxis],
                         X=[X[0]],
                         Y=[Y[0]])

        # Detach the jvm
        jutil.detach()

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
            self._raw_buffer.put(np.copy(self._pixel_buffer[:, 0:1024]))
            self._pixel_buffer[:, 0:1024] = self._pixel_buffer[:, 1024:2048]
            self._pixel_buffer[:, 1024:] = 0
            self._tile_x_offset += 1024
            self._tile_last_column = np.argwhere((self._pixel_buffer == 0).all(axis=0))[0, 0]

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

    def maximum_batch_size(self, tile_size, tile_stride=None):
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

        xyoffset = [(tile_size[0] - tile_stride[0]) / 2, (tile_size[1] - tile_stride[1]) / 2]

        num_tile_rows = int(np.ceil(self.num_y() / tile_stride[0]))
        num_tile_cols = (1024 - xyoffset[1]) // tile_stride[1]
        if num_tile_cols == 0:
            num_tile_cols = 1

        return int(num_tile_cols * num_tile_rows)

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

            print('Original and saved images are the same: {}'.format(np.array_equal(original_image,saved_image)))

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

        # calculate unpadding
        if not (set(tile_size) & set(tile_stride)):
            xyoffset = [int((tile_size[0] - tile_stride[0]) / 2), int((tile_size[1] - tile_stride[1]) / 2)]
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
        thread_pool = ThreadPoolExecutor(max_workers=2)

        # generate the indices for each tile
        # TODO: modify this to grab more than just the first z-index
        X = []
        Y = []
        Z = []
        C = []
        T = []
        x_list = np.array(np.arange(0, self.num_x(), tile_stride[1]))
        y_list = np.array(np.arange(0, self.num_y(), tile_stride[0]))
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