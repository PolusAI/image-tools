from bfio import BioReader,BioWriter
from tifffile import tifffile
from pathlib import Path
import bioformats
import numpy as np
from bfiop.bfio import BioReader,BioWriter
from concurrent.futures import ThreadPoolExecutor
import abc
from base_class import BioBase

class ReaderBackend(metaclass=abc.ABCMeta):
    
    _bioreader = None
    
    name = None
    
    def __init__(self,bioreader):
        self._bioreader = bioreader

class PythonReaderBackend(BioBase):

    name = 'python'

    def __init__(self):
        self._rdr = tifffile.TiffFile(self._file_path)

    def _chunk_indices(self, X, Y, Z):
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2

        offsets = []
        bytecounts = []

        x_tiles = np.arange(X[0] // 1024, np.ceil(X[1] / 1024), dtype=int)
        y_tile_stride = np.ceil(self.num_x() / 1024).astype(int)

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

        w, l, d = self._tile_indices[args[1]]

        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)

        if segment is None:
            segment = keyframe.nodata

        out[l: l + shape[1],
        w: w + shape[2],
        d, 0, 0] = segment.squeeze()

    def read_image(self,out,X,Y,Z,X_tile_shape,Y_tile_shape):
        self._out=out
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle

        # Do the work
        offsets, bytecounts = self._chunk_indices(X, Y, Z)
        self._tile_indices = []
        for z in range(0, Z[1] - Z[0]):
            for y in range(0, Y_tile_shape, self._TILE_SIZE):
                for x in range(0, X_tile_shape, self._TILE_SIZE):
                    self._tile_indices.append((x, y, z))

        with ThreadPoolExecutor(self._max_workers) as executor:
            executor.map(self._decode_tile, fh.read_segments(offsets, bytecounts))
        return out

    def close_image(self):
        self._rdr.close()

class JavaReaderBackend(BioBase):
    
   name = 'java'

   def __init__(self):
    self._rdr = bioformats.ImageReader(self._file_path)


   def read_image(self,X,Y,Z,C,T,x_range,y_range,Z_tile_shape,I):
       open_in_parts = (x_range) * (y_range) > self._pix['chunk']

       for ti, t in zip(range(0, len(T)), T):
            for zi, z in zip(range(0, Z_tile_shape), range(Z[0], Z[1])):
                if not open_in_parts:
                    if self._pix['interleaved']:
                        I_temp = self._rdr.read(c=None, z=z, t=t, rescale=False, XYWH=(X[0], Y[0], x_range, y_range))
                        for ci, c in zip(range(0, len(C)), C):
                            I[:, :, zi, ci, ti] = I_temp[:, :, c]
                    else:
                        for ci, c in zip(range(0, len(C)), C):
                            I[:, :, zi, ci, ti] = self._rdr.read(c=c, z=z, t=t, rescale=False, XYWH=(X[0], Y[0], x_range, y_range))

                else:
                    if self._pix['interleaved']:
                        for x in range(X[0], X[1], self._TILE_SIZE):
                            x_max = np.min([x + self._TILE_SIZE, X[1]])
                            x_range = x_max - x
                            for y in range(Y[0], Y[1], self._TILE_SIZE):
                                y_max = np.min([y + self._TILE_SIZE, Y[1]])
                                y_range = y_max - y
                                I[y - Y[0]:y_max - Y[0], x - X[0]:x_max - X[0], zi, :, ti] = self._rdr.read(
                                       c=None, z=z, t=t, rescale=False, XYWH=(x, y, x_range, y_range))
                    else:
                        for ci, c in zip(range(0, len(C)), C):
                            for x in range(X[0], X[1], self._TILE_SIZE):
                                x_max = np.min([x + self._TILE_SIZE, X[1]])
                                x_range = x_max - x
                                for y in range(Y[0], Y[1], self._TILE_SIZE):
                                    y_max = np.min(
                                           [y + self._TILE_SIZE, Y[1]])
                                    y_range = y_max - y
                                    I[y - Y[0]:y_max - Y[0], x - X[0]:x_max - X[0], zi, ci, ti] = self._rdr.read(
                                           c=c, z=z, t=t, rescale=False, XYWH=(x, y, x_range, y_range))
       return I

   def close_image(self):
       self._rdr.close()


BACKEND = PythonBackend


class WriterBackend(metaclass=abc.ABCMeta):
    _biowriter = None

    name = None

    def __init__(self, biowriter):
        self._biowriter = biowriter


class JavaWriterBackend(BioBase):
    name = 'java'




class PythonWriterBackend(BioBase):
    name = 'python'

    def __init__(self):

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






def set_backend(backend):
    
    backend = backend.lower()
    assert backend.lower() in ['python','java']
    
    global BACKEND
    if backend == 'python':
        BACKEND = PythonBackend

    elif backend == 'java':
        BACKEND = JavaBackend
        
    BioReader.set_backend(BACKEND)