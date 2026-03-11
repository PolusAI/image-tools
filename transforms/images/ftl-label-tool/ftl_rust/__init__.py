import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy
from bfio import BioReader
from bfio import BioWriter

from .ftl_rust import PolygonSet as RustPolygonSet
from .ftl_rust import extract_tile

__all__ = ['PolygonSet']

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("PolygonSet")
logger.setLevel(logging.INFO)


class PolygonSet:
    """ Uses the Rust implementation to build polygons representing objects and to label them.
    """
    def __init__(self, connectivity: int, bin_thresh: float):
        """ Creates a PolygonSet object to provide a minimal interface with the Rust implementation.

        Args:
            connectivity: Determines neighbors among pixels. Must be 1, 2 or 3.
            See the README for more details.
            bin_thresh: The threshold to use for binarizing the input image.
        """
        if not (1 <= connectivity <= 3):
            raise ValueError(f'connectivity must be 1, 2 or 3. Got {connectivity} instead')

        self.__polygon_set: RustPolygonSet = RustPolygonSet(connectivity)
        self.connectivity: int = connectivity
        self.bin_thresh: float = bin_thresh
        self.metadata = None
        self.num_polygons = 0

    def __len__(self):
        """ Returns the number of objects that were detected. """
        return self.num_polygons

    def dtype(self):
        """ Chooses the minimal dtype for labels depending on the number of objects. """
        if self.num_polygons < 2 ** 8:
            dtype = numpy.uint8
        elif self.num_polygons < 2 ** 16:
            dtype = numpy.uint16
        else:
            dtype = numpy.uint32
        return dtype

    @staticmethod
    def _get_iteration_params(z_shape, y_shape, x_shape):
        tile_size = 512 if z_shape > 1 else (1024 * 5)

        num_slices = z_shape // tile_size
        if z_shape % tile_size != 0:
            num_slices += 1

        num_cols = y_shape // tile_size
        if y_shape % tile_size != 0:
            num_cols += 1

        num_rows = x_shape // tile_size
        if x_shape % tile_size != 0:
            num_rows += 1

        return tile_size, num_slices, num_cols, num_rows
    
    def read_from(self, infile: Path):
        """ Reads from a .ome.tif file and finds and labels all objects.

        Args:
            infile: Path to an ome.tif file for which to produce labels.
        """
        logger.info(f'Processing {infile.name}...')
        with BioReader(infile) as reader:
            self.metadata = reader.metadata

            tile_size, num_slices, num_cols, num_rows = self._get_iteration_params(reader.Z, reader.Y, reader.X)
            tile_count = 0
            for z in range(0, reader.Z, tile_size):
                z_max = min(reader.Z, z + tile_size)
                for y in range(0, reader.Y, tile_size):
                    y_max = min(reader.Y, y + tile_size)
                    for x in range(0, reader.X, tile_size):
                        x_max = min(reader.X, x + tile_size)

                        tile = numpy.squeeze(reader[y:y_max, x:x_max, z::z_max, 0, 0])
                        
                        # If the image has float values, binarize it using the threshold
                        if tile.dtype == numpy.float32 or tile.dtype == numpy.float64:
                            tile = (tile > self.bin_thresh).astype(numpy.uint8)

                        # If the image is not binary, make it binary
                        tile = (tile != 0).astype(numpy.uint8)

                        if tile.ndim == 2:
                            tile = tile[numpy.newaxis, :, :]
                        else:
                            tile = tile.transpose(2, 0, 1)
                        self.__polygon_set.add_tile(tile, (z, y, x))
                        tile_count += 1
                        logger.debug(f'added tile #{tile_count} ({z}:{z_max}, {y}:{y_max}, {x}:{x_max})')
                    logger.info(f'Reading Progress {100 * tile_count / (num_slices * num_cols * num_rows):6.3f}%...')
        
        logger.info('digesting polygons...')
        self.__polygon_set.digest()

        self.num_polygons = self.__polygon_set.len()
        logger.info(f'collected {self.num_polygons} polygons')
        return self

    def write_to(self, outfile: Path):
        """ Writes a labelled ome.tif to the given path.

        This uses the metadata of the input file and sets the dtype depending on the number of labelled objects.

        Args:
            outfile: Path where the labelled image will be written.
        """
        with BioWriter(outfile, metadata=self.metadata, max_workers=cpu_count()) as writer:
            writer.dtype = self.dtype()
            logger.info(f'writing {outfile.name} with dtype {self.dtype()}...')

            tile_size, _, num_cols, num_rows = self._get_iteration_params(writer.Z, writer.Y, writer.X)
            tile_count = 0
            for z in range(writer.Z):
                for y in range(0, writer.Y, tile_size):
                    y_max = min(writer.Y, y + tile_size)
                    for x in range(0, writer.X, tile_size):
                        x_max = min(writer.X, x + tile_size)

                        tile = extract_tile(self.__polygon_set, (z, z + 1, y, y_max, x, x_max))
                        writer[y:y_max, x:x_max, z:z + 1, 0, 0] = tile.transpose(1, 2, 0)
                        tile_count += 1
                        logger.debug(f'Wrote tile {tile_count}, ({z}, {y}:{y_max}, {x}:{x_max})')
                logger.info(f'Writing Progress {100 * tile_count / (num_cols * num_rows * writer.Z):6.3f}%...')
        return self
