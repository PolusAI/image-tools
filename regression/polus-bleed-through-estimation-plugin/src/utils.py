import functools
import logging
import os
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable
from typing import Generator
from typing import Union

import numpy
from bfio import BioReader
from bfio import BioWriter

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT', '.ome.tif')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('utils')
logger.setLevel(POLUS_LOG)

NUM_THREADS = max(1, int(cpu_count() * 0.5))
TILE_SIZE_2D = 1024 * 2
TILE_SIZE_3D = 128
MIN_DATA_SIZE = 2 ** 19  # Load at least this much data
MAX_DATA_SIZE = 2 ** 31  # Limit to loading 500MB of pixels
EPSILON = 1e-8  # To avoid divide-by-zero errors


"""
A dictionary of scores for each tile in an image.
key: (6-tuple of indices) (z_min, z_max, y_min, y_max, x_min, x_max)
value: (float) score
"""
ScoresDict = dict[tuple[int, int, int, int, int, int], float]

"""
A list of coordinates for each tile that was selected by a Selector.
Each item is a 6-tuple of indices: (z_min, z_max, y_min, y_max, x_min, x_max)
"""
TileIndices = list[tuple[int, int, int, int, int, int]]

"""
A dictionary for a file in `filepattern`.
"""
FPFileDict = dict[str, Union[int, Path]]


def replace_extension(name: str, new_extension: str = None) -> str:
    """ Replaces the extension in the name of an input image with `POLUS_EXT`
        for writing corresponding output images.
    """
    new_extension = POLUS_EXT if new_extension is None else new_extension
    return (
        name
        .replace('.ome.tif', new_extension)
        .replace('.ome.zarr', new_extension)
    )


def count_tiles(reader_or_writer: Union[BioReader, BioWriter]) -> int:
    """ Returns the number of tiles in a BioReader/BioWriter.
    """
    tile_size = TILE_SIZE_2D if reader_or_writer.Z == 1 else TILE_SIZE_3D
    num_tiles = (
        len(range(0, reader_or_writer.Z, tile_size)) *
        len(range(0, reader_or_writer.Y, tile_size)) *
        len(range(0, reader_or_writer.X, tile_size))
    )
    return num_tiles


def normalize_tile(tile: numpy.ndarray, min_val: float, max_val: float) -> numpy.ndarray:
    """ min-max normalization of a tile from an image.
        Also converts the dtype to numpy.float32
    """
    tile = (numpy.asarray(tile, dtype=numpy.float32) - min_val) / (max_val - min_val + EPSILON)
    return numpy.clip(tile, 0., 1.)


def tile_indices(
        reader_or_writer: Union[BioReader, BioWriter],
) -> Generator[tuple[int, int, int, int, int, int], None, None]:
    """ A generator for the indices of all tiles in a BioReader/BioWriter.
    """
    tile_size = TILE_SIZE_2D if reader_or_writer.Z == 1 else TILE_SIZE_3D

    for z_min in range(0, reader_or_writer.Z, tile_size):
        z_max = min(reader_or_writer.Z, z_min + tile_size)

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z_min, z_max, y_min, y_max, x_min, x_max


def count_tiles_2d(reader_or_writer: Union[BioReader, BioWriter]) -> int:
    """ Returns the number of 2d tiles in a BioReader/BioWriter.
    """
    tile_size = TILE_SIZE_2D if reader_or_writer.Z == 1 else TILE_SIZE_3D
    num_tiles = (
        reader_or_writer.Z *
        len(range(0, reader_or_writer.Y, tile_size)) *
        len(range(0, reader_or_writer.X, tile_size))
    )
    return num_tiles


def tile_indices_2d(
        reader_or_writer: Union[BioReader, BioWriter],
) -> Generator[tuple[int, int, int, int, int], None, None]:
    """ A generator for the indices of all 2d tiles in a BioReader/BioWriter.
    """
    tile_size = TILE_SIZE_2D if reader_or_writer.Z == 1 else TILE_SIZE_3D

    for z in range(reader_or_writer.Z):

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z, y_min, y_max, x_min, x_max


class TimeIt:
    """ A class to provide a decorator for timing the execution of a function.
    """
    def __init__(self, template: str = 'completed {:s} in {:.3f} seconds'):
        self.template: str = template

    def __call__(self, function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()

            logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
