import functools
import logging
import os
import random
import time
import typing

import numpy

from . import constants
from . import types


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
    )
    return


def make_logger(name: str, level: types.LOG_LEVELS = None):
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.POLUS_LOG if level is None else level)
    return logger_


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    return


def replace_extension(name: str) -> str:
    """ Replaces the extension in the name of an input image with `POLUS_EXT`
    for writing corresponding output images.
    """
    return (
        name
        .replace('.ome.tif', constants.POLUS_EXT)
        .replace('.ome.zarr', constants.POLUS_EXT)
    )


def count_blocks(reader_or_writer: types.ReaderOrWriter, tile_size: int = None) -> int:
    """ Returns the number of 3d blocks in an image.
    """
    return len(list(block_indices(reader_or_writer, tile_size)))


def block_indices(
        reader_or_writer: types.ReaderOrWriter,
        tile_size: int = None,
) -> types.BlockGenerator:
    if tile_size is None:
        tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D

    for z_min in range(0, reader_or_writer.Z, tile_size):
        z_max = min(reader_or_writer.Z, z_min + tile_size)

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield y_min, y_max, x_min, x_max, z_min, z_max


def count_tiles(reader_or_writer: types.ReaderOrWriter, tile_size: int = None) -> int:
    """ Returns the number of 2d tiles in an image.
    """
    return len(list(tile_indices(reader_or_writer, tile_size)))


def tile_indices(
        reader_or_writer: types.ReaderOrWriter,
        tile_size: int = None,
) -> types.TileGenerator:
    if tile_size is None:
        tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D

    for z in range(reader_or_writer.Z):

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield y_min, y_max, x_min, x_max, z


class TimeIt:
    """ A class to provide a decorator for timing the execution of a function.
    This will log the time taken with the given logger.
    """
    def __init__(
            self,
            logger: logging.Logger,
            template: str = 'completed {:s} in {:.3f} seconds',
    ):
        self.logger = logger
        self.template: str = template

    def __call__(self, function: typing.Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            start = time.perf_counter()
            result = function(*args, **kwargs)
            end = time.perf_counter()

            self.logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
