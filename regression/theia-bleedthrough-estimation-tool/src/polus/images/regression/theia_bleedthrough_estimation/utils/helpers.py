"""Helper functions for theia_bleedthrough_estimation plugin."""

import functools
import logging
import time
import typing

import bfio

from . import constants

ReaderOrWriter = typing.Union[bfio.BioReader, bfio.BioWriter]
Tiles2D = typing.Generator[tuple[int, int, int, int, int], None, None]
Tiles3D = typing.Generator[tuple[int, int, int, int, int, int], None, None]


def make_logger(name: str, level: str = constants.POLUS_LOG) -> logging.Logger:
    """Creates a logger with the given name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def replace_extension(name: str, new_extension: typing.Optional[str] = None) -> str:
    """Replaces the extension in the name of an input image with `POLUS_IMG_EXT`."""
    new_extension = constants.POLUS_IMG_EXT if new_extension is None else new_extension
    return name.replace(".ome.tif", new_extension).replace(".ome.zarr", new_extension)


def tile_indices_2d(reader_or_writer: ReaderOrWriter) -> Tiles2D:
    """A generator for the indices of all 2d tiles in a BioReader/BioWriter."""
    tile_size = (
        constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D
    )

    for z in range(reader_or_writer.Z):
        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z, y_min, y_max, x_min, x_max


def count_tiles_2d(reader_or_writer: ReaderOrWriter) -> int:
    """Returns the number of 2d tiles in a BioReader/BioWriter."""
    return len(list(tile_indices_2d(reader_or_writer)))


def tile_indices_3d(reader_or_writer: ReaderOrWriter) -> Tiles3D:
    """A generator for the indices of all 3d chunks in a BioReader/BioWriter."""
    tile_size = (
        constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D
    )

    for z_min in range(0, reader_or_writer.Z, tile_size):
        z_max = min(reader_or_writer.Z, z_min + tile_size)

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z_min, z_max, y_min, y_max, x_min, x_max


def count_tiles_3d(reader_or_writer: ReaderOrWriter) -> int:
    """Returns the number of 3d chunks in a BioReader/BioWriter."""
    return len(list(tile_indices_3d(reader_or_writer)))


class TimeIt:
    """A class to provide a decorator for timing the execution of a function."""

    def __init__(  # noqa: D107
        self,
        logger: logging.Logger,
        template: str = "completed {:s} in {:.3f} seconds",
    ) -> None:
        self.template: str = template
        self.logger: logging.Logger = logger

    def __call__(self, function: typing.Callable):  # noqa: D102, ANN204
        @functools.wraps(function)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()

            self.logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
