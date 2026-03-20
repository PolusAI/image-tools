"""Helper functions for my plugins."""

import functools
import logging
import os
import random
import time
import typing

import numpy

from . import constants
from . import types


def seed_everything(seed: int) -> None:
    """Set a random seed for every relevant package."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.default_rng(seed)


def replace_extension(name: str) -> str:
    """Replace file extension based on env var."""
    return name.replace(".ome.tif", constants.POLUS_IMG_EXT).replace(
        ".ome.zarr",
        constants.POLUS_IMG_EXT,
    )


def count_blocks(
    reader_or_writer: types.ReaderOrWriter,
    tile_size: typing.Optional[int] = None,
) -> int:
    """Count the number of 3d blocks in an image."""
    return len(list(block_indices(reader_or_writer, tile_size)))


def block_indices(
    reader_or_writer: types.ReaderOrWriter,
    tile_size: typing.Optional[int] = None,
) -> types.BlockGenerator:
    """Iterate on 3d blocks in a bfio image."""
    if tile_size is None:
        tile_size = (
            constants.TILE_SIZE_2D
            if reader_or_writer.Z == 1
            else constants.TILE_SIZE_3D
        )

    for z_min in range(0, reader_or_writer.Z, tile_size):
        z_max = min(reader_or_writer.Z, z_min + tile_size)

        for x_min in range(0, reader_or_writer.X, tile_size):
            x_max = min(reader_or_writer.X, x_min + tile_size)

            for y_min in range(0, reader_or_writer.Y, tile_size):
                y_max = min(reader_or_writer.Y, y_min + tile_size)

                yield x_min, x_max, y_min, y_max, z_min, z_max


def count_tiles(
    reader_or_writer: types.ReaderOrWriter,
    tile_size: typing.Optional[int] = None,
) -> int:
    """Count the number of 2d tiles in an image."""
    return len(list(tile_indices(reader_or_writer, tile_size)))


def tile_indices(
    reader_or_writer: types.ReaderOrWriter,
    tile_size: typing.Optional[int] = None,
) -> types.TileGenerator:
    """Iterate on 2d tiles in a bfio image."""
    if tile_size is None:
        tile_size = constants.TILE_SIZE_2D

    for z in range(reader_or_writer.Z):
        for x_min in range(0, reader_or_writer.X, tile_size):
            x_max = min(reader_or_writer.X, x_min + tile_size)

            for y_min in range(0, reader_or_writer.Y, tile_size):
                y_max = min(reader_or_writer.Y, y_min + tile_size)

                yield x_min, x_max, y_min, y_max, z


class TimeIt:
    """A decorator for timing the execution of a function.

    This will log the time taken with the given logger.
    """

    def __init__(
        self,
        logger: logging.Logger,
        template: str = "completed {:s} in {:.3f} seconds",
    ) -> None:
        """Create the decorator."""
        self.logger = logger
        self.template: str = template

    def __call__(self, function: typing.Callable) -> typing.Callable:
        """Call the decorator."""

        @functools.wraps(function)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            start = time.perf_counter()
            result = function(*args, **kwargs)
            end = time.perf_counter()

            self.logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
