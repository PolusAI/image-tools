"""Helpers for the plugin."""

import logging
import multiprocessing
import os
import typing

import bfio
import numpy

MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
TILE_SIZE = 1_024


def replace_extension(name: str, new_extension: typing.Optional[str] = None) -> str:
    """Replaces the extension in the name of an input image with `POLUS_IMG_EXT`."""
    new_extension = POLUS_IMG_EXT if new_extension is None else new_extension
    return name.replace(".ome.tif", new_extension).replace(".ome.zarr", new_extension)


def tile_index_generator(
    image_shape: tuple[int, int],
    tile_size: int,
) -> typing.Generator[tuple[int, int, int, int], None, None]:
    """Generate the indices of the tiles in the input image.

    Args:
        image_shape: shape of the input image.
        tile_size: size of the tiles.

    Yields:
        4-tuples of the form (y_min, y_max, x_min, x_max)
    """
    for y_min in range(0, image_shape[0], tile_size):
        y_max = min(y_min + tile_size, image_shape[0])
        for x_min in range(0, image_shape[1], tile_size):
            x_max = min(x_min + tile_size, image_shape[1])
            yield y_min, y_max, x_min, x_max


def read_tile_multi_channel(
    reader: list[bfio.BioReader],
    indices: tuple[int, int, int, int],
) -> numpy.ndarray:
    """Read a tile from the input image.

    The input image is assumed to be a multi-channel image.

    Args:
        reader: input image.
        indices: 4-tuple of the form (y_min, y_max, x_min, x_max).

    Returns:
        tile.
    """
    y_min, y_max, x_min, x_max = indices
    return numpy.squeeze(reader[0][y_min:y_max, x_min:x_max, 0, :, 0])


def read_tile_single_channel(
    readers: list[bfio.BioReader],
    indices: tuple[int, int, int, int],
) -> numpy.ndarray:
    """Read a tile from the input images.

    The input images are assumed to be single-channel images.

    Args:
        readers: input images.
        indices: 4-tuple of the form (y_min, y_max, x_min, x_max).

    Returns:
        tile.
    """
    y_min, y_max, x_min, x_max = indices
    return numpy.stack(
        [
            numpy.squeeze(reader[y_min:y_max, x_min:x_max, 0, 0, 0])
            for reader in readers
        ],
        axis=2,
    )
