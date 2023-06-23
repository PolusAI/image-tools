"""Helper functions for the label_to_vector plugin."""

import logging
import pathlib
import shutil
import typing

import bfio
import numpy

from . import constants


def make_logger(
    name: str,
    level: typing.Optional[constants.LOG_LEVELS] = None,
) -> logging.Logger:
    """Creates a logger with the given name and level.

    Args:
        name: The name of the logger.
        level: The level of the logger. Defaults to POLUS_LOG.

    Returns:
        A logger with the given name and level.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(constants.POLUS_LOG if level is None else level)
    return logger_


def replace_extension(
    file: pathlib.Path,
    *,
    extension: typing.Optional[str] = None,
) -> str:
    """Replaces the extension of a file with a new one.

    Args:
        file: The file to replace the extension of.
        extension: The new extension to use. Defaults to POLUS_IMG_EXT.

    Returns:
        The file name with the new extension.
    """
    input_extension = "".join(
        s for s in file.suffixes[-2:] if len(s) < constants.SUFFIX_LEN
    )
    extension = constants.POLUS_EXT if extension is None else extension
    file_name = file.name
    if "_flow" in file_name:
        file_name = "".join(file_name.split("_flow"))
    if "_tmp" in file_name:
        file_name = "".join(file_name.split("_tmp"))
    return file_name.replace(input_extension, extension)


def determine_dtype(num_cells: int) -> numpy.dtype:
    """Determines the smallest numpy.dtype for the number of cells.

    Args:
        num_cells: Total number of cells in an image

    Returns:
        The smallest numpy.dtype that can be used for that array
    """
    if num_cells < 2**8:
        return numpy.uint8

    if num_cells < 2**16:
        return numpy.uint16

    if num_cells < 2**32:
        return numpy.uint32

    return numpy.uint64


def init_zarr_file(
    path: pathlib.Path,
    metadata: typing.Any,  # noqa: ANN401
) -> None:
    """Initializes the zarr file.

    Args:
        path: The path to the zarr file.
        metadata: The metadata to save to the zarr file.
    """
    with bfio.BioWriter(path, metadata=metadata) as writer:
        writer.dtype = numpy.uint32
        writer.C = 1
        writer.channel_names = ["label"]
        writer._backend._init_writer()


def zarr_to_tif(zarr_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Converts a zarr file to a tif file.

    Args:
        zarr_path: The path to the zarr file.
        out_path: The path to the tif file.
    """
    with (
        bfio.BioReader(zarr_path, max_workers=constants.NUM_THREADS) as reader,
        bfio.BioWriter(
            out_path,
            max_workers=constants.NUM_THREADS,
            metadata=reader.metadata,
        ) as writer,
    ):
        writer.dtype = numpy.uint32

        for z in range(reader.Z):
            for y in range(0, reader.Y, constants.TILE_SIZE):
                y_max = min(reader.Y, y + constants.TILE_SIZE)

                for x in range(0, reader.X, constants.TILE_SIZE):
                    x_max = min(reader.X, x + constants.TILE_SIZE)

                    tile = reader[y:y_max, x:x_max, z : z + 1, 0, 0]
                    writer[y:y_max, x:x_max, z : z + 1, 0, 0] = tile

    shutil.rmtree(zarr_path)
