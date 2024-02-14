"""Helper functions for the vector_to_label plugin."""

import pathlib
import shutil
import typing

import bfio
import numpy
from polus.plugins.formats.label_to_vector.utils import constants


def init_zarr_file(
    path: pathlib.Path,
    metadata: typing.Any,  # noqa: ANN401
    dtype: typing.Optional[numpy.dtype] = None,
) -> None:
    """Initializes the zarr file.

    Args:
        path: The path to the zarr file.
        metadata: The metadata to save to the zarr file.
        dtype: The dtype to use for the zarr file.
    """
    with bfio.BioWriter(path, metadata=metadata) as writer:
        writer.dtype = dtype or numpy.uint32
        writer.C = 1
        writer.channel_names = ["label"]
        writer._backend._init_writer()


def zarr_to_tif(
    zarr_path: pathlib.Path,
    out_path: pathlib.Path,
    dtype: typing.Optional[numpy.dtype] = None,
) -> None:
    """Converts a zarr file to a tif file.

    Args:
        zarr_path: The path to the zarr file.
        out_path: The path to the tif file.
        dtype: The dtype to use for the tif file.
    """
    with (
        bfio.BioReader(zarr_path, max_workers=constants.NUM_THREADS) as reader,
        bfio.BioWriter(
            out_path,
            max_workers=constants.NUM_THREADS,
            metadata=reader.metadata,
        ) as writer,
    ):
        writer.dtype = dtype or numpy.uint32

        for z in range(reader.Z):
            for y in range(0, reader.Y, constants.TILE_SIZE):
                y_max = min(reader.Y, y + constants.TILE_SIZE)

                for x in range(0, reader.X, constants.TILE_SIZE):
                    x_max = min(reader.X, x + constants.TILE_SIZE)

                    tile = reader[y:y_max, x:x_max, z : z + 1, 0, 0]
                    writer[y:y_max, x:x_max, z : z + 1, 0, 0] = tile

    shutil.rmtree(zarr_path)
