"""Helper types for my plugins."""
import typing

import bfio

ReaderOrWriter = typing.Union[bfio.BioReader, bfio.BioWriter]

BlockGenerator = typing.Generator[tuple[int, int, int, int, int, int], None, None]
# Generates 6-tuples of chunk indices in 3d ome-tif/zarr images.

TileGenerator = typing.Generator[tuple[int, int, int, int, int], None, None]
# Generates 5-tuples of tile indices in 2d or 3d ome-tif/zarr images.

LOG_LEVELS = typing.Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
]
