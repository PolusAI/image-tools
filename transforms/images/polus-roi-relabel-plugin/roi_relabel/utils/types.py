import bfio
import typing

ReaderOrWriter = typing.Union[bfio.BioReader, bfio.BioWriter]

BlockGenerator = typing.Generator[tuple[int, int, int, int, int, int], None, None]
""" Generates 6-tuples of chunk indices in 3d ome-tif/zarr images.
(y_min, y_max, x_min, x_max, z_min, z_max)
"""

TileGenerator = typing.Generator[tuple[int, int, int, int, int], None, None]
""" Generates 5-tuples of tile indices in 2d or 3d ome-tif/zarr images.
(y_min, y_max, x_min, x_max, z)
"""

LOG_LEVELS = typing.Literal[
    'CRITICAL',
    'ERROR',
    'WARNING',
    'INFO',
    'DEBUG',
]
