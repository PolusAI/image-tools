import pathlib
import random
import typing

import bfio
import numpy
import scipy.ndimage

from roi_relabel.utils import helpers
from . import roi as roi
from . import graph as graph
from ..utils import constants

METHODS = [
    'contiguous',
    'randomize',
    'randomByte',
    'graphColoring',
    'optimizedGraphColoring',
]

# noinspection PyTypeHints
Methods = typing.Literal[tuple(METHODS)]


def _relabel_tile(
        inp_tile: numpy.ndarray,
        mapping: dict[int, int],
) -> numpy.ndarray:
    """ Relabels RoIs in a tile using the given label mapping.

    Args:
        inp_tile: A labeled tile.
        mapping: A dict mapping labels in `inp_tile` to new labels

    Returns:
        A tile remapped labels.
    """
    out_tile = numpy.copy(inp_tile)
    for k, v in mapping.items():
        mask = inp_tile == k
        out_tile[mask] = v
    return out_tile


def relabel(
        inp_path: pathlib.Path,
        out_path: pathlib.Path,
        method: Methods,
        range_multiplier: float = 1.0,
) -> bool:
    """ Relabels the image at `inp_path` using `method` and writes it out to
    `out_path`.

    This function is meant to be run in a thread.

    Args:
        inp_path: to the input image. This must be readable by `bfio`.
        out_path: to the output image. This must be either an `ome.tiff` or
         `ome.zarr` image.
        method: to be used for relabeling objects. See `README.md` for details.
        range_multiplier: to be used for graph coloring methods.

    Returns:
        `True` indicating that the output file was written successfully.
    """

    with bfio.BioReader(inp_path) as reader:

        rois: dict[int, roi.RoI] = dict()
        dtype = reader.dtype

        for (y_min, y_max, x_min, x_max, z_min, z_max) in helpers.block_indices(reader):

            tile = reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0]

            x_loc: slice
            y_loc: slice
            for l_ in map(int, numpy.unique(tile)):
                # noinspection PyUnresolvedReferences
                (y_loc, x_loc) = scipy.ndimage.find_objects(tile == l_)[0]
                r = roi.RoI(
                    start=roi.Point(x_loc.start, y_loc.start),
                    end=roi.Point(x_loc.stop, y_loc.stop),
                    label=int(l_)
                )
                if l_ in rois:
                    rois[l_] = rois[l_].merge_with(r)
                else:
                    rois[l_] = r

        if 0 in rois.keys():
            rois.pop(0)
        labels = list(rois.keys())

        mapping: dict[int, int]
        if method == 'contiguous':
            mapping = {k: i for i, k in enumerate(labels, start=1)}
        elif method == 'randomize':
            random.shuffle(labels)
            mapping = {k: i for i, k in enumerate(labels, start=1)}
        elif method == 'randomByte':
            random.shuffle(labels)
            mapping = {k: (1 + i % 255) for i, k in enumerate(labels, start=1)}
            dtype = numpy.uint8
        else:
            g = graph.Graph(list(rois.values()), range_multiplier)
            max_val = int(numpy.iinfo(dtype).max)

            if method == 'graphColoring':
                colors = g.coloring(max_val, optimize=False)
            elif method == 'optimizedGraphColoring':
                colors = g.coloring(max_val, optimize=True)
            else:
                raise ValueError(f'`method` "{method}" is invalid. Choose from {Methods}.')

            mapping = {k: c for k, (_, c) in enumerate(colors.items())}

        with bfio.BioWriter(out_path, metadata=reader.metadata) as writer:
            writer.dtype = dtype
            tile_size = max(1024, constants.TILE_SIZE_2D)

            for (y_min, y_max, x_min, x_max, z) in helpers.tile_indices(reader, tile_size):

                inp_tile = reader[y_min:y_max, x_min:x_max, z, 0, 0]
                out_tile = _relabel_tile(inp_tile, mapping).astype(writer.dtype)
                writer[y_min:y_max, x_min:x_max, z, 0, 0] = out_tile

    return True


__all__ = [
    'relabel',
    'METHODS',
]
