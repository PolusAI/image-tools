from collections import Generator
from pathlib import Path
from typing import Optional

import numpy
from bfio import BioReader

from . import constants
from . import local_distogram as distogram

# TODO: My PR with several performance improvements to the distogram package is
#  still under review. For now, we will use a local copy of the, as yet,
#  unpublished version of distogram. Once that PR is merged and the pypi package
#  is updated,, I plan on coming back here to add distogram to requirements.txt

""" The generator of coordinates of a tile as (y_min, y_max, x_min, x_max).
"""
TileIndices = Generator[tuple[int, int, int, int], None, None]

"""
A Bounding-Box is a 6-tuple (z1, z2, y1, y2, x1, x2) defining a cuboid.
z1, y1 and x1 are inclusive.
z2, y2 and x2 are exclusive.
"""
BoundingBox = tuple[int, int, int, int, int, int]


def distogram_from_batch(values: list[float], bin_count: int, weighted_diff: bool) -> distogram.Distogram:
    """ Create a distogram from a batch of values rather than a stream of values.

    Sometimes, O(n.log(n)) is faster than O(n). Python's built-in sort function is
     fast enough that it allows us to outperform the theoretically faster update
     algorithm for a Distogram.

    Args:
        values: The values to add in a single batch.
        bin_count: number of bins to use in the distogram.
        weighted_diff: whether the bin widths are weighted by density.

    Returns:
        A Distogram
    """
    values = list(sorted(values))
    step = len(values) // bin_count
    values = [values[i: i + step] for i in range(0, len(values), step)]
    bins = [(v[0], len(v)) for v in values]

    h = distogram.Distogram(bin_count, weighted_diff)
    h.bins = bins
    h.min = values[0]
    h.max = values[-1]

    # noinspection PyProtectedMember
    h.diffs = distogram._compute_diffs(h)
    return h


def replace_extension(name: str, new_extension: str = None) -> str:
    """ Replaces the extension in the name of an input image with `POLUS_EXT`
     for writing corresponding output images. """
    new_extension = constants.POLUS_EXT if new_extension is None else new_extension
    return (
        name
        .replace('.ome.tif', new_extension)
        .replace('.ome.zarr', new_extension)
    )


def iter_tiles_2d(file_path: Path) -> TileIndices:
    """ A Generator of tile_indices in a 3d image.

    Args:
        file_path: Path to the image.

    Yields:
        A 4-tuple representing the coordinates of each tile.
    """
    with BioReader(file_path) as reader:
        x_end, y_end = reader.X, reader.Y

    for y_min in range(0, y_end, constants.TILE_STRIDE):
        y_max = min(y_end, y_min + constants.TILE_STRIDE)

        for x_min in range(0, x_end, constants.TILE_STRIDE):
            x_max = min(x_end, x_min + constants.TILE_STRIDE)

            yield x_min, x_max, y_min, y_max


def iter_strip(file_path: Path, index: int, axis: int) -> TileIndices:
    """ A Generator of tile_indices in the indexed strip along the given axis.

    Args:
        file_path: Path to the image.
        index: index of the current strip.
        axis: 0 for a horizontal strip, 1 for a vertical strip

    Yields:
        A 4-tuple representing the coordinates of each tile.
    """
    with BioReader(file_path) as reader:
        if axis == 0:
            x_end, y_end = reader.X, reader.Y
        else:
            x_end, y_end = reader.Y, reader.X

    num_strips = y_end // constants.TILE_STRIDE
    if y_end % constants.TILE_STRIDE != 0:
        num_strips += 1
    num_tiles_in_strip = x_end // constants.TILE_STRIDE
    if x_end % constants.TILE_STRIDE != 0:
        num_tiles_in_strip += 1

    y_min = index * constants.TILE_STRIDE
    y_max = min(y_end, y_min + constants.TILE_STRIDE)

    for i in range(num_tiles_in_strip):
        x_min = i * constants.TILE_STRIDE
        x_max = min(x_end, x_min + constants.TILE_STRIDE)

        if axis == 0:
            yield x_min, x_max, y_min, y_max
        else:
            yield y_min, y_max, x_min, x_max


def rolling_mean(values: list[float], *, prepend_zeros: bool = False) -> list[float]:
    """ Compute a rolling mean over a list of values.

    This implementation is faster than using numpy.convolve

    Args:
        values: A list of raw values.
        prepend_zeros: Whether to prepend the list with WINDOW_SIZE zeros.

    Returns:
        A list of rolling-mean values.
    """
    if prepend_zeros:
        zeros = [0.] * constants.WINDOW_SIZE
        values = zeros + values

    sums = numpy.cumsum(values)
    means = [
        abs(float(a - b)) / constants.WINDOW_SIZE
        for a, b in zip(sums[constants.WINDOW_SIZE:], sums[:-constants.WINDOW_SIZE])
    ]
    return means


def smoothed_gradients(values: list[float], *, prepend_zeros: bool = False) -> list[float]:
    """ Compute the smoothed gradients between smoothed adjacent values from the given list of values.

    This implementation is faster than using numpy.convolve

    Args:
        values: A list of raw values.
        prepend_zeros: Whether to prepend the list with WINDOW_SIZE zeros.

    Returns:
        A list of smoothed gradients.
    """
    smoothed_values = rolling_mean(values, prepend_zeros=prepend_zeros)

    raw_gradients = [
        float(a - b)
        for a, b in zip(smoothed_values[1:], smoothed_values[:-1])
    ]

    return rolling_mean(raw_gradients, prepend_zeros=prepend_zeros)


def find_spike(values: list[float], threshold: float) -> Optional[tuple[int, float]]:
    """ Returns the index and value of the first gradient that is greater than
     or equal to the given threshold. If no such gradient exists, returns None.

    Args:
        values: A list of entropy-gradient values.
        threshold: A threshold to check against

    Returns:
        If a valid value exists, a 2-tuple of index and value, otherwise None.
    """
    spikes = filter(
        lambda index_gradient: index_gradient[1] >= threshold,
        (v for v in enumerate(values)),
    )
    return next(spikes, None)


def bounding_box_superset(bounding_boxes: list[BoundingBox]) -> BoundingBox:
    """ Given a list of bounding-boxes, determine the bounding-box that bounds
     all given bounding-boxes.

    This is used to ensure that all images in a group are cropped in a
    consistent manner.

    Args:
        bounding_boxes: A list of bounding boxes.

    Returns:
        A 6-tuple of integers representing a bounding-box.
    """
    z1s, z2s, y1s, y2s, x1s, x2s = zip(*bounding_boxes)
    return min(z1s), max(z2s), min(y1s), max(y2s), min(x1s), max(x2s)
