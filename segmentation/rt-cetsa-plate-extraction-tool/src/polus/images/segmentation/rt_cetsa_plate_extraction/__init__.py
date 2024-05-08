"""RT_CETSA Plate Extraction Tool."""

__version__ = "0.1.0"

import itertools
import pathlib

import numpy
import tifffile
from skimage.draw import disk
from skimage.transform import rotate

from . import core


def extract_plate(file_path: pathlib.Path) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Extract plate from RT_CETSA image.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple containing the plate image and the mask.
    """
    image = tifffile.imread(file_path)
    params = core.get_plate_params(image)
    rotated_image = rotate(image, params.rotate, preserve_range=True)[
        params.bbox[0] : params.bbox[1],
        params.bbox[2] : params.bbox[3],
    ].astype(image.dtype)

    mask = numpy.zeros_like(rotated_image, dtype=numpy.uint16)
    for i, (x, y) in enumerate(itertools.product(params.X, params.Y), start=1):
        rr, cc = disk((y, x), params.radii)
        mask[rr, cc] = i

    return rotated_image, mask
