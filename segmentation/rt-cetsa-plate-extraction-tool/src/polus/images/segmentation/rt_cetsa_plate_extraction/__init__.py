"""RT_CETSA Plate Extraction Tool."""

__version__ = "0.1.0"

import itertools
import logging
import os
import pathlib

import bfio
import numpy
import tifffile
from skimage.draw import disk
from skimage.transform import rotate

from . import core

logger = logging.getLogger("plate_extraction")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def extract_plates(
    inp_files: list[pathlib.Path],
    out_dir: pathlib.Path,
) -> None:
    """Extract all plates from RT_CETSA images.

    This will rotate the image, crop the plate, and save the cropped plates. It
    will also generate a single mask for all the plates.

    Args:
        inp_files: List of paths to the image files.
        out_dir: Path to the output directory.
    """
    logger.info("Extracting plates from RT_CETSA images ...")

    # determine rotation from the first image
    logger.info(f"Computing rotation from first image: {inp_files[0].name}")
    image = tifffile.imread(inp_files[0])
    params = core.get_plate_params(image)

    # Create a mask
    logger.info("Creating a mask ...")
    rotated_image = rotate(image, params.rotate, preserve_range=True)[
        params.bbox[0] : params.bbox[1],
        params.bbox[2] : params.bbox[3],
    ].astype(image.dtype)

    rotated_params = core.get_plate_params(rotated_image)
    mask = numpy.zeros_like(rotated_image, dtype=numpy.uint16)
    for i, (x, y) in enumerate(
        itertools.product(rotated_params.X, rotated_params.Y),
        start=1,
    ):
        rr, cc = disk((y, x), rotated_params.radii)
        mask[rr, cc] = i
    with bfio.BioWriter(out_dir / "mask.ome.tiff") as writer:
        writer.Y = mask.shape[0]
        writer.X = mask.shape[1]
        writer.Z = 1
        writer.C = 1
        writer.T = 1
        writer.dtype = mask.dtype
        writer[:] = mask

    # Extract the plates
    logger.info("Extracting plates ...")
    for i, f in enumerate(inp_files, start=1):
        logger.info(f"Processing: {i}/{len(inp_files)}: {f.name}")
        image = tifffile.imread(f)
        rotated_image = rotate(image, params.rotate, preserve_range=True)[
            params.bbox[0] : params.bbox[1],
            params.bbox[2] : params.bbox[3],
        ].astype(image.dtype)
        with bfio.BioWriter(out_dir / f"{f.stem}.ome.tiff") as writer:
            writer.Y = rotated_image.shape[0]
            writer.X = rotated_image.shape[1]
            writer.Z = 1
            writer.C = 1
            writer.T = 1
            writer.dtype = rotated_image.dtype
            writer[:] = rotated_image
