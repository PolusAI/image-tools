"""Utilities for the apply flatfield plugin."""

import logging
import multiprocessing
import os
import pathlib
import re
import typing

import bfio
import numpy

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)


def load_img(path: pathlib.Path, i: int) -> tuple[int, numpy.ndarray]:
    """Load image from path.

    This method is intended to be used in a thread. The index is used to
    identify the image after it has been loaded so that it images can be sorted
    in the correct order.

    Args:
        path: path to image
        i: index of image
    """
    with bfio.BioReader(path, MAX_WORKERS) as reader:
        image = reader[:, :, :, 0, 0].squeeze()
    return i, image


def save_img(
    inp_path: pathlib.Path,
    image: numpy.ndarray,
    out_dir: pathlib.Path,
    data_type: typing.Optional[bool] = False,
) -> None:
    """Save image to disk.

    Args:
        inp_path: path to input image
        image: image to be saved
        out_dir: directory to save image
        data_type: Save images in original dtype
    """
    match = re.search(r"^(.*?)\.", inp_path.name)
    if match is not None:
        name = match.group(1)
    else:
        ValueError("Unable to detect files in a directory")
    out_path = out_dir / f"{name}{POLUS_IMG_EXT}"
    with bfio.BioReader(inp_path, MAX_WORKERS) as reader, bfio.BioWriter(
        out_path,
        MAX_WORKERS,
        metadata=reader.metadata,
    ) as writer:
        if data_type:
            info_uint_type = numpy.iinfo(reader.dtype)
            scaled_image = (
                (image - numpy.min(image))
                / (numpy.max(image) - numpy.min(image))
                * int(info_uint_type.max)
            )
            uint_image = scaled_image.astype(reader.dtype)
            writer.dtype = reader.dtype
            writer[:] = uint_image
        else:
            writer.dtype = image.dtype
            writer[:] = image
