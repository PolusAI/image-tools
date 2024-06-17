"""Provides the ImageJ Threshold intermodes tool."""


__version__ = "0.5.0-dev0"


import logging
import os
import pathlib
import typing

import bfio
import imagej
from polus.images.segmentation.imagej_threshold_apply import ij_typing

TILE_SIZE = 2048

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

logger = logging.getLogger("polus.images.segmentation.imagej_threshold_intermodes")
logger.setLevel(POLUS_LOG)


def threshold_intermodes(
    inp_path: pathlib.Path,
    out_dir: pathlib.Path,
    ij: typing.Any = None,
) -> None:
    """Apply a threshold to an image."""
    if ij is None:
        logger.debug("Starting ImageJ...")
        ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4")
        logger.debug(f"Loaded ImageJ version: {ij.getVersion()}")

    inp_name = (inp_path.name).split(".")[0]
    out_path = out_dir / f"{inp_name}{POLUS_EXT}"
    with bfio.BioReader(inp_path) as br:
        dtype = br.dtype
        ij_type: ij_typing.IjType = ij_typing.IjType.from_dtype(dtype)

        with bfio.BioWriter(out_path, metadata=br.metadata) as bw:
            for y_min in range(0, br.Y, TILE_SIZE):
                y_max = min(y_min + TILE_SIZE, br.Y)
                for x_min in range(0, br.X, TILE_SIZE):
                    x_max = min(x_min + TILE_SIZE, br.X)
                    logger.debug(
                        f"Processing tile: ({x_min}:{x_max}, {y_min}:{y_max})",
                    )
                    tile = br[y_min:y_max, x_min:x_max, ...]
                    img = ij_type.cast_image_to_ij(ij, tile)
                    img = ij.op().threshold().intermodes(img)
                    tile = ij_type.cast_ij_to_image(ij, img)
                    bw[y_min:y_max, x_min:x_max, ...] = tile

    logger.debug(f"Thresholding complete: {out_path}")
