"""Ome Converter."""

import logging
import os
import pathlib
from concurrent.futures import as_completed
from itertools import product
from sys import platform
from typing import Optional

import filepattern as fp
import numpy as np
import preadator
from bfio import BioReader
from bfio import BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

TILE_SIZE = 2**13


if platform.startswith("linux"):
    NUM_THREADS = len(os.sched_getaffinity(0)) // 2  # type: ignore
else:
    NUM_THREADS = os.cpu_count()  # type: ignore


def write_image(
    br: BioReader,
    c: int,
    image: np.ndarray,
    out_path: pathlib.Path,
    max_workers: int,
) -> None:
    """Write an image to OME-TIFF or OME-ZARR format using BioFormats.

    This function converts a given image to the OME-TIFF or OME-ZARR format,
    ensuring that the data type is compatible and handling any necessary
    byte order adjustments. It utilizes the BioWriter for writing the image
    and manages channel names based on the provided BioReader.

    Args:
        br:  An instance of BioReader containing metadata for the image.
        c: The index of the channel in the image.
        image: The image data to be written.
        out_path: Path to an output image.
        max_workers: The maximum number of worker threads to use for writing.
    """
    if image.dtype == ">u2":
        image = image.byteswap().newbyteorder("<")

    with BioWriter(
        out_path,
        max_workers,
    ) as bw:
        # Handling of parsing channels when channels names are not provided.
        if bw.channel_names != [None]:
            bw.channel_names = [br.channel_names[c]]
        bw.C = 1
        bw.T = 1
        bw.Z = 1
        bw.X = image.shape[1]
        bw.Y = image.shape[0]
        bw.dtype = image.dtype
        bw[:] = image


def convert_image(
    inp_image: pathlib.Path,
    file_extension: str,
    out_dir: pathlib.Path,
) -> None:
    """Convert bioformats supported datatypes to ome.tif or ome.zarr file format.

    Args:
        inp_image: Path of an input image.
        file_extension: Type of data conversion.
        out_dir: Path to output directory.
    """
    # Loop through timepoints, channels and z-slices
    with BioReader(inp_image, max_workers=NUM_THREADS) as br:
        for t, c, z in product(range(br.T), range(br.C), range(br.Z)):
            extension = "".join(
                [
                    suffix
                    for suffix in inp_image.suffixes[-2:]
                    if len(suffix) < 6  # noqa: PLR2004
                ],
            )

            out_path = out_dir.joinpath(
                inp_image.name.replace(extension, file_extension),
            )
            if br.C > 1:
                out_path = out_dir.joinpath(
                    out_path.name.replace(
                        file_extension,
                        f"_c{c}" + file_extension,
                    ),
                )
            if br.T > 1:
                out_path = out_dir.joinpath(
                    out_path.name.replace(
                        file_extension,
                        f"_t{t}" + file_extension,
                    ),
                )

            if br.Z > 1:
                out_path = out_dir.joinpath(
                    out_path.name.replace(
                        file_extension,
                        f"_z{z}" + file_extension,
                    ),
                )

            # Process each tile in the image using itertools.product
            for y, x in product(range(0, br.Y, TILE_SIZE), range(0, br.X, TILE_SIZE)):
                y_max = min(br.Y, y + TILE_SIZE)
                x_max = min(br.X, x + TILE_SIZE)

                image = br[
                    y:y_max,
                    x:x_max,
                    z,
                    c,
                    t,
                ]
                write_image(
                    br=br,
                    c=c,
                    image=image,
                    out_path=out_path,
                    max_workers=NUM_THREADS,
                )


def batch_convert(
    inp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    file_pattern: Optional[str],
    file_extension: str,
) -> None:
    """Convert bioformats supported datatypes in batches to ome.tif or ome.zarr.

    Args:
        inp_dir: Path of an input directory.
        out_dir: Path to output directory.
        file_extension: Type of data conversion.
        file_pattern: A pattern to select image data.
    """
    logger.info(f"inp_dir = {inp_dir}")
    logger.info(f"out_dir = {out_dir}")
    logger.info(f"file_pattern = {file_pattern}")
    logger.info(f"file_extension = {file_extension}")

    file_pattern = ".+" if file_pattern is None else file_pattern

    fps = fp.FilePattern(inp_dir, file_pattern)

    with preadator.ProcessManager(
        name="ome_converter",
        num_processes=NUM_THREADS,
        threads_per_process=2,
    ) as executor:
        threads = []
        for files in fps():
            file = files[1][0]
            threads.append(
                executor.submit(convert_image, file, file_extension, out_dir),
            )

        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=5,
            desc=f"converting images to {file_extension}",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            f.result()
