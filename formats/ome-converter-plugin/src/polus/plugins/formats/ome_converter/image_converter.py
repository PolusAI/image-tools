"""Ome Converter."""
import enum
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from sys import platform
from typing import Optional

import filepattern as fp
from bfio import BioReader, BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

TILE_SIZE = 2**13


if platform == "linux" or platform == "linux2":
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    NUM_THREADS = max(cpu_count() // 2, 2)

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")


class Extension(str, enum.Enum):
    """Extension types to be converted."""

    OMETIF = ".ome.tif"
    OMEZARR = ".ome.zarr"
    Default = POLUS_IMG_EXT


def convert_image(
    inp_image: pathlib.Path, file_extension: Extension, out_dir: pathlib.Path
) -> None:
    """Convert bioformats supported datatypes to ome.tif or ome.zarr file format.

    Args:
        inp_image: Path of an input image.
        file_extension: Type of data conversion.
        out_dir: Path to output directory.
    """
    with BioReader(inp_image) as br:
        # Loop through timepoints
        for t in range(br.T):
            # Loop through channels
            for c in range(br.C):
                extension = "".join(
                    [suffix for suffix in inp_image.suffixes[-2:] if len(suffix) < 6]
                )

                out_path = out_dir.joinpath(
                    inp_image.name.replace(extension, file_extension)
                )
                if br.C > 1:
                    out_path = out_dir.joinpath(
                        out_path.name.replace(file_extension, f"_c{c}" + file_extension)
                    )
                if br.T > 1:
                    out_path = out_dir.joinpath(
                        out_path.name.replace(file_extension, f"_t{t}" + file_extension)
                    )

                with BioWriter(
                    out_path,
                    max_workers=NUM_THREADS,
                    metadata=br.metadata,
                ) as bw:
                    bw.C = 1
                    bw.T = 1

                    # Handling of parsing channels when channels names are not provided.
                    if not bw.channel_names == [None]:
                        bw.channel_names = [br.channel_names[c]]


                    # Loop through z-slices
                    for z in range(br.Z):
                        # Loop across the length of the image
                        for y in range(0, br.Y, TILE_SIZE):
                            y_max = min([br.Y, y + TILE_SIZE])

                            bw.max_workers = NUM_THREADS
                            br.max_workers = NUM_THREADS

                            # Loop across the depth of the image
                            for x in range(0, br.X, TILE_SIZE):
                                x_max = min([br.X, x + TILE_SIZE])
                                bw[
                                    y:y_max, x:x_max, z : z + 1, 0, 0  # noqa: E203
                                ] = br[
                                    y:y_max, x:x_max, z : z + 1, c, t  # noqa: E203
                                ]


def batch_convert(
    inp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    file_pattern: Optional[str],
    file_extension: Extension,
) -> None:
    """Convert bioformats supported datatypes in batches to ome.tif or ome.zarr file format.

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

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    file_pattern = ".+" if file_pattern is None else file_pattern

    fps = fp.FilePattern(inp_dir, file_pattern)

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        threads = []
        for files in fps():
            file = files[1][0]
            threads.append(
                executor.submit(convert_image, file, file_extension, out_dir)
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
