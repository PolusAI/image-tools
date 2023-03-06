"""Ome Converter."""
import logging
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from multiprocessing import cpu_count
from typing import Optional

import filepattern as fp
from bfio import BioReader, BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)

TILE_SIZE = 2**13

num_threads = max([cpu_count() // 2, 2])


class Extension(str, Enum):
    """Extension types to be converted."""

    OMETIF = ".ome.tif"
    OMEZARR = ".ome.zarr"
    Default = "default"


def convert_image(
    inp_image: pathlib.Path, file_extension: str, out_dir: pathlib.Path
) -> None:
    """Convert bioformats supported datatypes to ome.tif or ome.zarr file format.

    Args:
        inp_image: Path of an input image.
        file_extension: Type of data conversion.
        out_dir: Path to output directory.

    Returns:
        Images with either ome.tif or ome.zarr file format.
    """
    assert file_extension in [
        ".ome.zarr",
        ".ome.tif",
    ], "Invalid fileExtension !! it should be either .ome.tif or .ome.zarr"

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
                    max_workers=num_threads,
                    metadata=br.metadata,
                ) as bw:
                    bw.C = 1
                    bw.T = 1
                    bw.channel_names = [br.channel_names[c]]

                    # Loop through z-slices
                    for z in range(br.Z):
                        # Loop across the length of the image
                        for y in range(0, br.Y, TILE_SIZE):
                            y_max = min([br.Y, y + TILE_SIZE])

                            bw.max_workers = num_threads
                            br.max_workers = num_threads

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
    file_pattern: Optional[str] = ".+",
    file_extension: Optional[str] = ".ome.tif",
) -> None:
    """Convert bioformats supported datatypes in batches to ome.tif or ome.zarr file format.

    Args:
        inp_dir: Path of an input directory.
        out_dir: Path to output directory.
        file_extension: Type of data conversion.
        file_pattern: A pattern to select image data.

    Returns:
        Images with either ome.tif or ome.zarr file format.
    """
    logger.info(f"inp_dir = {inp_dir}")
    logger.info(f"out_dir = {out_dir}")
    logger.info(f"file_pattern = {file_pattern}")
    logger.info(f"file_extension = {file_extension}")

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"

    assert file_extension in [
        ".ome.zarr",
        ".ome.tif",
    ], "Invalid fileExtension !! it should be either .ome.tif or .ome.zarr"

    numworkers = max(cpu_count() // 2, 2)

    fps = fp.FilePattern(inp_dir, file_pattern)

    with ProcessPoolExecutor(max_workers=numworkers) as executor:
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
            time.sleep(0.2)
            f.result()
