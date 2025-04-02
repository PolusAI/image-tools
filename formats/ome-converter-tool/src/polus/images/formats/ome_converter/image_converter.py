"""Ome Converter."""
import logging
import os
import pathlib
import platform
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from itertools import product
from typing import Optional

import filepattern as fp
import numpy as np
from bfio import BioReader
from bfio import BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

TILE_SIZE = 2**13
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
NUM_THREADS_ENV = os.environ.get("NUM_THREADS")

if not NUM_THREADS_ENV or NUM_THREADS_ENV == "1":
    if platform.system().lower() == "linux":
        try:
            NUM_THREADS = len(os.sched_getaffinity(0)) // 2  # type: ignore
        except AttributeError:
            cpu_count = os.cpu_count()
            NUM_THREADS = cpu_count // 2 if cpu_count is not None else 1
    else:
        cpu_count = os.cpu_count()
        NUM_THREADS = cpu_count // 2 if cpu_count is not None else 1
else:
    NUM_THREADS = int(NUM_THREADS_ENV)  # Convert str to int safely

NUM_THREADS = max(1, NUM_THREADS)  # Ensure at least 1 thread, type is int


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


def get_num_series(inp_image: pathlib.Path) -> int:
    """Get the number of series/levels in an individual image file.

    Args:
        inp_image: Path to the input image file.

    Returns:
        int: Number of series/levels in the image.
    """
    try:
        # Open the specific file with BioReader
        br = BioReader(inp_image, max_workers=1)

        # Initialize series count
        num_series = 1
        # For formats with metadata.images, verify it's for this file only
        if hasattr(br.metadata, "images"):
            # Check if images relate to this specific file
            image_count = len(br.metadata.images)
            num_series = image_count
            logger.debug(
                f"count:{num_series} for {inp_image.name}",
            )

        # Log and return result
        logger.info(f"File {inp_image.name} has {num_series}")

        # Clean up
        br.close()
        del br

        return num_series

    except (OSError, ValueError) as e:
        logger.error(f"Error determining number of series for {inp_image}: {e!s}")
        return 1


def convert_image(
    inp_image: pathlib.Path,
    file_extension: str,
    out_dir: pathlib.Path,
) -> None:
    """Convert bioformats supported datatypes to ome.tif or ome.zarr file format."""
    # First, determine the number of series
    num_series = get_num_series(inp_image)
    logger.info(f"Detected {num_series} series/levels in image {inp_image.name}")

    # Process each series independently
    for idx in range(num_series):
        logger.info(f"Processing levels {idx} of {inp_image.name}")

        try:
            # Explicitly set the series/level parameter when opening the file
            with BioReader(inp_image, max_workers=NUM_THREADS, level=idx) as br:
                logger.debug(
                    f"Level {idx}: T={br.T}, C={br.C}, Z={br.Z}, Y={br.Y}, X={br.X}",
                )

                # Process each view (t, c, z)
                for t, c, z in product(range(br.T), range(br.C), range(br.Z)):
                    # Build the output path
                    if inp_image.suffix:
                        extension_len = 6
                        extension = "".join(
                            [
                                suffix
                                for suffix in inp_image.suffixes[-2:]
                                if len(suffix) < extension_len
                            ],
                        )
                        out_path = out_dir.joinpath(
                            inp_image.name.replace(extension, file_extension),
                        )
                    else:
                        out_path = out_dir.joinpath(f"{inp_image.name}{file_extension}")

                    # Build suffix components
                    suffix_parts = []
                    if br.C > 1:
                        suffix_parts.append(f"_c{c}")
                    if br.T > 1:
                        suffix_parts.append(f"_t{t}")
                    if br.Z > 1:
                        suffix_parts.append(f"_z{z}")
                    if num_series > 1:
                        suffix_parts.append(f"_level_{idx}")

                    # Apply combined suffix
                    if suffix_parts:
                        suffix = "".join(suffix_parts)
                        out_path = out_dir.joinpath(
                            out_path.name.replace(
                                file_extension,
                                f"{suffix}{file_extension}",
                            ),
                        )

                    # Try primary method first, then fall back to chunked approach
                    process_single_view(br, c, t, z, out_path)

        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to process series {idx} of {inp_image.name}: {e!s}",
            )


def process_single_view(
    br: "BioReader",
    c: int,
    t: int,
    z: int,
    out_path: pathlib.Path,
) -> None:
    """Process a single view (t,c,z) from a BioReader."""
    try:
        # Try direct read first
        final_image = br[:, :, z, c, t]

        write_image(
            br=br,
            c=c,
            image=final_image,
            out_path=out_path,
            max_workers=NUM_THREADS,
        )
        logger.debug(f"Successfully processed {out_path.name} using direct read")

    except (OSError, ValueError) as e:
        logger.warning(
            f"Direct read failed for {out_path.name}, trying chunked approach: {e!s}",
        )
        try:
            # Fallback to chunked approach
            actual_y, actual_x = br.Y, br.X

            if actual_y <= 0 or actual_x <= 0:
                logger.error(
                    f"Invalid dimensions: {actual_y}x{actual_x} for {out_path.name}",
                )
                return

            final_image = np.zeros((actual_y, actual_x), dtype=br.dtype)

            # Use smaller chunks for problematic images
            chunk_size = min(TILE_SIZE, min(actual_y, actual_x) // 2)

            # Track successful chunks for debugging
            total_chunks = 0
            successful_chunks = 0

            for y in range(0, actual_y, chunk_size):
                for x in range(0, actual_x, chunk_size):
                    y_max = min(actual_y, y + chunk_size)
                    x_max = min(actual_x, x + chunk_size)
                    total_chunks += 1

                    try:
                        chunk = br[y:y_max, x:x_max, z, c, t]
                        final_image[y:y_max, x:x_max] = chunk
                        successful_chunks += 1
                    except (OSError, ValueError) as e:
                        logger.warning(
                            f"Failed to read chunk at ({y}:{y_max}, {x}:{x_max}): {e}",
                        )

            logger.info(
                f"Chunked read: {successful_chunks}/{total_chunks} successfully",
            )

            write_image(
                br=br,
                c=c,
                image=final_image,
                out_path=out_path,
                max_workers=NUM_THREADS,
            )
            logger.debug(
                f"Successfully processed {out_path.name} using chunked approach",
            )

        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to process {out_path.name} with chunked approach: {e!s}",
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

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for files in fps():
            file = files[1][0]
            futures.append(executor.submit(convert_image, file, POLUS_IMG_EXT, out_dir))

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            mininterval=5,
            desc=f"converting images to {POLUS_IMG_EXT}",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            f.result()
