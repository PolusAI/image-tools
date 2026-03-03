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
from bfio import BioReader
from bfio import BioWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

TILE_SIZE = 2**13
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")


def get_num_threads() -> int:
    """Determine default number of threads (half CPU cores, min 1)."""
    # Allow environment override
    if "NUM_THREADS" in os.environ:
        return max(1, int(os.environ["NUM_THREADS"]))
    try:
        if platform.system().lower() == "linux" and hasattr(os, "sched_getaffinity"):
            cpu_count = len(os.sched_getaffinity(0))
        else:
            cpu_count = os.cpu_count() or 1  # fallback if None
        return max(1, cpu_count // 2)
    except Exception:
        return 1

NUM_THREADS: int = get_num_threads()

# NUM_WORKERS: default to 1 process
NUM_WORKERS = max(1, int(os.environ.get("NUM_WORKERS", "1")))


def get_num_series(inp_image: pathlib.Path) -> int:
    """Get the number of series/levels in an individual image file.

    Args:
        inp_image: Path to the input image file.

    Returns:
        int: Number of series/levels in the image.
    """
    try:
        # Open the specific file with BioReader
        br = BioReader(inp_image, max_workers=NUM_THREADS)

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


def convert_image( # noqa:C901,PLR0912
    inp_image: pathlib.Path,
    file_extension: str,
    out_dir: pathlib.Path,
) -> None:
    """Convert bioformats supported datatypes to ome.tif or ome.zarr file format."""
    # First, determine the number of series
    num_series = get_num_series(inp_image)

    logger.info(f"Detected {num_series} series/levels in image {inp_image.name}")

    files_written = 0
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
                        suffix_parts.append(f"_s{idx}")

                    # Apply combined suffix
                    if suffix_parts:
                        suffix = "".join(suffix_parts)
                        out_path = out_dir.joinpath(
                            out_path.name.replace(
                                file_extension,
                                f"{suffix}{file_extension}",
                            ),
                        )

                    with BioWriter(
                        out_path,
                        max_workers=NUM_THREADS,
                        metadata=br.metadata,
                    ) as bw:
                        bw.C = 1
                        bw.T = 1
                        bw.Z = 1
                        if bw.channel_names != [None]:
                            bw.channel_names = [br.channel_names[c]]

                        for y in range(0, br.Y, TILE_SIZE):
                            y_max = min(br.Y, y + TILE_SIZE)
                            for x in range(0, br.X, TILE_SIZE):
                                x_max = min(br.X, x + TILE_SIZE)
                                bw[
                                    y:y_max,
                                    x:x_max,
                                    0,
                                    0,
                                    0,
                                ] = br[
                                    y:y_max,
                                    x:x_max,
                                    z : z + 1,
                                    c,
                                    t,
                                ]

                    files_written += 1
                    logger.debug(f"Written: {out_path.name}")

        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to process series {idx} of {inp_image.name}: {e!s}",
            )


def batch_convert(
    inp_dir: pathlib.Path | list[pathlib.Path],
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

    if isinstance(inp_dir, pathlib.Path):
        file_pattern = ".+" if file_pattern is None else file_pattern
        fps = fp.FilePattern(inp_dir, file_pattern)
        files = [files[1][0] for files in fps()]
    else:
        files = list(inp_dir)

    if not files:
        logger.warning("No files found to process.")
        return

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
    ) as executor:
        futures = [
            executor.submit(convert_image, file, POLUS_IMG_EXT, out_dir)
            for file in files
        ]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            mininterval=5,
            desc=f"converting images to {POLUS_IMG_EXT}",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            try:
                f.result()
            except (OSError, ValueError) as e:
                logger.error(f"Failed to convert file: {e}")
