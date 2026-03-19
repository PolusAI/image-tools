"""CLI for volumetric intensity projections (max, min, mean) on ome.tif collections."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from bfio.bfio import BioReader
from bfio.bfio import BioWriter

logger = logging.getLogger("main")

# x,y size of the 3d image chunk to be loaded into memory
TILE_SIZE = 1024

# depth of the 3d image chunk
TILE_SIZE_Z = 128


def max_min_projection(  # noqa: PLR0913
    br: BioReader,
    bw: BioWriter,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    max_workers: int,
    *,
    method: Callable[..., Any] = np.max,
) -> None:
    """Max or min intensity projection over Z for one XY tile.

    Args:
        br: Open input reader.
        bw: Open output writer.
        x_range: X bounds ``(x, x_max)``.
        y_range: Y bounds ``(y, y_max)``.
        max_workers: Worker count for bfio I/O.
        method: NumPy reducer over the Z stack (e.g. :func:`numpy.max`).
    """
    br.max_workers = max_workers
    bw.max_workers = max_workers

    # x,y range of the volume
    x, x_max = x_range
    y, y_max = y_range

    # iterate over depth
    out_image: np.ndarray | None = None
    for z in range(0, br.Z, TILE_SIZE_Z):
        z_max = min(br.Z, z + TILE_SIZE_Z)
        tile = method(br[y:y_max, x:x_max, z:z_max, 0, 0], axis=2)
        out_image = tile if out_image is None else np.dstack((out_image, tile))

    if out_image is None:
        return

    # output image
    bw[y:y_max, x:x_max, 0:1, 0, 0] = method(out_image, axis=2)


def mean_projection(
    br: BioReader,
    bw: BioWriter,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    max_workers: int,
) -> None:
    """Mean intensity projection over Z for one XY tile.

    Args:
        br: Open input reader.
        bw: Open output writer.
        x_range: X bounds ``(x, x_max)``.
        y_range: Y bounds ``(y, y_max)``.
        max_workers: Worker count for bfio I/O.
    """
    br.max_workers = max_workers
    bw.max_workers = max_workers

    # x,y range of the volume
    x, x_max = x_range
    y, y_max = y_range

    # iterate over depth
    out_image = np.zeros((y_max - y, x_max - x), dtype=np.float64)
    for z in range(0, br.Z, TILE_SIZE_Z):
        z_max = min(br.Z, z + TILE_SIZE_Z)
        out_image += np.sum(
            br[y:y_max, x:x_max, z:z_max, ...].astype(np.float64),
            axis=2,
        ).squeeze()

    # output image
    out_image /= br.Z
    bw[y:y_max, x:x_max, 0:1, 0, 0] = out_image.astype(br.dtype)


def process_image(
    input_img_path: str | Path,
    output_img_path: str | Path,
    projection: Callable[..., None],
    method: Callable[..., Any] | None,
) -> None:
    """Run the chosen projection over all XY tiles for one image pair."""
    max_workers = max(1, (os.cpu_count() or 4) // 2)

    with BioReader(input_img_path, max_workers=max_workers) as br, BioWriter(
        output_img_path,
        metadata=br.metadata,
        max_workers=max_workers,
    ) as bw:
        # output image is 2d
        bw.Z = 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for x in range(0, br.X, TILE_SIZE):
                x_max = min(br.X, x + TILE_SIZE)
                for y in range(0, br.Y, TILE_SIZE):
                    y_max = min(br.Y, y + TILE_SIZE)
                    if method is not None:
                        futures.append(
                            executor.submit(
                                projection,
                                br,
                                bw,
                                (x, x_max),
                                (y, y_max),
                                max_workers,
                                method=method,
                            ),
                        )
                    else:
                        futures.append(
                            executor.submit(
                                projection,
                                br,
                                bw,
                                (x, x_max),
                                (y, y_max),
                                max_workers,
                            ),
                        )
            for fut in futures:
                fut.result()


def run_projection(
    input_dir: str,
    output_dir: str,
    projection: Callable[..., None],
    method: Callable[..., Any] | None,
) -> None:
    """Process every ``.ome.tif`` in ``input_dir`` and write to ``output_dir``."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    input_files = [
        f.name
        for f in input_path.iterdir()
        if f.is_file() and f.name.endswith(".ome.tif")
    ]

    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_image,
                    str(input_path / image_name),
                    str(output_path / image_name),
                    projection,
                    method,
                )
                for image_name in input_files
            ]
            for fut in futures:
                fut.result()
    except Exception:  # noqa: BLE001
        # Workers may raise arbitrary image/bfio errors; log full traceback for
        # operators.
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Exiting the workflow..")


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="Calculate volumetric intensity projections",
    )

    # Input arguments
    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True,
    )
    parser.add_argument(
        "--projectionType",
        dest="projectionType",
        type=str,
        help="Type of volumetric intensity projection",
        required=True,
    )
    # Output arguments
    parser.add_argument(
        "--outDir",
        dest="outDir",
        type=str,
        help="Output collection",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.inpDir
    if Path.is_dir(Path(args.inpDir).joinpath("images")):
        input_dir = str(Path(args.inpDir).joinpath("images").absolute())
    logger.info("inpDir = %s", input_dir)
    projection_type = args.projectionType
    logger.info("projectionType = %s", projection_type)
    output_dir = args.outDir
    logger.info("outDir = %s", output_dir)

    # initialize projection function (single callable type for max/min vs mean)
    projection_fn: Callable[..., None]
    reducer: Callable[..., Any] | None
    if projection_type == "max":
        projection_fn = max_min_projection
        reducer = np.max
    elif projection_type == "min":
        projection_fn = max_min_projection
        reducer = np.min
    elif projection_type == "mean":
        projection_fn = mean_projection
        reducer = None
    else:
        logger.error("Unknown projectionType: %s", projection_type)
        sys.exit(1)

    run_projection(input_dir, output_dir, projection_fn, reducer)
