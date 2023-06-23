"""Helpers for basic flatfield estimation plugin."""
import concurrent.futures
import logging
import multiprocessing
import os
import pathlib
import random

import bfio
import filepattern
import numpy

__all__ = ["MAX_WORKERS", "POLUS_IMG_EXT", "POLUS_LOG", "get_image_stack"]

MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))

logger = logging.getLogger(__name__)
logger.setLevel(POLUS_LOG)


def _load_img(path: pathlib.Path, i: int) -> tuple[int, numpy.ndarray]:
    with bfio.BioReader(path, max_workers=1) as reader_:
        img = numpy.squeeze(reader_[:, :, 0, 0, 0])
    return i, img


def get_image_stack(image_paths: list[pathlib.Path]) -> numpy.ndarray:
    """Load a list of images and stack them into a single numpy array."""
    n = 1024
    if len(image_paths) > n:
        random.shuffle(image_paths)
        image_paths = image_paths[:n]

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for i, path in enumerate(image_paths):
            futures.append(executor.submit(_load_img, path, i))

        images = []
        for future in concurrent.futures.as_completed(futures):
            images.append(future.result())
    # images = [_load_img(path, i) for i, path in enumerate(image_paths)]

    images = [img for _, img in sorted(images, key=lambda x: x[0])]

    return numpy.stack(images)


def get_output_path(image_paths: list[pathlib.Path]) -> str:
    """Try to infer a filename from a list of paths."""
    # Try to infer a filename
    # noinspection PyBroadException
    try:
        pattern = filepattern.infer_pattern(files=[path.name for path in image_paths])
        fp = filepattern.FilePattern(path=str(image_paths[0].parent), pattern=pattern)
        base_output = fp.output_name()

    # Fallback to the first filename
    except Exception:  # noqa: BLE001
        base_output = image_paths[0].name

    return base_output


def get_suffix(base_output: str) -> str:
    """Extract the suffix from a filename."""
    suffix_len = 6
    return "".join(
        [
            suffix
            for suffix in pathlib.Path(base_output).suffixes[-2:]
            if len(suffix) < suffix_len
        ],
    )
