import logging
import os
from multiprocessing import cpu_count
from pathlib import Path

import numpy

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

try:
    import cupy

    NUM_GPUS = cupy.cuda.runtime.getDeviceCount()
    HAS_CUDA = NUM_GPUS > 0
except ImportError:
    NUM_GPUS = 0
    HAS_CUDA = False

NUM_THREADS = max(1, cpu_count() // 2)

TILE_SIZE = 2048
TILE_OVERLAP = 64

# Diffusion parameters
DIFF_FUEL = 1.0  # Amount of fuel to add to the flame on each iteration
DIFF_SMOLDER = 0.00001  # Heat threshold for zeroing out values every 10 iterations

# Heat shock to add to all non-zero points every 10 iterations
# Increasing this value permits heat to fill narrower regions
DIFF_SHOCK = 0.01


def replace_extension(file: Path, *, extension: str = None) -> str:
    input_extension = "".join(s for s in file.suffixes[-2:] if len(s) <= 5)
    extension = POLUS_EXT if extension is None else extension
    file_name = file.name
    if "_flow" in file_name:
        file_name = "".join(file_name.split("_flow"))
    if "_tmp" in file_name:
        file_name = "".join(file_name.split("_tmp"))
    new_name = file_name.replace(input_extension, extension)
    return new_name


def determine_dtype(num_cells: int):
    """Determines the smallest numpy.dtype for the number of cells.

    Args:
        num_cells: Total number of cells in an image

    Returns:
        The smallest dtype to use for that array
    """
    if num_cells < 2 ** 8:
        return numpy.uint8
    elif num_cells < 2 ** 16:
        return numpy.uint16
    elif num_cells < 2 ** 32:
        return numpy.uint32
    else:
        return numpy.uint64
