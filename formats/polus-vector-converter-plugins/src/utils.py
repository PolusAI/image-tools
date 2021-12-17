import logging
import os
from multiprocessing import cpu_count
from pathlib import Path

import numpy
import torch

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT', '.ome.tif')

USE_GPU = torch.cuda.is_available()
DEVICES = list(range(torch.cuda.device_count())) if USE_GPU else None
NUM_THREADS = len(DEVICES) if USE_GPU else max(1, int(cpu_count() * 0.5))

TILE_SIZE = 2048
TILE_OVERLAP = 64


def replace_extension(file: Path, *, extension: str = None) -> str:
    input_extension = ''.join(s for s in file.suffixes[-2:] if len(s) <= 5)
    extension = POLUS_EXT if extension is None else extension
    file_name = file.name
    if '_flow' in file_name:
        file_name = ''.join(file_name.split('_flow'))
    if '_tmp' in file_name:
        file_name = ''.join(file_name.split('_tmp'))
    new_name = file_name.replace(input_extension, extension)
    return new_name


def determine_dtype(num_cells: int):
    """ Determines the smallest numpy.dtype for the number of cells.

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
