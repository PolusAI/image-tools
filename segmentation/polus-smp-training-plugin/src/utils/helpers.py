import logging
import os
from pathlib import Path
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import numpy
import torch
from bfio import BioReader
from filepattern import FilePattern
from tqdm import tqdm

__all__ = [
    'POLUS_LOG',
    'TILE_STRIDE',
    'Tiles',
    'get_labels_mapping',
    'get_tiles_mapping',
    'get_device_memory',
]

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))

TILE_STRIDE = 256

# List of 5-tuples of (file-path, x_min, x_max, y_min, y_max)
Tiles = List[Tuple[Path, int, int, int, int]]


def get_labels_mapping(
        images_fp: FilePattern,
        labels_fp: Optional[FilePattern],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """ Creates a filename map between images and labels
    In the case where image filenames have different filename 
    pattern than label filenames, this function creates a map
    between the corresponding images and labels
    
    Args:
        images_fp: filepattern object for images
        labels_fp: filepattern object for labels

    Returns:
        dictionary containing mapping between image & label names
    """
    # TODO(Najib): Get this working again. This used to do many-to-one matching
    # labels_map = {
    #     file[0]['file']: labels_fp.get_matching(**{
    #         k.upper(): v
    #         for k, v in file[0].items()
    #         if k != 'file'
    #     })[0]['file']
    #     for file in images_fp()
    # }
    # image_array = numpy.zeros((len(images_fp())))
    
    image_list = list()
    image_paths = [
        Path(file[0]['file']).resolve()
        for file in images_fp()
    ]

    label_list = list()
    label_paths = [
        Path(file[0]['file']).resolve()
        for file in labels_fp()
    ]

    for image_file, label_file in tqdm(
            zip(image_paths, label_paths),
            desc=f'loading {len(image_paths)} images',
    ):
        # TODO(Najib): Use set intersection to find matching pairs and raise warning about unmatched images/labels.
        if label_file is not None:
            assert image_file.name == label_file.name

        # better to apply preprocessing here once instead of per batch.
        with BioReader(image_file) as image_reader:
            image_shape = image_reader.shape
            image_reader = image_reader[:].reshape(image_shape[:2])
            image_list.append(image_reader)

        if label_file is not None:
            with BioReader(label_file) as label_reader:
                label_shape = label_reader.shape
                label_reader = label_reader[:].reshape(label_shape[:2])
                label_list.append(label_reader)
            assert image_shape == label_shape
        else:
            label_list.append(None)

    # using numpy array is faster.
    image_array = numpy.stack(image_list, axis=0)
    label_array = numpy.stack(label_list, axis=0)

    return image_array, label_array


def iter_tiles_2d(image_path: Path) -> Generator[Tuple[Path, int, int, int, int], None, None]:
    # TODO(Najib): Open relevant PR to bfio for tile-iterators
    with BioReader(image_path) as reader:
        y_end, x_end = reader.Y, reader.X

    for y_min in range(0, y_end, TILE_STRIDE):
        y_max = min(y_end, y_min + TILE_STRIDE)
        if (y_max - y_min) != TILE_STRIDE:
            continue

        for x_min in range(0, x_end, TILE_STRIDE):
            x_max = min(x_end, x_min + TILE_STRIDE)
            if (x_max - x_min) != TILE_STRIDE:
                continue

            yield image_path, y_min, y_max, x_min, x_max


def get_tiles_mapping(image_paths: List[Path]) -> Tiles:
    """ creates a tile map for the Dataset class
    This function iterates over all the files in the input 
    collection and creates a dictionary that can be used in 
    __getitem__ function in the Dataset class. 
    
    Args:
        image_paths: The paths to the images.
        
    Returns:
        All tile mappings
    """
    tiles: Tiles = list()

    for file_name in image_paths:
        tiles.extend(iter_tiles_2d(file_name))

    return tiles


def get_device_memory(device: torch.device) -> int:
    """ Calculates the amount of memory available on the given device.
    """
    if 'cpu' in device.type:
        _, _, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        free_memory *= (1024 ** 2)
        # Use up to a quarter of RAM for CPU training
        free_memory = free_memory // 4
    else:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory

    return free_memory
