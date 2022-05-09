import logging
import os
import pathlib
import typing

import bfio.OmeXml
import numpy
import psutil
import torch
import zarr

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))
NUM_WORKERS = max(1, os.cpu_count() // 2)

TILE_STRIDE = 1024
MEMORY_FACTOR = 0.25


def get_available_memory(devices: typing.Optional[typing.List[torch.device]]) -> float:
    if devices is None:
        memory = psutil.virtual_memory().available
    else:
        memory = sum(
            torch.cuda.get_device_properties(device).total_memory
            for device in devices
        )
    return memory * MEMORY_FACTOR


def get_zarr_path(output_dir: pathlib.Path, file_name: str) -> pathlib.Path:
    return output_dir.joinpath(file_name.replace('.tif', '.zarr'))


def init_zarr_file(reader_path: pathlib.Path, writer_path: pathlib.Path) -> bool:
    with bfio.BioReader(reader_path) as reader:
        metadata = reader.metadata

    with bfio.BioWriter(writer_path, metadata=metadata) as writer:
        writer.dtype = numpy.float32

        # noinspection PyProtectedMember
        writer._backend._init_writer()

    return True


def write_to_zarr(
        path: pathlib.Path,
        indices: typing.Tuple[int, int, int, int],
        tile: numpy.ndarray,
) -> bool:
    tile_c, tile_y, tile_x = tile.shape
    (y_min, y_max, x_min, x_max) = indices
    y_end, x_end = y_max - y_min, x_max - x_min

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(path))[0]
    # zarr ordering is (t, c, z, y, x)
    zarr_root[0:1, 0:tile_c, 0:1, y_min:y_max, x_min:x_max] = tile[:, 0:y_end, 0:x_end]

    return True
