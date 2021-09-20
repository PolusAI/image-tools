import argparse
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Tuple

import filepattern
import numpy
import zarr
from bfio.bfio import BioReader
from bfio.bfio import BioWriter

import dynamics
import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("label-to-vector")
logger.setLevel(utils.POLUS_LOG)


def init_zarr_file(path: Path, ndims: int, metadata: Any):
    with BioWriter(path, metadata=metadata) as writer:
        writer.dtype = numpy.float32
        writer.C = ndims + 2
        if ndims == 2:
            writer.channel_names = ['cell_probability', 'flow_y', 'flow_x', 'labels']
        else:
            writer.channel_names = ['cell_probability', 'flow_z', 'flow_y', 'flow_x', 'labels']
        # noinspection PyProtectedMember
        writer._backend._init_writer()
    return


def flow_thread(
        file_name: Path,
        zarr_path: Path,
        coordinates: Tuple[int, int, Optional[int]],
        device: Optional[int],
) -> bool:
    x, y, z = coordinates
    ndims = 2 if z is None else 3
    z = 0 if z is None else z

    with BioReader(file_name) as reader:
        x_shape, y_shape, z_shape = reader.X, reader.Y, reader.Z

        x_min = max(0, x - utils.TILE_OVERLAP)
        x_max = min(x_shape, x + utils.TILE_SIZE + utils.TILE_OVERLAP)

        y_min = max(0, y - utils.TILE_OVERLAP)
        y_max = min(y_shape, y + utils.TILE_SIZE + utils.TILE_OVERLAP)

        z_min = max(0, z - utils.TILE_OVERLAP)
        z_max = min(z_shape, z + utils.TILE_SIZE + utils.TILE_OVERLAP)

        masks = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0])

    masks = masks if ndims == 2 else numpy.transpose(masks, (2, 0, 1))
    masks_shape = masks.shape
    if not numpy.any(masks):
        logger.debug(f'Tile (x, y, z) = {x, y, z} in file {file_name.name} has no objects. Setting flows to zero...')
        flows = numpy.zeros((ndims, *masks.shape), dtype=numpy.float32)
    else:
        # Normalize
        labels, masks = numpy.unique(masks, return_inverse=True)
        if len(labels) == 1:
            logger.debug(f'Tile (x, y, z) = {x, y, z} in file {file_name.name} has only one object.')
            masks += 1

        masks = numpy.reshape(masks, newshape=masks_shape)
        flows = dynamics.masks_to_flows(masks, device=device)
        logger.debug(f'Computed flows on tile (x, y, z) = {x, y, z} in file {file_name.name}')

    # Zarr axes ordering should be (t, c, z, y, x). Add missing t, c, and z axes
    if ndims == 2:
        masks = masks[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
        flows = flows[numpy.newaxis, :, numpy.newaxis, :, :]
    else:
        masks = masks[numpy.newaxis, numpy.newaxis, :, :]
        flows = flows[numpy.newaxis, :, :, :]

    x_overlap = x - x_min
    x_min, x_max = x, min(x_shape, x + utils.TILE_SIZE)
    cx_min, cx_max = x_overlap, x_max - x_min + x_overlap

    y_overlap = y - y_min
    y_min, y_max = y, min(y_shape, y + utils.TILE_SIZE)
    cy_min, cy_max = y_overlap, y_max - y_min + y_overlap

    z_overlap = z - z_min
    z_min, z_max = z, min(z_shape, z + utils.TILE_SIZE)
    cz_min, cz_max = z_overlap, z_max - z_min + z_overlap

    masks = masks[:, :, cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]
    flows = flows[:, :, cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(zarr_path))[0]
    zarr_root[0:1, 0:1, z_min:z_max, y_min:y_max, x_min:x_max] = numpy.asarray(masks != 0, dtype=numpy.float32)
    zarr_root[0:1, 1:ndims + 1, z_min:z_max, y_min:y_max, x_min:x_max] = flows
    zarr_root[0:1, ndims + 1:ndims + 2, z_min:z_max, y_min:y_max, x_min:x_max] = numpy.asarray(masks, dtype=numpy.float32)

    return True


def main(
        input_dir: Path,
        file_pattern: str,
        output_dir: Path,
):
    fp = filepattern.FilePattern(input_dir, file_pattern)
    files = [Path(file[0]['file']).resolve() for file in fp]
    files = list(filter(
        lambda file_path: file_path.name.endswith('.ome.tif') or file_path.name.endswith('.ome.zarr'),
        files
    ))

    executor = (ThreadPoolExecutor if utils.USE_GPU else ProcessPoolExecutor)(utils.NUM_THREADS)
    processes: list[Future[bool]] = list()

    for in_file in files:
        with BioReader(in_file) as reader:
            x_shape, y_shape, z_shape = reader.X, reader.Y, reader.Z
            metadata = reader.metadata

        ndims = 2 if z_shape == 1 else 3

        out_file = output_dir.joinpath(utils.replace_extension(in_file, extension='_flow.ome.zarr'))
        init_zarr_file(out_file, ndims, metadata)

        tile_count = 0
        for z in range(0, z_shape, utils.TILE_SIZE):
            z = None if ndims == 2 else z

            for y in range(0, y_shape, utils.TILE_SIZE):
                for x in range(0, x_shape, utils.TILE_SIZE):
                    coordinates = x, y, z
                    device = (tile_count % utils.NUM_THREADS) if utils.USE_GPU else None
                    tile_count += 1

                    # flow_thread(in_file, out_file, coordinates, device)

                    processes.append(executor.submit(
                        flow_thread,
                        in_file,
                        out_file,
                        coordinates,
                        device,
                    ))

    done, not_done = wait(processes, 0)
    while len(not_done) > 0:
        logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
        for r in done:
            r.result()
        done, not_done = wait(processes, 5)
    executor.shutdown()

    return


if __name__ == "__main__":
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='label_to_vec',
        description='Flow field calculations to convert labels to vectors.',
    )

    # Input arguments
    parser.add_argument(
        '--inpDir',
        dest='inpDir',
        type=str,
        help='Input image collection to be processed by this plugin.',
        required=True,
    )

    parser.add_argument(
        '--filePattern',
        dest='filePattern',
        type=str,
        help='Image-name pattern to use when selecting images to process.',
        required=False,
        default='.+'
    )

    # Output arguments
    # noinspection DuplicatedCode
    parser.add_argument(
        '--outDir',
        dest='outDir',
        type=str,
        help='Output collection.',
        required=True,
    )

    # Parse the arguments
    _args = parser.parse_args()

    _input_dir = Path(_args.inpDir).resolve()
    if _input_dir.joinpath('images').is_dir():
        # switch to images folder if present
        _input_dir = _input_dir.joinpath('images')
    logger.info(f'inpDir = {_input_dir}')

    _file_pattern = _args.filePattern
    logger.info(f'filePattern = {_file_pattern}')

    _output_dir = Path(_args.outDir).resolve()
    logger.info(f'outDir = {_output_dir}')

    main(
        input_dir=_input_dir,
        file_pattern=_file_pattern,
        output_dir=_output_dir,
    )
