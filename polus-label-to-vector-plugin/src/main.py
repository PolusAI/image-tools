import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from multiprocessing import cpu_count
from pathlib import Path

import filepattern
import numpy
import torch
import zarr
from bfio.bfio import BioReader
from bfio.bfio import BioWriter
from cellpose import dynamics

logging.getLogger('cellpose').setLevel(logging.CRITICAL)

TILE_SIZE = 2048
TILE_OVERLAP = 512


def clean_3d_flow(mask_slice, use_gpu, device) -> numpy.ndarray:
    """ Adjusts the labels in the given slice before using cellpose to calculate flows.

    Args:
        mask_slice: A 2d slice from a 3d image.
        use_gpu: Whether to use the GPU
        device: Which CUDA device to sue

    Returns:
        2d-flow for the given slice. This flow is aggregated by the calling function.
    """
    function = dynamics.masks_to_flows_gpu if use_gpu else dynamics.masks_to_flows_cpu

    if numpy.any(mask_slice):
        shape = numpy.shape(mask_slice)
        labels, mask_slice = numpy.unique(mask_slice, return_inverse=True)
        mask_slice = mask_slice.reshape(shape)
        if len(labels) == 1:
            mask_slice += 1
        flow_subset = function(mask_slice, device=device)[0]
    else:
        flow_subset = numpy.zeros_like(mask_slice, dtype=numpy.float32)

    return flow_subset


def flow_thread_3d(
        input_path: Path,
        zarr_path: Path,
        use_gpu: bool,
        device: torch.device,
        coordinates: tuple[int, int],
) -> bool:
    """ Converts labels to flows.

    This function converts labels in each tile to vector field.

    Args:
        input_path: Path of input image collection.
        zarr_path: Path where output zarr file will be saved.
        use_gpu: Path where output zarr file will be saved.
        device: Path where output zarr file will be saved.
        coordinates: (x, y) coordinates of the tile in the image.

    """

    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    flow_logger = logging.getLogger("flow_3d")
    flow_logger.setLevel(logging.DEBUG)  # TODO: Change back to INFO

    # noinspection PyTypeChecker
    root = zarr.open(str(zarr_path))[0]
    x, y = coordinates

    try:
        with BioReader(input_path) as reader:
            x_min = max(0, x - TILE_OVERLAP)
            x_max = min(reader.X, x + TILE_SIZE + TILE_OVERLAP)
            y_min = max(0, y - TILE_OVERLAP)
            y_max = min(reader.Y, y + TILE_SIZE + TILE_OVERLAP)

            tile = reader[y_min:y_max, x_min:x_max, 0:reader.Z, 0, 0].squeeze()
            tile = tile.transpose(2, 0, 1)

            if not numpy.any(tile):
                flow_logger.debug(f'got tile with no objects in file {input_path.name}')
                flow_shape = (3, reader.Z, y_max - y_min, x_max - x_min)
                flow = numpy.zeros(flow_shape, dtype=numpy.float32)
            else:
                # Normalize
                labels, masks = numpy.unique(tile, return_inverse=True)
                if len(labels) == 1:
                    flow_logger.debug(f'got tile with only one object in file {input_path.name}')
                    masks += 1

                masks = masks.reshape(reader.Z, y_max - y_min, x_max - x_min)
                _z_shape, _y_shape, _x_shape = masks.shape
                flow = numpy.zeros((3, _z_shape, _y_shape, _x_shape), numpy.float32)

                _message_left = 'Computed flows on'
                _message_right = f'in tile(y,x) {y}:{y_max} {x}:{x_max} in file {input_path.name}'
                for _z in range(_z_shape):
                    flow[[1, 2], _z, :, :] += clean_3d_flow(masks[_z, :, :], use_gpu, device)
                    flow_logger.debug(f'{_message_left} z-slice {_z}/{_z_shape} {_message_right}')

                for _y in range(_y_shape):
                    flow[[0, 2], :, _y, :] += clean_3d_flow(masks[:, _y, :], use_gpu, device)
                    flow_logger.debug(f'{_message_left} y-slice {_y}/{_y_shape} {_message_right}')

                for _x in range(_x_shape):
                    flow[[0, 1], :, :, x] += clean_3d_flow(masks[:, :, _x], use_gpu, device)
                    flow_logger.debug(f'{_message_left} x-slice {_x}/{_y_shape} {_message_right}')

                flow_logger.debug(f'Computed flows on tile(y,x) {y}:{y_max} {x}:{x_max} in file {input_path.name}')

            x_overlap = x - x_min
            x_min, x_max = x, min(reader.X, x + TILE_SIZE)

            y_overlap = y - y_min
            y_min, y_max = y, min(reader.Y, y + TILE_SIZE)

            root[0:1, 0:1, 0:reader.Z, y_min:y_max, x_min:x_max, ] = (
                tile[numpy.newaxis, numpy.newaxis, 0:reader.Z,
                     y_overlap:y_max - y_min + y_overlap,
                     x_overlap:x_max - x_min + x_overlap] > 0
            )

            root[0:1, 1:4, 0:reader.Z, y_min:y_max, x_min:x_max] = (
                flow[numpy.newaxis, :, 0:reader.Z,
                     y_overlap:y_max - y_min + y_overlap,
                     x_overlap:x_max - x_min + x_overlap]
            )

            root[0:1, 4:5, 0:reader.Z, y_min:y_max, x_min:x_max, ] = (
                tile[numpy.newaxis, numpy.newaxis, 0:reader.Z,
                     y_overlap:y_max - y_min + y_overlap,
                     x_overlap:x_max - x_min + x_overlap]
            ).astype(numpy.float32)
    except Exception as e:
        logging.error(f'failed on {input_path.name} with error {e}')
        raise e

    return True


def flow_thread_2d(
        input_path: Path,
        zarr_path: Path,
        use_gpu: bool,
        device: torch.device,
        coordinates: tuple[int, int, int],
) -> bool:
    """ Converts labels to flows.

    This function converts labels in each tile to vector field.

    Args:
        input_path: Path of input image collection.
        zarr_path: Path where output zarr file will be saved.
        use_gpu: Path where output zarr file will be saved.
        device: Path where output zarr file will be saved.
        coordinates: (x, y, z) coordinates of the tile in the image.
    """

    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    flow_logger = logging.getLogger("flow_2d")
    flow_logger.setLevel(logging.INFO)

    # noinspection PyTypeChecker
    root = zarr.open(str(zarr_path))[0]
    x, y, z = coordinates

    with BioReader(input_path) as reader:
        x_min = max([0, x - TILE_OVERLAP])
        x_max = min([reader.X, x + TILE_SIZE + TILE_OVERLAP])
        y_min = max([0, y - TILE_OVERLAP])
        y_max = min([reader.Y, y + TILE_SIZE + TILE_OVERLAP])

        tile = reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze()

        if not numpy.any(tile):
            flow_logger.debug(f'got tile with no objects in file {input_path.name}')
            flow_shape = (y_max - y_min, x_max - x_min, reader.Z, 2, 1)
            flow = numpy.zeros(flow_shape, dtype=numpy.float32)
        else:
            # Normalize
            labels, masks = numpy.unique(tile, return_inverse=True)
            if len(labels) == 1:
                flow_logger.debug(f'got tile with only one object in file {input_path.name}')
                masks += 1

            masks = masks.reshape(y_max - y_min, x_max - x_min)
            flow = dynamics.masks_to_flows(masks, use_gpu, device)[0]
            flow = flow[:, :, :, numpy.newaxis, numpy.newaxis].transpose(1, 2, 3, 0, 4)

        flow_logger.debug(f'Computed flows on tile(y,x) {y}:{y_max} {x}:{x_max} in file {input_path.name}')

        x_overlap = x - x_min
        x_min, x_max = x, min(reader.X, x + TILE_SIZE)

        y_overlap = y - y_min
        y_min, y_max = y, min(reader.Y, y + TILE_SIZE)

        zarr_ordering = (4, 3, 2, 0, 1)

        root[0:1, 0:1, 0:reader.Z, y_min:y_max, x_min:x_max, ] = (
            tile[y_overlap:y_max - y_min + y_overlap,
                 x_overlap:x_max - x_min + x_overlap,
                 numpy.newaxis, numpy.newaxis, numpy.newaxis] > 0
        ).transpose(*zarr_ordering)

        root[0:1, 1:3, 0:reader.Z, y_min:y_max, x_min:x_max] = (
            flow[y_overlap:y_max - y_min + y_overlap,
                 x_overlap:x_max - x_min + x_overlap,
                 ...]
        ).transpose(*zarr_ordering)

        root[0:1, 3:4, 0:reader.Z, y_min:y_max, x_min:x_max, ] = (
            tile[y_overlap:y_max - y_min + y_overlap,
                 x_overlap:x_max - x_min + x_overlap,
                 numpy.newaxis, numpy.newaxis, numpy.newaxis]
        ).astype(numpy.float32).transpose(*zarr_ordering)

    return True


def main(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = None,
) -> None:
    """ Turn labels into flow fields.

    Args:
        input_dir: Path to the input directory.
        output_dir: Path to the output directory.
        file_pattern: FilePattern to use for grouping images.
    """

    # Use a gpu if it's available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'Running on: {device}')

    # Determine the number of threads to run on
    num_threads = max(1, int(cpu_count() * 0.8))
    logger.info(f'Number of threads: {num_threads}')

    # Get all file names in inpDir image collection based on input pattern
    if file_pattern:
        fp = filepattern.FilePattern(str(input_dir), file_pattern)
        input_dir_files = [file[0]['file'].name for file in fp()]
        logger.info(f'Processing {(len(input_dir_files))} labels based on file pattern')
    else:
        input_dir_files = [file.name for file in input_dir.iterdir() if file.is_file()]

    # Loop through files in inpDir image collection and process
    processes = list()

    if use_gpu:
        executor = ThreadPoolExecutor(num_threads)
    else:
        executor = ProcessPoolExecutor(num_threads)

    for f in input_dir_files:
        with BioReader(Path(input_dir).joinpath(f).resolve()) as reader:
            out_file = Path(output_dir).joinpath(f.replace('.ome', '_flow.ome').replace('.tif', '.zarr')).resolve()
            ndims = 2 if reader.Z == 1 else 3

            with BioWriter(out_file, metadata=reader.metadata) as writer:
                writer.dtype = numpy.float32
                if ndims == 2:
                    writer.C = 4
                    writer.channel_names = ['cell_probability', 'x', 'y', 'labels']
                else:
                    writer.C = 5
                    writer.channel_names = ['cell_probability', 'x', 'y', 'z', 'labels']

                # noinspection PyProtectedMember
                writer._backend._init_writer()

                for x in range(0, reader.X, TILE_SIZE):
                    for y in range(0, reader.Y, TILE_SIZE):
                        if ndims == 2:
                            for z in range(reader.Z):
                                processes.append(executor.submit(
                                    flow_thread_2d,
                                    Path(input_dir).joinpath(f).resolve(),
                                    out_file,
                                    use_gpu,
                                    device,
                                    (x, y, z),
                                ))
                        else:
                            processes.append(executor.submit(
                                flow_thread_3d,
                                Path(input_dir).joinpath(f).resolve(),
                                out_file,
                                use_gpu,
                                device,
                                (x, y),
                            ))

    done, not_done = wait(processes, 0)
    while len(not_done) > 0:
        logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
        for r in done:
            r.result()
        done, not_done = wait(processes, 5)
    executor.shutdown()


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(
        format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
    )
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments
    parser.add_argument(
        '--inpDir',
        dest='inpDir',
        type=str,
        help='Input image collection to be processed by this plugin',
        required=True,
    )

    parser.add_argument(
        '--filePattern',
        dest='filePattern',
        type=str,
        help='Input file name pattern.',
        required=False,
    )

    # Output arguments
    parser.add_argument(
        '--outDir',
        dest='outDir',
        type=str,
        help='Output collection',
        required=True,
    )

    # Parse the arguments
    _args = parser.parse_args()

    _input_dir = Path(_args.inpDir).resolve()
    if Path.is_dir(_input_dir.joinpath('images')):
        # Switch to images folder if present
        _input_dir = _input_dir.joinpath('images')
    logger.info('inpDir = {}'.format(_input_dir))

    _file_pattern = _args.filePattern
    logger.info('File pattern = {}'.format(_file_pattern))

    _output_dir = Path(_args.outDir).resolve()
    logger.info('outDir = {}'.format(_output_dir))

    main(
        _input_dir,
        _output_dir,
        _file_pattern,
    )
