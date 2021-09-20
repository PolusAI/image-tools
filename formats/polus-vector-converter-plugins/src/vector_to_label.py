import argparse
import logging
import shutil
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import filepattern
import numpy
import zarr
from bfio import BioReader
from bfio import BioWriter

import dynamics
import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("vector-to-label")
logger.setLevel(utils.POLUS_LOG)


ThreadFuture = Tuple[Optional[numpy.ndarray], numpy.ndarray, numpy.ndarray, numpy.uint32]


def init_zarr_file(path: Path, metadata: Any):
    with BioWriter(path, metadata=metadata) as writer:
        writer.dtype = numpy.uint32
        writer.C = 1
        writer.channel_names = ['label']
        # noinspection PyProtectedMember
        writer._backend._init_writer()
    return


def reconcile_overlap(
        previous_values: numpy.ndarray,
        current_values: numpy.ndarray,
        tile: numpy.ndarray,
) -> Tuple[numpy.ndarray, list, list]:
    """ Resolve label values between tiles

    This function takes a row/column from the previous tile and a row/column
    from the current tile and finds labels that that likely match. If labels
    in the current tile should be replaced with labels from the previous tile,
    the pixels in the current tile are removed from ``tile`` and the label value
    and pixel coordinates of the label are stored in ``labels`` and ``indices``
    respectively.

    Args:
        previous_values: Previous tile edge values
        current_values: Current tile edge values
        tile: Current tile pixel values, flattened

    Returns:
        The modified tile with overlapping labels removed,
         a list of new labels, and
         a list of indices associated with the new labels.
    """
    # Get a list of unique values in the previous and current tiles
    previous_labels = numpy.unique(previous_values)
    if previous_labels[0] == 0:
        previous_labels = previous_labels[1:]

    current_labels = numpy.unique(current_values)
    if current_labels[0] == 0:
        current_labels = current_labels[1:]

    # Initialize outputs
    labels, indices = list(), list()

    if previous_labels.size != 0 and current_labels.size != 0:
        # Find overlapping indices
        for label in current_labels:

            new_labels, counts = numpy.unique(previous_values[current_values == label], return_counts=True)

            if new_labels.size == 0:
                continue

            if new_labels[0] == 0:
                new_labels = new_labels[1:]
                counts = counts[1:]

            if new_labels.size == 0:
                continue

            # Get the most frequently occurring overlapping label
            labels.append(new_labels[numpy.argmax(counts)])

            # Add indices to output, remove pixel values from the tile
            indices.append(numpy.argwhere(tile == label))
            tile[indices[-1]] = 0

    return tile, labels, indices


def vector_thread(
        in_path: Path,
        zarr_path: Path,
        coordinates: Tuple[int, int, int],
        reader_shape: Tuple[int, int, int],
        flow_error_threshold: float,
        mask_size_threshold: float,
        interpolate: bool,
        cell_probability_threshold: float,
        num_iterations: int,
        min_mask_size: int,
        device: Optional[int],
        future_z: Optional[Future],
        future_y: Optional[Future],
        future_x: Optional[Future],
) -> ThreadFuture:
    x, y, z = coordinates
    z_shape, y_shape, x_shape = reader_shape
    ndims = 2 if z_shape == 1 else 3

    # Get information from previous tiles/chunks (if there were any)
    future_z = None if future_z is None else future_z.result()[0]
    future_y = None if future_y is None else future_y.result()[1]
    future_x = None if future_x is None else future_x.result()[2]

    # Get offset to make labels consistent between tiles
    offset_z = 0 if future_z is None else numpy.max(future_z)
    offset_y = 0 if future_y is None else numpy.max(future_y)
    offset_x = 0 if future_x is None else numpy.max(future_x)
    offset = max(offset_z, offset_y, offset_x)

    x_min, x_max = max(0, x - utils.TILE_OVERLAP), min(x_shape, x + utils.TILE_SIZE + utils.TILE_OVERLAP)
    y_min, y_max = max(0, y - utils.TILE_OVERLAP), min(y_shape, y + utils.TILE_SIZE + utils.TILE_OVERLAP)
    z_min, z_max = max(0, z - utils.TILE_OVERLAP), min(z_shape, z + utils.TILE_SIZE + utils.TILE_OVERLAP)

    with BioReader(in_path) as reader:
        cell_probabilities = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0:1, 0])
        flows = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 1:ndims + 1, 0])

    # arrays are stored as (y, x, z, c, t) but need to be processed as (c, z, y, x)
    if ndims == 2:
        flows = numpy.transpose(flows, (2, 0, 1))
    else:
        cell_probabilities = numpy.transpose(cell_probabilities, (2, 0, 1))
        flows = numpy.transpose(flows, (3, 2, 0, 1))

    is_cell: numpy.ndarray = (cell_probabilities > cell_probability_threshold)

    locations = dynamics.follow_flows(
        flows=-flows * is_cell,
        num_iterations=num_iterations,
        interpolate=interpolate,
        device=device,
    )
    labels = dynamics.get_masks(
        locations=locations,
        is_cell=is_cell,
        flows=flows,
        flow_error_threshold=flow_error_threshold,
        mask_size_threshold=mask_size_threshold,
        device=device,
    )
    labels = dynamics.fill_holes_and_remove_small_masks(labels, min_mask_size)

    x_overlap, x_min, x_max = x - x_min, x, min(x_shape, x + utils.TILE_SIZE)
    y_overlap, y_min, y_max = y - y_min, y, min(y_shape, y + utils.TILE_SIZE)
    z_overlap, z_min, z_max = z - z_min, z, min(z_shape, z + utils.TILE_SIZE)

    if ndims == 2:
        labels = labels[
            y_overlap:y_max - y_min + y_overlap,
            x_overlap:x_max - x_min + x_overlap,
        ]

        current_z = None
        current_y = labels[0, :].squeeze()
        current_x = labels[:, 0].squeeze()
    else:
        labels = labels[
            z_overlap:z_max - z_min + z_overlap,
            y_overlap:y_max - y_min + y_overlap,
            x_overlap:x_max - x_min + x_overlap,
        ]

        current_z = labels[0, :, :].squeeze()
        current_y = labels[:, 0, :].squeeze()
        current_x = labels[:, :, 0].squeeze()

    shape = labels.shape
    labels = labels.reshape(-1)
    if y > 0:
        labels, labels_y, indices_y = reconcile_overlap(future_y.squeeze(), current_y, labels)
    if x > 0:
        labels, labels_x, indices_x = reconcile_overlap(future_x.squeeze(), current_x, labels)
    if z > 0:
        labels, labels_z, indices_z = reconcile_overlap(future_z.squeeze(), current_z, labels)

    uniques, labels = numpy.unique(labels, return_inverse=True)
    labels = numpy.asarray(labels, numpy.uint32)
    labels[labels > 0] = labels[labels > 0] + offset
    max_label = numpy.max(uniques) + offset

    if y > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_y, indices_y):
            if index.size == 0:
                continue
            labels[index] = label

    if x > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_x, indices_x):
            if index.size == 0:
                continue
            labels[index] = label

    if z > 0:
        # noinspection PyUnboundLocalVariable
        for label, index in zip(labels_z, indices_z):
            if index.size == 0:
                continue
            labels[index] = label

    # Zarr axes ordering should be (t, c, z, y, x). Add missing t, c, and z axes
    labels = numpy.asarray(numpy.reshape(labels, shape), dtype=numpy.uint32)
    if ndims == 2:
        labels = labels[numpy.newaxis, numpy.newaxis, numpy.newaxis, :, :]
    else:
        labels = labels[numpy.newaxis, numpy.newaxis, :, :, :]

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(zarr_path))[0]
    zarr_root[0:1, 0:1, z_min:z_max, y_min:y_max, x_min:x_max] = labels

    if ndims == 2:
        return None, labels[0, 0, 0, -1, :], labels[0, 0, 0, :, -1], max_label
    else:
        return labels[0, 0, -1, :, :], labels[0, 0, :, -1, :], labels[0, 0, :, :, -1], max_label


def zarr_to_tif(zarr_path: Path, out_path: Path):
    utils.TILE_SIZE = 1024

    with BioReader(zarr_path, max_workers=utils.NUM_THREADS) as reader:
        with BioWriter(out_path, metadata=reader.metadata, max_workers=utils.NUM_THREADS) as writer:
            writer.dtype = numpy.uint32

            for z in range(reader.Z):

                for y in range(0, reader.Y, utils.TILE_SIZE):
                    y_max = min(reader.Y, y + utils.TILE_SIZE)

                    for x in range(0, reader.X, utils.TILE_SIZE):
                        x_max = min(reader.X, x + utils.TILE_SIZE)

                        tile = reader[y:y_max, x:x_max, z:z + 1, 0, 0]
                        writer[y:y_max, x:x_max, z:z + 1, 0, 0] = tile

    shutil.rmtree(zarr_path)
    return


def vector_to_label(
        in_path: Path,
        flow_error_threshold: float,
        mask_size_threshold: float,
        interpolate: bool,
        cell_probability_threshold: float,
        num_iterations: int,
        min_mask_size: int,
        output_dir: Path,
):
    # TODO: This next line breaks in the docker container because the base pytorch container comes with python3.7
    #  Apparently, it's a classic serialization bug on thread-locks that was fixed for python3.9.
    #  Until pytorch provides a container with python3.9 or polus provides a container with python3.9 and pytorch,
    #  we are stuck with this.
    # executor = (ThreadPoolExecutor if utils.USE_GPU else ProcessPoolExecutor)(utils.NUM_THREADS)
    executor = ThreadPoolExecutor(utils.NUM_THREADS)

    with BioReader(in_path) as reader:
        reader_shape = (reader.Z, reader.Y, reader.X)
        metadata = reader.metadata

    zarr_path = output_dir.joinpath(utils.replace_extension(in_path, extension='_tmp.ome.zarr'))
    init_zarr_file(zarr_path, metadata)

    threads: Dict[tuple[int, int, int], Future[ThreadFuture]] = dict()
    thread_kwargs: Dict[str, Any] = {
        'in_path': in_path,
        'zarr_path': zarr_path,
        'coordinates': (0, 0, 0),
        'reader_shape': reader_shape,
        'flow_error_threshold': flow_error_threshold,
        'mask_size_threshold': mask_size_threshold,
        'interpolate': interpolate,
        'cell_probability_threshold': cell_probability_threshold,
        'num_iterations': num_iterations,
        'min_mask_size': min_mask_size,
        'device': None,
        'future_z': None,
        'future_y': None,
        'future_x': None,
    }

    tile_count = 0
    for z_index, z in enumerate(range(0, reader_shape[0], utils.TILE_SIZE)):
        for y_index, y in enumerate(range(0, reader_shape[1], utils.TILE_SIZE)):
            for x_index, x in enumerate(range(0, reader_shape[2], utils.TILE_SIZE)):

                device = (tile_count % utils.NUM_THREADS) if utils.USE_GPU else None
                tile_count += 1
                thread_kwargs['coordinates'] = x, y, z
                thread_kwargs['device'] = device
                thread_kwargs['future_z'] = None if z_index == 0 else threads[(z_index - 1, y_index, x_index)]
                thread_kwargs['future_y'] = None if y_index == 0 else threads[(z_index, y_index - 1, x_index)]
                thread_kwargs['future_x'] = None if x_index == 0 else threads[(z_index, y_index, x_index - 1)]

                threads[(z_index, y_index, x_index)] = executor.submit(vector_thread, **thread_kwargs)

    done, not_done = wait(threads.values(), 0)
    while len(not_done) > 0:
        logger.info(f'File {in_path.name}, Progress: {100 * len(done) / len(threads):6.3f}%')
        [future.result() for future in done]
        done, not_done = wait(threads.values(), 15)
    executor.shutdown()

    out_path = output_dir.joinpath(utils.replace_extension(zarr_path))
    zarr_to_tif(zarr_path, out_path)
    return


def main(
        input_dir: Path,
        file_pattern: str,
        flow_error_threshold: float,
        mask_size_threshold: float,
        interpolate: bool,
        cell_probability_threshold: float,
        num_iterations: int,
        min_mask_size: int,
        output_dir: Path,
):
    fp = filepattern.FilePattern(input_dir, file_pattern)
    files = [Path(files.pop()['file']).resolve() for files in fp]
    files = list(filter(
        lambda file_path: str(file_path).endswith('.ome.zarr'),
        files
    ))

    if len(files) == 0:
        logger.critical('No flow files detected.')
        return

    for i, in_path in enumerate(files, start=1):
        logger.info(f'Processing image ({i}/{len(files)}): {in_path}')
        vector_to_label(
            in_path,
            flow_error_threshold,
            mask_size_threshold,
            interpolate,
            cell_probability_threshold,
            num_iterations,
            min_mask_size,
            output_dir,
        )
    return


if __name__ == '__main__':
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='vector_to_label',
        description='Flow field calculations convert vectors to labels.',
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
        default='.+',
    )

    parser.add_argument(
        '--flowThreshold',
        dest='flowThreshold',
        type=float,
        help='Flow-error Threshold. Margin between flow-fields computed from labelled masks against input flow-fields.',
        required=False,
        default=1.0,
    )

    parser.add_argument(
        '--maskSizeThreshold',
        dest='maskSizeThreshold',
        type=float,
        help='Maximum fraction of a tile that a labelled object can cover.',
        required=False,
        default=1.0,
    )

    parser.add_argument(
        '--interpolate',
        dest='interpolate',
        type=str,
        help='Whether to use bilinear/trilinear interpolation on 2d/3d flow-fields respectively.',
        required=False,
        default='true',
    )

    parser.add_argument(
        '--cellprobThreshold',
        dest='cellprobThreshold',
        type=float,
        help='Cell Probability Threshold.',
        required=False,
        default=0.4,
    )

    parser.add_argument(
        '--numIterations',
        dest='numIterations',
        type=int,
        help='Number of iterations for which to follow flows.',
        required=False,
        default=200,  # TODO: Figure out how to set this intelligently
    )

    parser.add_argument(
        '--minObjectSize',
        dest='minObjectSize',
        type=int,
        help='Minimum number of pixels for an object to be valid. Any object with fewer pixels is removed.',
        required=False,
        default=15,
    )

    # Output arguments
    # noinspection DuplicatedCode
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
    if _input_dir.joinpath('images').is_dir():
        # switch to images folder if present
        _input_dir = _input_dir.joinpath('images')
    logger.info(f'inpDir = {_input_dir}')

    _file_pattern = _args.filePattern
    logger.info(f'filePattern = {_file_pattern}')

    _flow_error_threshold = float(_args.flowThreshold)
    if not 0. <= _flow_error_threshold <= 1.0:
        _message = f'flowThreshold must be a float between 0 and 1. Got {_flow_error_threshold} instead.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'flowThreshold = {_flow_error_threshold:.3f}')
    if _flow_error_threshold == 1.0:
        _flow_error_threshold = None

    _mask_size_threshold = _args.maskSizeThreshold
    if not 0. <= _mask_size_threshold <= 1.0:
        _message = f'maskSizeThreshold must be a float between 0 and 1. Got {_mask_size_threshold} instead.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'maskSizeThreshold = {_mask_size_threshold}')
    if _mask_size_threshold == 1.0:
        _mask_size_threshold = None

    _interpolate = (_args.interpolate == 'true')
    logger.info(f'interpolate = {_interpolate}')

    _cell_probability_threshold = _args.cellprobThreshold
    if not 0. <= _cell_probability_threshold <= 1.0:
        _message = f'cellprobThreshold must be a float between 0 and 1. Got {_cell_probability_threshold} instead.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'cellprobThreshold = {_cell_probability_threshold}')

    _num_iterations = max(_args.numIterations, 200)
    logger.info(f'numIterations = {_num_iterations}')

    _min_mask_size = max(_args.minObjectSize, 15)
    logger.info(f'minObjectSize = {_min_mask_size}')

    _output_dir = Path(_args.outDir).resolve()
    logger.info(f'outDir = {_output_dir}')

    main(
        input_dir=_input_dir,
        file_pattern=_file_pattern,
        flow_error_threshold=_flow_error_threshold,
        mask_size_threshold=_mask_size_threshold,
        interpolate=_interpolate,
        cell_probability_threshold=_cell_probability_threshold,
        num_iterations=_num_iterations,
        min_mask_size=_min_mask_size,
        output_dir=_output_dir,
    )
