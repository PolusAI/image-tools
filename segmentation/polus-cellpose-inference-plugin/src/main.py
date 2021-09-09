import argparse
import logging
import os
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from multiprocessing import Queue
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Tuple

import cellpose
import cellpose.models
import cellpose.transforms
import filepattern
import numpy
import torch
import zarr
from bfio import BioReader
from bfio import BioWriter

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))

# Cellpose does not give you an option to set log level.
# We have to manually set the log levels to prevent Cellpose from spamming the command line.
logging.getLogger('cellpose.core').setLevel(logging.CRITICAL)
logging.getLogger('cellpose.io').setLevel(logging.CRITICAL)
logging.getLogger('cellpose.models').setLevel(logging.CRITICAL)
logging.getLogger('cellpose.transforms').setLevel(logging.CRITICAL)

# Initialize the logger
logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# Setup pytorch for multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

""" Global Settings """
TILE_SIZE_2D = 1024
TILE_SIZE_3D = 512

# TODO: Adaptively scale Overlap parameter using the ratio of physical sizes of
#  pixels in input images to the physical sizes of pixels used by Cellpose.
TILE_OVERLAP_2D = 64  # The expected object diameter should be 30 at most
TILE_OVERLAP_3D = 256

# Use a gpu if it's available
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    _devices = list()
    for _device in range(torch.cuda.device_count()):
        # TODO: On multi-gpu systems, the 0th gpu if overused. Do more testing...
        _replicates = max(
            min(
                # 3d model uses 5.1GB of GPU memory. We add 20% headroom for some flexibility.
                int(torch.cuda.get_device_properties(_device).total_memory / (1.2 * 5.1 * (10 ** 9))),
                2,
            ),
            1
        )
        _devices.extend((
            torch.device(f'cuda:{_device}') for _ in range(_replicates)
        ))

    DEVICES = Queue(len(_devices))
    list(map(DEVICES.put, _devices))
else:
    DEVICES = Queue(1)
    DEVICES.put(torch.device("cpu"))

logger.info(f'Using devices {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}')
logger.info(f'Running {DEVICES.qsize()} workers')

DIAMETER_MODES = [
    'Manual',
    'PixelSize',
    'FirstImage',
    'EveryImage',
]

PRETRAINED_MODELS = [
    'cyto',
    'cyto2',
    'nuclei',
]

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {
    'm': 10 ** -6,
    'cm': 10 ** -4,
    'mm': 10 ** -3,
    'µm': 1,
    'nm': 10 ** 3,
}

CELLPOSE_MODEL: Optional[cellpose.models.CellposeModel] = None
SIZE_MODEL: Optional[cellpose.models.SizeModel] = None


def initialize_model(pretrained_model: str):
    """ Initialize the cellpose model within a worker

    This function is designed to be run as the initializer for a
    ProcessPoolExecutor. The size model and cellpose segmentation models are
    stored inside each process as a global variable that can be called by the
    segmentation thread.

    Args:
        pretrained_model: The name or path of the model
    """
    global SIZE_MODEL
    global CELLPOSE_MODEL

    if pretrained_model in PRETRAINED_MODELS:
        model = cellpose.models.Cellpose(
            model_type=pretrained_model,
            gpu=USE_GPU,
            device=DEVICES.get(block=False),
        )
        SIZE_MODEL = model.sz
        CELLPOSE_MODEL = model.cp

    else:
        try:
            CELLPOSE_MODEL = cellpose.models.CellposeModel(
                pretrained_model=pretrained_model,
                gpu=USE_GPU,
                device=DEVICES.get(block=False),
            )
        except Exception as e:
            logger.error(f'Could not build CellposeModel from custom model {pretrained_model}.')
            raise e

    CELLPOSE_MODEL.batch_size = 8
    return


def estimate_diameter(input_path: Path):
    logger.debug(f'estimating diameter for {input_path.name}...')

    with BioReader(input_path) as reader:
        if reader.Z == 1:
            x_min = max(0, reader.X // 2 - TILE_SIZE_2D)
            x_max = min(reader.X, x_min + TILE_SIZE_2D * 2)

            y_min = max(0, reader.Y // 2 - TILE_SIZE_2D)
            y_max = min(reader.Y, y_min + TILE_SIZE_2D * 2)

            tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]
            try:
                diameter, _ = SIZE_MODEL.eval(tile, channels=[0, 0])
            except Exception as e:
                logger.error(
                    f'No default diameter available. If you are using a custom model, '
                    f'you need to have an \'eval\' method which returns a 2-tuple '
                    f'whose 0th element is the diameter to be used.'
                )
                raise e
        else:
            logger.warning(
                f'Cellpose does not provide utilities for estimating diameter in 3d images. '
                f'Using model defaults instead...'
            )
            try:
                diameter = SIZE_MODEL.diam_mean
            except Exception as e:
                logger.error(
                    f'No default diameter available. If you are using a custom model, '
                    f'please add a \'diam_mean\' property to represent the default diameter.'
                )
                raise e

    return diameter


def init_zarr_file(
        output_dir: Path,
        file_name: str,
        metadata: Any,
        ndims: int,
) -> Path:
    """ Initializes an output zarr-file using the metadata form a BioReader.

    Args:
        output_dir: Directory for storing the zarr file.
        file_name: Name of input file from which to create zarr file.
        metadata: metadata from BioReader on the input file.
        ndims: 2 for 2d image, 3 for 3d image.

    Returns:
        Path of the zarr file that was created.
    """
    out_file = output_dir.joinpath(
        file_name.replace('.ome', '_flow.ome').replace('.tif', '.zarr')
    )
    logger.debug(f'Initializing zarr output file {out_file.name}...')
    with BioWriter(out_file, metadata=metadata) as writer:
        writer.dtype = numpy.float32

        if ndims == 2:
            writer.C = 4
            writer.channel_names = ['cell_probability', 'flow_y', 'flow_x', 'labels']
        else:
            writer.C = 5
            writer.channel_names = ['cell_probability', 'flow_z', 'flow_y', 'flow_x', 'labels']

        # noinspection PyProtectedMember
        writer._backend._init_writer()
    return out_file


def segment_thread(
        input_path: Path,
        zfile: Path,
        coordinates: Tuple[int, int, int],
        diameter: float,
):
    """ Run cellpose on a tile/chunk from an image.

    This function is meant to be run inside a thread/process.

    Args:
        input_path: Path to input file.
        zfile: Path to the zarr output file.
        coordinates: x, y, z coordinates of the tile.
        diameter: Diameter, in number of pixels, of objects in image.

    Returns:
        Returns True when completed.
    """
    x, y, z = coordinates

    with BioReader(input_path) as reader:
        # Find the edges of the tile/chunk we need to read.
        ndims = 2 if reader.Z == 1 else 3
        tile_size, tile_size_z = (TILE_SIZE_2D, 1) if ndims == 2 else (TILE_SIZE_3D, TILE_SIZE_3D)
        tile_overlap = TILE_OVERLAP_2D if ndims == 2 else TILE_OVERLAP_3D

        x_min = max(0, x - tile_overlap)
        x_max = min(reader.X, x + tile_size + tile_overlap)

        y_min = max(0, y - tile_overlap)
        y_max = min(reader.Y, y + tile_size + tile_overlap)

        # Read the tile/chunk and transpose axes if needed.
        if ndims == 2:
            tile = reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze()
            logger.debug(f'segmenting 2d tile (y, x) {y_min}:{y_max}, {x_min}:{x_max}...')
        else:
            z_min = max(0, z - tile_overlap)
            z_max = min(reader.Z, z + tile_size_z + tile_overlap)

            tile = reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0].squeeze()
            tile = tile.transpose(2, 0, 1)
            logger.debug(f'segmenting 3d chunk (z, y, x) {z_min}:{z_max}, {y_min}:{y_max}, {x_min}:{x_max}...')

        # Adjust indices for overlap
        x_overlap, x_min, x_max = x - x_min, x, min(reader.X, x + tile_size)
        y_overlap, y_min, y_max = y - y_min, y, min(reader.Y, y + tile_size)
        if ndims == 2:
            z_overlap, z_min, z_max = 0, 0, 1
        else:
            z_overlap, z_min, z_max = z - z_min, z, min(reader.Z, z + tile_size_z)

    # Convert image to be cellpose-compatible
    converted_tile = cellpose.transforms.convert_image(tile, channels=[0, 0], do_3D=(ndims == 3))
    logger.debug(f'converted tile from shape {tile.shape} to {converted_tile.shape}')

    # Find rescaling factor and compute flows and probabilities
    if not diameter:
        diameter, _ = SIZE_MODEL.eval(tile, channels=[0, 0])
    rescale = CELLPOSE_MODEL.diam_mean / numpy.array(diameter, dtype=numpy.float32)

    _, [_, flows, probabilities, _], _ = CELLPOSE_MODEL.eval(
        converted_tile[numpy.newaxis, ...],
        channels=[0, 0],
        rescale=rescale,
        resample=True,
        compute_masks=False,
        do_3D=(ndims == 3),
    )
    logger.debug(f'collected probabilities of shape {probabilities.shape} and flows of shape {flows.shape}')

    # Add axes for missing 't' and 'c' dimensions
    probabilities = probabilities[numpy.newaxis, numpy.newaxis, ...]
    flows = flows[numpy.newaxis, ...]
    if ndims == 2:  # Add axes for missing 'z' dimension
        probabilities = probabilities[:, :, numpy.newaxis, ...]
        flows = flows[:, :, numpy.newaxis, ...]

    logger.debug(
        f'writing tile with (overlap, min, max) over (y, x, z):'
        f'{(y_overlap, y_min, y_max)}, {(x_overlap, x_min, x_max)}, {(z_overlap, z_min, z_max)}'
    )

    # noinspection PyTypeChecker
    zarr_root = zarr.open(str(zfile))[0]
    # zarr ordering is (t, c, z, y, x)
    zarr_root[0:1, 0:1, z_min:z_max, y_min:y_max, x_min:x_max] = probabilities[
        ...,
        z_overlap:z_max - z_min + z_overlap,
        y_overlap:y_max - y_min + y_overlap,
        x_overlap:x_max - x_min + x_overlap,
    ]

    zarr_root[0:1, 1:ndims + 1, z_min:z_max, y_min:y_max, x_min:x_max] = flows[
        ...,
        z_overlap:z_max - z_min + z_overlap,
        y_overlap:y_max - y_min + y_overlap,
        x_overlap:x_max - x_min + x_overlap,
    ]

    return True


def main(
        input_dir: Path,
        file_pattern: str,
        diameter: float,
        diameter_mode: str,
        pretrained_model: str,
        output_dir: Path,
):
    """
    Runs cellpose model inference on each image in the collection whose name
     matches the given file-pattern. Saves the cell-probabilities and flow
     fields for each image as a zarr file in the output-directory.

    Args:
        input_dir: Path to input collection.
        file_pattern: File-pattern for matching names of input images.
        diameter: Estimated diameter of objects in the image.
        diameter_mode: How to determine diameter of objects in images to let Cellpose properly resize images.
        pretrained_model: Name of builtin model (`cyto`, `cyto2`, `nuclei`) or path to custom model.
        output_dir: Path to output collection.
    """
    # Get all matching file names in the image collection
    fp = filepattern.FilePattern(input_dir, file_pattern)
    input_dir_files: list[Path] = [Path(file[0]['file']).resolve() for file in fp()]

    # Initialize the process pool
    if DEVICES.qsize() == 1:
        initialize_model(pretrained_model)
        executor = ThreadPoolExecutor(1)
    else:
        executor = ProcessPoolExecutor(
            DEVICES.qsize(),
            initializer=initialize_model,
            initargs=(pretrained_model, ),
        )

    # TODO: More testing with other diameter modes...
    # Set `diameter` for each `diameter_mode`
    if diameter_mode == 'Manual':
        if not diameter:
            logger.warning(f"""
            Manual diameter-mode specified, but manual {diameter}. 
            Using Cellpose model defaults...
            """)
            try:
                diameter = CELLPOSE_MODEL.diam_mean
            except Exception as e:
                logger.error("""
                Manual diameter selected, but no diameter supplied and model has no default diameter.
                """)
                raise e
    elif diameter_mode == 'FirstImage':
        logger.debug('Estimating diameter from the first image...')
        diameter = estimate_diameter(input_dir_files[0])
    else:  # if diameter_mode in ['PixelSize', 'EveryImage']
        logger.debug('No diameter provided...')
        diameter = 0

    # Loop through files in the input collection and create a process for each tile from each image.
    processes: list[Future[bool]] = list()
    for file_path in input_dir_files:
        with BioReader(file_path) as reader:
            ndims = 2 if reader.Z == 1 else 3
            tile_size, tile_size_z = (TILE_SIZE_2D, 1) if ndims == 2 else (TILE_SIZE_3D, TILE_SIZE_3D)

            out_file = init_zarr_file(output_dir, file_path.name, reader.metadata, ndims)

            # Determine diameter from relative pixel sizes.
            if diameter_mode == 'PixelSize':  # estimate diameter from pixel size
                x_size, y_size, z_size = reader.ps_x, reader.ps_y, reader.ps_z

                if (x_size is None) and (y_size is None) and (z_size is None):
                    raise ValueError(f'No pixel size stored in the metadata. Try using a diameterMode other than \'PixelSize\'.')
                logger.debug(f'image {file_path.name} has pixel sizes (y, x, z) {y_size}, {x_size}, {z_size}')

                size = y_size if y_size is not None else x_size if x_size is not None else z_size
                y_size = size if y_size is None else y_size
                x_size = size if x_size is None else x_size
                z_size = size if z_size is None else z_size

                # Estimate diameter based off model diam_mean and pixel size
                size = x_size[0] * UNITS[x_size[1]] + y_size[0] * UNITS[y_size[1]]
                if z_size is not None:
                    size += (z_size[0] * UNITS[z_size[1]])
                diameter = 1.5 / size

                try:
                    diameter *= CELLPOSE_MODEL.diam_mean
                except Exception as e:
                    logger.error('Custom pretrained model has no default diameter.')
                    raise e

            logger.debug(f'Processing image {file_path.name} with diameter {diameter:.3f}')
            segment_kwargs = {
                'input_path': file_path,
                'zfile': out_file,
                'coordinates': (0, 0, 0),
                'diameter': diameter,
            }

            # Iterating based on tile/chunk size...
            for z in range(0, reader.Z, tile_size_z):
                for x in range(0, reader.X, tile_size):
                    for y in range(0, reader.Y, tile_size):
                        segment_kwargs['coordinates'] = (x, y, z)
                        processes.append(executor.submit(segment_thread, **segment_kwargs))

    # Display progress and wait till finished.
    done, not_done = wait(processes, 0)
    while len(not_done) > 0:
        logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
        for r in done:
            r.result()
        done, not_done = wait(processes, 15)
    executor.shutdown()
    return


if __name__ == '__main__':
    """ Argument parsing """
    logger.info('Parsing arguments...')
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments
    parser.add_argument(
        '--inpDir',
        dest='inpDir',
        type=str,
        help='Input image collection to be processed by this plugin.',
        required=True,
    )

    parser.add_argument(
        '--diameter',
        dest='diameter',
        type=float,
        default=0,
        help='Cell diameter, if 0 cellpose will estimate for each image. Not available for 3d images.',
        required=False,
    )

    parser.add_argument(
        '--diameterMode',
        dest='diameterMode',
        type=str,
        help='Method of setting diameter. Must be one of PixelSize, Manual, FirstImage, EveryImage',
        required=True,
    )

    parser.add_argument(
        '--filePattern',
        dest='filePattern',
        type=str,
        help='File pattern for selecting files to segment.',
        required=True,
    )

    parser.add_argument(
        '--pretrainedModel',
        dest='pretrainedModel',
        type=str,
        help='Model to use',
        required=True,
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
        # switch to images folder if present
        _input_dir = _input_dir.joinpath('images')
    logger.info(f'inpDir = {_input_dir}')

    _diameter = _args.diameter
    logger.info(f'diameter = {_diameter}')

    _diameter_mode = _args.diameterMode
    # Sanity check on diameter mode
    if _diameter_mode not in DIAMETER_MODES:
        _message = f'diameterMode must be one of {DIAMETER_MODES}. Got {_diameter_mode} instead.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'diameterMode = {_diameter_mode}')

    _file_pattern = _args.filePattern
    logger.info(f'filePattern = {_file_pattern}')

    _pretrained_model = _args.pretrainedModel
    logger.info(f'pretrained_model = {_pretrained_model}')

    _output_dir = Path(_args.outDir).resolve()
    logger.info(f'outDir = {_output_dir}')

    main(
        input_dir=_input_dir,
        file_pattern=_file_pattern,
        diameter=_diameter,
        diameter_mode=_diameter_mode,
        pretrained_model=_pretrained_model,
        output_dir=_output_dir,
    )
