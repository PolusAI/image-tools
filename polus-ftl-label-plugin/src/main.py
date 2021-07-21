import argparse
import logging
from pathlib import Path

import numpy
from bfio import BioReader
from bfio import BioWriter
from preadator import ProcessManager

import ftl
from ftl_rust import PolygonSet

# Initialize the logger
logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def label_cython(
        input_path: Path,
        output_path: Path,
        connectivity: int,
):
    """ Label the input image and writes labels back out.

    Args:
        input_path: Path to input image.
        output_path: Path for output image.
        connectivity: Connectivity kind.
    """
    with ProcessManager.thread() as active_threads:
        with BioReader(
            input_path,
            max_workers=active_threads.count,
        ) as reader:

            with BioWriter(
                output_path,
                max_workers=active_threads.count,
                metadata=reader.metadata,
            ) as writer:
                # Load an image and convert to binary
                image = numpy.squeeze(reader[..., 0, 0] > 0)

                if connectivity > image.ndim:
                    ProcessManager.log(
                        f'{input_path.name}: Connectivity is not less than or equal to the number of image dimensions, '
                        f'skipping this image. connectivity={connectivity}, ndim={image}'
                    )
                    return

                # Run the labeling algorithm
                labels = ftl.label_nd(image.squeeze(), connectivity)

                # Save the image
                writer.dtype = labels.dtype
                writer[:] = labels
    return True


def filter_by_size(file_paths: list[Path], size_threshold: int) -> tuple[list[Path], list[Path]]:
    """ Partitions the input files by the memory-footprint for the images.

    Args:
        file_paths: The list of files to partition.
        size_threshold: The memory-size (in MB) to use as a threshold.

    Returns:
        A 2-tuple of lists of paths.
         The first list contains small images and the second list contains large images.
    """
    small_files, large_files = list(), list()
    threshold: int = size_threshold * 1024 * 1024

    for file_path in file_paths:
        with BioReader(file_path) as reader:
            num_pixels = numpy.prod(reader.shape)
            dtype = reader.dtype

        if dtype in (numpy.uint8, bool):
            pixel_bytes = 8
        elif dtype == numpy.uint16:
            pixel_bytes = 16
        elif dtype == numpy.uint32:
            pixel_bytes = 32
        else:
            pixel_bytes = 64

        image_size = num_pixels * pixel_bytes
        (small_files if image_size <= threshold else large_files).append(file_path)

    return small_files, large_files


if __name__ == "__main__":
    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='main',
        description='Label objects in a 2d or 3d binary image.',
    )

    parser.add_argument(
        '--connectivity',
        dest='connectivity',
        type=str,
        help='City block connectivity, must be less than or equal to the number of dimensions',
        required=True,
    )

    parser.add_argument(
        '--inpDir',
        dest='inpDir',
        type=str,
        help='Input image collection to be processed by this plugin',
        required=True,
    )

    parser.add_argument(
        '--outDir',
        dest='outDir',
        type=str,
        help='Output collection',
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    _connectivity = int(args.connectivity)
    logger.info('connectivity = {}'.format(_connectivity))

    _input_dir = Path(args.inpDir).resolve()
    logger.info('inpDir = {}'.format(_input_dir))

    _output_dir = Path(args.outDir).resolve()
    logger.info('outDir = {}'.format(_output_dir))

    # We only need a thread manager since labeling and image reading/writing
    # release the gil
    ProcessManager.init_threads()

    # Get all file names in inpDir image collection
    _files = list(filter(
        lambda _file: _file.is_file() and _file.name.endswith('.ome.tif'),
        _input_dir.iterdir()
    ))
    _small_files, _large_files = filter_by_size(_files, 500)

    if _small_files:
        for _infile in _small_files:
            ProcessManager.submit_thread(
                label_cython,
                _infile,
                _output_dir.joinpath(_infile.name),
                _connectivity,
            )
        ProcessManager.join_threads()

    if _large_files:
        for _infile in _large_files:
            _outfile = _output_dir.joinpath(_infile.name)
            PolygonSet(_connectivity).read_from(_infile).write_to(_outfile)
