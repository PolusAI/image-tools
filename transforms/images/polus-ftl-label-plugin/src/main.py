import argparse
import logging
import os
from pathlib import Path

import ftl
import numpy
from bfio import BioReader
from bfio import BioWriter
from ftl_rust import PolygonSet
from preadator import ProcessManager

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")  # TODO: Figure out how to use this

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def get_output_name(filename: str) -> str:
    name = filename.split(".ome")[0]
    return f"{name}{POLUS_EXT}"


def filter_by_size(
    file_paths: list[Path],
    size_threshold: int,
) -> tuple[list[Path], list[Path]]:
    """Partitions the input files by the memory-footprint for the images.

    Args:
        file_paths: The list of files to partition.
        size_threshold: The memory-size (in MB) to use as a threshold.

    Returns:
        A 2-tuple of lists of paths.
         The first list contains small images and the second list contains large images.
    """
    small_files, large_files = [], []
    threshold: int = size_threshold * 1024 * 1024

    for file_path in file_paths:
        with BioReader(file_path) as reader:
            num_pixels = numpy.prod(reader.shape)
            dtype = reader.dtype

        if dtype in (numpy.uint8, bool):
            pixel_bytes = 8
        elif dtype == numpy.uint16:
            pixel_bytes = 16
        elif dtype == numpy.uint32 or dtype == numpy.float32:
            pixel_bytes = 32
        else:
            pixel_bytes = 64

        image_size = num_pixels * (pixel_bytes / 8)  # Convert bits to bytes
        (small_files if image_size <= threshold else large_files).append(file_path)

    return small_files, large_files


def label_cython(
    input_path: Path,
    output_path: Path,
    connectivity: int,
    bin_thresh: float,
):
    """Label the input image and writes labels back out.

    Args:
        input_path: Path to input image.
        output_path: Path for output image.
        connectivity: Connectivity kind.
        bin_thresh: Binarization threshold.
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
                image = numpy.squeeze(reader[..., 0, 0])

                # If the image has float values, binarize it using the threshold
                if image.dtype == numpy.float32 or image.dtype == numpy.float64:
                    image = (image > bin_thresh).astype(numpy.uint8)

                if not numpy.any(image):
                    writer.dtype = numpy.uint8
                    writer[:] = numpy.zeros_like(image, dtype=numpy.uint8)
                    return None

                image = image > 0
                if connectivity > image.ndim:
                    ProcessManager.log(
                        f"{input_path.name}: Connectivity is not less than or equal to the number of image dimensions, "
                        f"skipping this image. connectivity={connectivity}, ndim={image.ndim}",
                    )
                    return None

                # Run the labeling algorithm
                labels = ftl.label_nd(image, connectivity)

                # Save the image
                writer.dtype = labels.dtype
                writer[:] = labels
    return True


if __name__ == "__main__":
    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="Label objects in a 2d or 3d binary image.",
    )

    parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        required=True,
        help="Input image collection to be processed by this plugin",
    )

    parser.add_argument(
        "--connectivity",
        dest="connectivity",
        type=str,
        required=True,
        help="City block connectivity, must be less than or equal to the number of dimensions",
    )

    parser.add_argument(
        "--binarizationThreshold",
        dest="bin_thresh",
        type=str,
        required=True,
        help="For images containing probability values. Must be between 0 and 1.0.",
    )

    parser.add_argument(
        "--outDir",
        dest="outDir",
        type=str,
        required=True,
        help="Output collection",
    )

    # Parse the arguments
    args = parser.parse_args()

    _connectivity = int(args.connectivity)
    logger.info(f"connectivity = {_connectivity}")

    _bin_thresh = float(args.bin_thresh)
    assert 0 <= _bin_thresh <= 1, "bin_thresh must be between 0 and 1"
    logger.info(f"bin_thresh = {_bin_thresh:.2f}")

    _input_dir = Path(args.inpDir).resolve()
    assert _input_dir.exists(), f"{_input_dir } does not exist."
    if _input_dir.joinpath("images").is_dir():
        _input_dir = _input_dir.joinpath("images")
    logger.info(f"inpDir = {_input_dir}")

    _output_dir = Path(args.outDir).resolve()
    assert _output_dir.exists(), f"{_output_dir } does not exist."
    logger.info(f"outDir = {_output_dir}")

    # We only need a thread manager since labeling and image reading/writing
    # release the gil
    ProcessManager.init_threads()

    # Get all file names in inpDir image collection
    _files = list(
        filter(
            lambda _file: _file.is_file() and _file.name.endswith(".ome.tif"),
            _input_dir.iterdir(),
        ),
    )
    _small_files, _large_files = filter_by_size(_files, 500)

    logger.info(f"processing {len(_files)} images in total...")
    logger.info(f"processing {len(_small_files)} small images with cython...")
    logger.info(f"processing {len(_large_files)} large images with rust")

    if _small_files:
        for _infile in _small_files:
            ProcessManager.submit_thread(
                label_cython,
                _infile,
                _output_dir.joinpath(get_output_name(_infile.name)),
                _connectivity,
                _bin_thresh,
            )
        ProcessManager.join_threads()

    if _large_files:
        for _infile in _large_files:
            _outfile = _output_dir.joinpath(get_output_name(_infile.name))
            PolygonSet(_connectivity, _bin_thresh).read_from(_infile).write_to(_outfile)
