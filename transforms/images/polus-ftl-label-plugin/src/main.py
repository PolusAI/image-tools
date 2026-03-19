"""CLI for FTL label plugin: Cython path for small images, Rust for large."""
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ftl
import numpy
from bfio import BioReader
from bfio import BioWriter
from ftl_rust import PolygonSet

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
    """Strip .ome* suffix and apply POLUS_EXT for output filename."""
    name = filename.split(".ome", 1)[0]
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
    small_files: list[Path] = []
    large_files: list[Path] = []
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

        image_size = num_pixels * (pixel_bytes / 8)  # Convert bits to bytes
        (small_files if image_size <= threshold else large_files).append(file_path)

    return small_files, large_files


def label_cython(input_path: Path, output_path: Path, connectivity: int) -> bool:
    """Label the input image and writes labels back out.

    Args:
        input_path: Path to input image.
        output_path: Path for output image.
        connectivity: Connectivity kind.
    """
    max_workers = max(1, (os.cpu_count() or 4) // 2)
    with BioReader(input_path, max_workers=max_workers) as reader, BioWriter(
        output_path,
        max_workers=max_workers,
        metadata=reader.metadata,
    ) as writer:
        # Load an image and convert to binary
        image = numpy.squeeze(reader[..., 0, 0])

        if not numpy.any(image):
            writer.dtype = numpy.uint8
            writer[:] = numpy.zeros_like(image, dtype=numpy.uint8)
            return True

        image = image > 0
        if connectivity > image.ndim:
            logger.warning(
                "%s: Connectivity is not less than or equal to the number of "
                "image dimensions, skipping this image. connectivity=%s, ndim=%s",
                input_path.name,
                connectivity,
                image.ndim,
            )
            return False

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
        help=(
            "City block connectivity, must be less than or equal to the number "
            "of dimensions"
        ),
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

    _input_dir = Path(args.inpDir).resolve()
    if not _input_dir.exists():
        msg = f"{_input_dir} does not exist."
        raise FileNotFoundError(msg)
    if _input_dir.joinpath("images").is_dir():
        _input_dir = _input_dir.joinpath("images")
    logger.info(f"inpDir = {_input_dir}")

    _output_dir = Path(args.outDir).resolve()
    if not _output_dir.exists():
        msg = f"{_output_dir} does not exist."
        raise FileNotFoundError(msg)
    logger.info(f"outDir = {_output_dir}")

    # Get all file names in inpDir image collection
    _files = list(
        filter(
            lambda _file: _file.is_file() and _file.name.endswith(".ome.tif"),
            _input_dir.iterdir(),
        ),
    )
    _small_files, _large_files = filter_by_size(_files, 500)

    logger.info("processing %s images in total...", len(_files))
    logger.info("processing %s small images with cython...", len(_small_files))
    logger.info("processing %s large images with rust", len(_large_files))

    if _small_files:
        max_workers = max(1, (os.cpu_count() or 4) // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    label_cython,
                    _infile,
                    _output_dir.joinpath(get_output_name(_infile.name)),
                    _connectivity,
                )
                for _infile in _small_files
            ]
            for f in futures:
                f.result()

    if _large_files:
        for _infile in _large_files:
            _outfile = _output_dir.joinpath(get_output_name(_infile.name))
            PolygonSet(_connectivity).read_from(_infile).write_to(_outfile)
