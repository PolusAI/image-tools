"""FTL Label Tool."""
import logging
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy
import typer
from bfio import BioReader
from bfio import BioWriter
from ftl_rust import PolygonSet

try:
    import ftl

    FTL_CYTHON_AVAILABLE = True
except ImportError:
    ftl = None  # type: ignore[assignment]
    FTL_CYTHON_AVAILABLE = False

app = typer.Typer()

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")
_NUM_THREADS: int = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def get_output_name(filename: str) -> str:
    """Generate the output filename using the configured extension."""
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
        elif dtype == numpy.uint32 or dtype == numpy.float32:
            pixel_bytes = 32
        else:
            pixel_bytes = 64

        image_size = num_pixels * (pixel_bytes / 8)  # Convert bits to bytes
        (small_files if image_size <= threshold else large_files).append(file_path)

    return small_files, large_files


def label_cython(args: tuple[Path, Path, int, float]) -> bool | None:
    """Label a small image using the Cython implementation.

    Accepts a single tuple so it can be used directly with ThreadPool.map.

    Args:
        args: (input_path, output_path, connectivity, bin_thresh)

    Returns:
        True on success, None if the image was skipped.
    """
    input_path, output_path, connectivity, bin_thresh = args
    with BioReader(input_path, max_workers=_NUM_THREADS) as reader, BioWriter(
        output_path,
        max_workers=_NUM_THREADS,
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
            logger.warning(
                (
                    f"{input_path.name}: Connectivity is not less than or equal to "
                    f"the number of image dimensions, skipping this image. "
                    f"connectivity={connectivity}, ndim={image.ndim}"
                ),
            )
            return None

        # Run the labeling algorithm
        labels = ftl.label_nd(image, connectivity)

        # Save the image
        writer.dtype = labels.dtype
        writer[:] = labels
    return True


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collection to be processed by this plugin.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    connectivity: int = typer.Option(
        ...,
        "--connectivity",
        help="City block connectivity. Must be <= number of image dimensions.",
        min=1,
        max=3,
    ),
    binarization_threshold: float = typer.Option(
        0.5,
        "--binarizationThreshold",
        help="Binarization threshold for probability images. Must be between 0 and 1.",
        min=0.0,
        max=1.0,
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection for labelled images.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """Label objects in a 2D or 3D binary image using the FTL algorithm.

    Small images (< 500 MB) are processed with the Cython implementation via a
    ThreadPool.  Large images are processed with the Rust tiled implementation
    sequentially (the Rust layer manages its own parallelism internally).

    Args:
        inp_dir: Path to the input image collection.
        connectivity: City block connectivity (1 = face, 2 = edge, 3 = corner).
        binarization_threshold: Threshold for binarizing float probability images.
        out_dir: Path to the output image collection.
    """
    logger.info(f"inpDir                = {inp_dir}")
    logger.info(f"connectivity          = {connectivity}")
    logger.info(f"binarizationThreshold = {binarization_threshold:.2f}")
    logger.info(f"outDir                = {out_dir}")
    logger.info(f"threads               = {_NUM_THREADS}")

    # Get all file names in inpDir image collection
    if inp_dir.joinpath("images").is_dir():
        inp_dir = inp_dir / "images"

    files = [f for f in inp_dir.iterdir() if f.is_file() and f.name.endswith(POLUS_EXT)]
    if not files:
        logger.warning(f"No {POLUS_EXT} files found in {inp_dir}")
        raise typer.Exit(0)

    small_files, large_files = filter_by_size(files, 500)

    logger.info(f"Processing {len(files)} image(s) total...")
    logger.info(f"  {len(small_files)} small -> Cython / ThreadPool path")
    logger.info(f"  {len(large_files)} large -> Rust path")

    # Small files: run label_cython in a thread pool
    if small_files:
        task_args = [
            (
                infile,
                out_dir / get_output_name(infile.name),
                connectivity,
                binarization_threshold,
            )
            for infile in small_files
        ]
        with ThreadPool(processes=_NUM_THREADS) as pool:
            results = pool.map(label_cython, task_args)

        skipped = results.count(None)
        if skipped:
            logger.warning(
                (
                    f"{skipped} small image(s) were skipped "
                    " (empty or mismatched connectivity)."
                ),
            )

        # Large files: Rust handles tiling and internal parallelism
    if large_files:
        for infile in large_files:
            outfile = out_dir / get_output_name(infile.name)
            PolygonSet(connectivity, binarization_threshold).read_from(infile).write_to(
                outfile,
            )


if __name__ == "__main__":
    app()
