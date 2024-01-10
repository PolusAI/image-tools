"""A wrapper package around basicpy for use as a polus plugin."""
import logging
import pathlib

import basicpy
import bfio
import numpy

from . import utils

__all__ = ["estimate", "__version__"]

__version__ = "2.1.0"

# Set the basicpy logger to warning
logging.getLogger("basicpy.basicpy").setLevel(logging.WARNING)

# Set the plugin logger
logger = logging.getLogger(__name__)
logger.setLevel(utils.POLUS_LOG)


def estimate(
    image_paths: list[pathlib.Path],
    out_dir: pathlib.Path,
    get_darkfield: bool = False,
    extension: str = ".ome.tif",
) -> None:
    """Run BasicPy to estimate flatfield components.

    Save the flatfield and darkfield components as separate images in `out_dir`.
    The names of the output images will be generated using `filepattern`.

    Args:
        image_paths: list of paths to input images.
        out_dir: where the outputs will be written.
        get_darkfield: whether to estimate the darkfield component.
        extension: output file extension to use.
    """
    logger.info("Loading images ...")
    img_stack = utils.get_image_stack(image_paths)

    # Run basic fit
    logger.info("Beginning flatfield estimation ...")
    model = basicpy.BaSiC(
        get_darkfield=get_darkfield,
        sort_intensity=True,
        fitting_mode="approximate",
    )
    model.fit(img_stack)

    # Export the flatfield image
    base_output = utils.get_output_path(image_paths)
    suffix = utils.get_suffix(base_output)
    flatfield_out = base_output.replace(suffix, "_flatfield" + extension)
    logger.info(f"Saving flatfield: {flatfield_out} ...")

    with bfio.BioReader(image_paths[0], max_workers=2) as br:
        metadata = br.metadata

    with bfio.BioWriter(
        out_dir.joinpath(flatfield_out),
        metadata=metadata,
        max_workers=2,
    ) as bw:
        bw.dtype = numpy.float32
        bw[:] = model.flatfield

    # Export the darkfield image
    if get_darkfield:
        darkfield_out = base_output.replace(suffix, "_darkfield" + extension)
        logger.info(f"Saving darkfield: {darkfield_out} ...")

        with bfio.BioWriter(
            out_dir.joinpath(darkfield_out),
            metadata=metadata,
            max_workers=2,
        ) as bw:
            bw.dtype = numpy.float32
            bw[:] = model.darkfield

    # TODO: Add photobleach image
