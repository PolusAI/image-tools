"""A wrapper package around basicpy for use as a polus plugin."""
import logging
import pathlib

import basicpy
import bfio
import numpy

from . import utils

__all__ = ["basic"]

__version__ = "2.0.0-dev19"

# Set the basicpy logger to warning
logging.getLogger("basicpy.basicpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(utils.POLUS_LOG)


def basic(
    image_paths: list[pathlib.Path],
    out_dir: pathlib.Path,
    get_darkfield: bool = False,
    extension: str = ".ome.tif",
) -> None:
    """Run BasicPy to estimate flatfield components.

    Args:
        image_paths: list of paths to input images.
        out_dir: where the outputs will be written.
        get_darkfield: whether to estimate the darkfield component.
        extension: output file extension to use.
    """
    base_output = utils.get_output_path(image_paths)

    logger.info("Loading images ...")
    img_stk = utils.get_image_stack(image_paths)

    # Run basic fit
    logger.info("Beginning flatfield estimation ...")
    b = basicpy.BaSiC(
        get_darkfield=get_darkfield,
        lambda_flatfield_coef=500,
        intensity=True,
        fitting_mode="approximate",
    )
    b.fit(img_stk)

    # Export the flatfield image
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
        bw[:] = b.flatfield

    # Export the darkfield image
    if get_darkfield:
        darkfield_out = base_output.replace(suffix, "_darkfield" + extension)
        with bfio.BioWriter(
            out_dir.joinpath(darkfield_out),
            metadata=metadata,
            max_workers=2,
        ) as bw:
            bw.dtype = numpy.float32
            bw[:] = b.darkfield
