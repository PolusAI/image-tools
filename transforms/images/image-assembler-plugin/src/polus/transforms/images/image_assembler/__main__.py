"""Package entrypoint for image assembly."""
# Base packages
import logging
from os import environ
from pathlib import Path

import typer
from polus.transforms.images.image_assembler.image_assembler import assemble_image

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)


def main(
    img_path: Path = typer.Option(
        ...,
        "--imgPath",
        "-i",
        help="Absolute path to the input image collection directory to be processed by this plugin.",
    ),
    stitch_path: Path = typer.Option(
        ..., "--stitchPath", "-s", help="Absolute path to a stitching vector directory."
    ),
    out_dir: Path = typer.Option(
        ..., "--outDir", "-o", help="Absolute path to the output collection directory."
    ),
    timeslice_naming: bool = typer.Option(
        False, "--timesliceNaming", "-t", help="Use timeslice number as image name."
    ),
):
    """Assemble images from a single stitching vector."""
    # if the input image collection contains a images subdirectory, we use that
    if img_path.joinpath("images").is_dir():
        img_path = img_path.joinpath("images")

    logger.info(f"imgPath: {img_path}")
    logger.info(f"stitchPath: {stitch_path}")
    logger.info(f"outDir: {out_dir}")
    logger.info(f"timesliceNaming: {timeslice_naming}")

    assemble_image(img_path, stitch_path, out_dir, timeslice_naming)


typer.run(main)
