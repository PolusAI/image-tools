"""Package entrypoint for the precompute_slide package."""

# Base packages
import logging
from os import environ
from pathlib import Path
from typing_extensions import Annotated
from enum import Enum
import typer
from .precompute_slide import precompute_slide
from .utils import PyramidType, ImageType

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.visualization.precompute_slide.__main__")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)

# TODO CHECK Why do we need that
POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Precompute Slide plugin.")

@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input image collection to be processed by this plugin.",
        case_sensitive=False
    ),
    filepattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filepattern of the images in input.",
        case_sensitive=False
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
        case_sensitive=False
    ),
    pyramid_type: PyramidType = typer.Option(
        PyramidType.zarr,
        "--pyramidType",
        "-p",
        help="type of pyramid.",
        case_sensitive=False
        ),
    image_type: ImageType = typer.Option(
        ImageType.image,
        "--imageType",
        "-t",
        help="type of image.",
        case_sensitive=False
        )
):
    """Precompute slide plugin command line."""

    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {filepattern}")
    logger.info(f"outDir: {out_dir}")
    logger.info(f"pyramidType: {pyramid_type}")
    logger.info(f"imageType: {image_type}")

    # TODO check how to remove this implicit conventions. Bug prone
    if inp_dir.joinpath("images").exists():
        inp_dir = inp_dir.joinpath("images")
    logger.info(f"changing input_dir to  {inp_dir}")
    if not inp_dir.exists():
        raise ValueError("inpDir does not exist", inp_dir)

    if not out_dir.exists():
        raise ValueError("outDir does not exist", out_dir)

    # Validate the pyramid type
    if pyramid_type not in ["Neuroglancer", "DeepZoom", "Zarr"]:
        msg = "pyramidType must be one of ['Neuroglancer','DeepZoom', 'Zarr']. got :"
        raise ValueError(
            msg,
            pyramid_type,
        )

    # TODO CHECK THAT
    if image_type == ImageType.segmentation and pyramid_type!= PyramidType.neuroglancer:
            raise ValueError("Segmentation type can only be used for Neuroglancer pyramids.")


    precompute_slide(inp_dir, pyramid_type, image_type, filepattern, out_dir)


if __name__ == "__main__":
    app()