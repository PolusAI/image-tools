"""Package entrypoint for the precompute_slide package."""

# Base packages
import logging
from os import environ
from pathlib import Path

import typer

from .precompute_slide import precompute_slide
from .utils import ImageType
from .utils import PyramidType

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger("polus.plugins.visualization.precompute_slide")
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger.setLevel(POLUS_LOG)

# TODO CHECK Why do we need that
POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Precompute Slide plugin.")


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input image collection to be processed by this plugin.",
        case_sensitive=False,
    ),
    filepattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filepattern of the images in input.",
        case_sensitive=False,
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
        case_sensitive=False,
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    pyramid_type: PyramidType = typer.Option(
        ...,
        "--pyramidType",
        "-p",
        help="type of pyramid. Must be one of ['Neuroglancer','DeepZoom', 'Zarr']",
        case_sensitive=False,
    ),
    image_type: ImageType = typer.Option(
        ...,
        "--imageType",
        "-t",
        help="type of image. Must be one of ['image','segmentation']",
        case_sensitive=False,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-v",  # TODO CHECK if we want to standardize to -p or -v
        help="Preview the output without running the plugin",
        show_default=False,
    ),
) -> None:
    """Precompute slide plugin command line."""
    if isinstance(pyramid_type, str):
        pyramid_type = PyramidType[pyramid_type]

    if isinstance(image_type, str):
        image_type = ImageType[image_type]

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
        msg = "inpDir does not exist"
        raise ValueError(msg, inp_dir)

    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)

    if preview:
        logger.info("Previewing the output without running the plugin")
        msg = "Preview is not implemented yet."
        raise NotImplementedError(msg)

    # TODO CHECK THAT
    if (
        image_type == ImageType.Segmentation
        and pyramid_type != PyramidType.Neuroglancer
    ):
        msg = "Segmentation type can only be used for Neuroglancer pyramids."
        raise ValueError(msg)

    precompute_slide(inp_dir, pyramid_type, image_type, filepattern, out_dir)


if __name__ == "__main__":
    app()
