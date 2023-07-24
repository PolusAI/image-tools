"""CLI for the plugin."""

import logging
import pathlib

import typer
from polus.plugins.visualization.precompute_slide import precompute_slide
from polus.plugins.visualization.precompute_slide import utils

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.visualization.precompute_slide")
logger.setLevel(utils.POLUS_LOG)

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to folder with CZI files",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="The output directory for ome.tif files",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    pyramid_type: utils.PyramidType = typer.Option(
        ...,
        "--pyramidType",
        "-p",
        help="Build a DeepZoom or Neuroglancer pyramid",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filepattern of the images in input",
        show_default=".*",
    ),
    image_type: utils.ImageType = typer.Option(
        "image",
        "--imageType",
        "-t",
        help="Either a image or a segmentation, defaults to image",
        show_default="image",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-v",
        help="Preview the output without running the plugin",
        show_default=False,
    ),
) -> None:
    """Generate a precomputed slide for Polus Volume Viewer.

    Args:
        inp_dir: Path to folder with CZI files
        out_dir: The output directory for ome.tif files
        pyramid_type: Build a DeepZoom or Neuroglancer pyramid
        file_pattern: Filepattern of the images in input
        image_type: Either a image or a segmentation, defaults to image
        preview: Preview the output without running the plugin
    """
    if inp_dir.joinpath("images").exists():
        logger.info("Using images folder in input directory")
        inp_dir = inp_dir.joinpath("images")

    if image_type == "segmentation" and pyramid_type == "DeepZoom":
        msg = "Segmentation type cannot be used for DeepZoom pyramids."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"outDir: {out_dir}")
    logger.info(f"pyramidType: {pyramid_type.value}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"imageType: {image_type.value}")
    logger.info(f"preview: {preview}")

    if preview:
        logger.info("Previewing the output without running the plugin")
        msg = "Preview is not implemented yet."
        raise NotImplementedError(msg)

    precompute_slide(inp_dir, pyramid_type, image_type, file_pattern, out_dir)


if __name__ == "__main__":
    app()


"""
python -m polus.plugins.visualization.precompute_slide \
    --inpDir "./data/inp_dir" \
    --outDir "./data/out_dir" \
    --pyramidType "Neuroglancer" \
    --filePattern ".*" \
    --imageType "image"
"""
