"""CLI for the pyramid_generator_3d tool."""

import logging
import os
import pathlib

import typer

from polus.images.formats.pyramid_generator_3d.pyramid_generator_3d import pyramid_generator_3d

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.formats.pyramid_generator_3d")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
app = typer.Typer()

@app.command()
def main(
    fileExtension: str = typer.Option(
        ...,
        help="None",
    ),
    filePattern: str = typer.Option(
        ,
        help="None",
    ),
    inpDir: pathlib.Path = typer.Option(
        ...,
        help="None",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
    ),
    outDir: pathlib.Path = typer.Option(
        ...,
        help="None",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
    ),
):
    """CLI for the pyramid_generator_3d tool."""
    pass


if __name__ == '__main__':
    app()
