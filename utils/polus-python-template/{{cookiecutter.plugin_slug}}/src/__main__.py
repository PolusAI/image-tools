"""Package entrypoint for the {{cookiecutter.package_name}} package."""

# Base packages
import logging
from os import environ
from pathlib import Path

import typer

from .{{cookiecutter.package_name}} import awesome_function

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input image collection to be processed by this plugin.",
    ),
    filepattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
    )
):
    """{{cookiecutter.plugin_name}}."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {filepattern}")
    logger.info(f"outDir: {out_dir}")

    awesome_function(inp_dir, filepattern, out_dir)


typer.run(main)
