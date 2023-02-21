"""Ome Converter."""
import logging
import os
import pathlib
from typing import Optional

import filepattern as fp
import typer
from preadator import ProcessManager

from polus.plugins.formats.ome_converter.image_converter import image_converter

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.formats.ome_converter")


def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
    ),
    pattern: str = typer.Option(
        None, "--filePattern", help="A filepattern defining the images to be converted"
    ),
    file_extension: Optional[str] = typer.Option(
        None, "--fileExtension", help="Type of data conversion"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr file format."""
    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"

    FILE_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

    FILE_EXT = FILE_EXT if file_extension is None else file_extension

    ProcessManager.init_processes(name="OME Converter")

    if pattern is None:
        pattern = ".*"

    fps = fp.FilePattern(inp_dir, pattern)

    with ProcessManager.process():
        for files in fps():
            ProcessManager.submit_process(
                image_converter, pathlib.Path(files[1][0]), FILE_EXT, out_dir
            )

    ProcessManager.join_processes()


if __name__ == "__main__":
    typer.run(main)
