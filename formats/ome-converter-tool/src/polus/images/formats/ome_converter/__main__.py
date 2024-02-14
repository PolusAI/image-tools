"""Ome Converter."""
import json
import logging
import os
import pathlib
from concurrent.futures import as_completed
from typing import Any
from typing import Optional

import filepattern as fp
import preadator
import typer
from polus.images.formats.ome_converter.image_converter import NUM_THREADS
from polus.images.formats.ome_converter.image_converter import Extension
from polus.images.formats.ome_converter.image_converter import convert_image
from tqdm import tqdm

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.formats.ome_converter")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
        exists=True,
        resolve_path=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
    ),
    pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="A filepattern defining the images to be converted",
    ),
    file_extension: Extension = typer.Option(
        Extension,
        "--fileExtension",
        help="Type of data conversion",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
        exists=True,
        resolve_path=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {pattern}")
    logger.info(f"fileExtension = {file_extension}")

    fps = fp.FilePattern(inp_dir, pattern)

    if preview:
        with out_dir.joinpath("preview.json").open("w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": pattern,
                "outDir": [],
            }
            for file in fps():
                out_name = str(file[1][0].name.split(".")[0]) + file_extension
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)
        return

    with preadator.ProcessManager(
        name="ome_converter",
        num_processes=NUM_THREADS,
        threads_per_process=2,
    ) as executor:
        threads = []
        for files in fps():
            file = files[1][0]
            threads.append(
                executor.submit_process(convert_image, file, file_extension, out_dir),
            )

        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=5,
            desc=f"converting images to {file_extension}",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            f.result()


if __name__ == "__main__":
    app()
