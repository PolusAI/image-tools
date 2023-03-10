"""Ome Converter."""
import json
import os
import logging
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

import filepattern as fp
import typer
from tqdm import tqdm

from polus.plugins.formats.ome_converter.image_converter import Extension, convert_image

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.formats.ome_converter")


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
    ),
    pattern: str = typer.Option(
        ".+", "--filePattern", help="A filepattern defining the images to be converted"
    ),
    file_extension: Extension = typer.Option(
        None, "--fileExtension", help="Type of data conversion"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr file format."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {pattern}")
    logger.info(f"fileExtension = {file_extension}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"

    numworkers =  max(os.cpu_count() // 2, 1)

    fps = fp.FilePattern(inp_dir, pattern)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": pattern,
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].name.split(".")[0]) + file_extension
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    with ProcessPoolExecutor(max_workers=numworkers) as executor:
        threads = []
        for files in fps():
            file = files[1][0]
            threads.append(
                executor.submit(convert_image, file, file_extension, out_dir)
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
            time.sleep(0.2)
            f.result()


if __name__ == "__main__":
    app()
