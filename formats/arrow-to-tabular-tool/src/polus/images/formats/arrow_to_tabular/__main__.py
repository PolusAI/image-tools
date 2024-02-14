"""Arrow to Tabular."""
import json
import logging
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Optional

import filepattern as fp
import typer
from tqdm import tqdm

from polus.images.formats.arrow_to_tabular.arrow_to_tabular import (
    Format,
    arrow_tabular,
)

# Set number of processors for scalability
max_workers = max(1, cpu_count() // 2)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.formats.arrow_to_tabular")


def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input data",
    ),
    file_format: Format = typer.Option(
        None, "--fileFormat", help="Filepattern of desired tabular output file"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this plugin",
    ),
) -> None:
    """Execute Main function."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"fileFormat = {file_format}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"
    FILE_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

    if file_format == Format.Default:
        file_format = FILE_EXT
    elif file_format == Format.CSV:
        file_format = ".csv"
    elif file_format == Format.PARQUET:
        file_format = ".parquet"
    elif file_format == None:
        file_format = FILE_EXT

    assert file_format in [
        ".csv",
        ".parquet",
    ], f"This tabular file format: {file_format} is not support supported by this plugin!! Choose either CSV or Parquet FileFormat"

    pattern_list = [".feather", ".arrow"]
    pattern = [f.suffix for f in inp_dir.iterdir() if f.suffix in pattern_list][0]
    assert (
        pattern in pattern_list
    ), f"This input file extension {pattern} is not support supported by this plugin!! It should be either .feather and .arrow files"
    filepattern = {".feather": ".*.feather", ".arrow": ".*.arrow"}

    featherPattern = filepattern[pattern]

    fps = fp.FilePattern(inp_dir, featherPattern)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": featherPattern,
                "outDir": [],
            }
            for file in fps():
                out_name = str(file[1][0].stem) + file_format
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    with ProcessPoolExecutor(max_workers) as executor:
        processes = []
        for files in fps:
            file = files[1][0]
            processes.append(executor.submit(arrow_tabular, file, file_format, out_dir))

        for process in tqdm(
            as_completed(processes), desc="Arrow --> Tabular", total=len(processes)
        ):
            process.result()

    logger.info("Finished all processes!")


if __name__ == "__main__":
    typer.run(main)
