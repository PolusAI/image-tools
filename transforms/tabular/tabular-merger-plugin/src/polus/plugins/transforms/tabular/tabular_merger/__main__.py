"""Tabular Merger."""
import json
import logging
import pathlib
import time
from typing import Any, Optional

import filepattern as fp
import typer

from polus.plugins.transforms.tabular.tabular_merger import tabular_merger as tm

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.transforms.tabular.csv_merger")


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input generic data collection to be processed by this plugin",
    ),
    file_pattern: str = typer.Option(".+", "--filePattern", help="file_pattern"),
    strip_extension: bool = typer.Option(
        False,
        "--stripExtension",
        help="Should csv be removed from the filename when indicating which file a row in a csv file came from?",
    ),
    file_extension: tm.Extension = typer.Option(
        None, "--fileExtension", help="File format of an output combined file"
    ),
    dim: tm.Dimensions = typer.Option(
        ..., "--dim", help="Perform `rows` or `columns` merging"
    ),
    same_rows: bool = typer.Option(
        False, "--sameRows", help="Only merge files with the same number of rows?"
    ),
    same_columns: bool = typer.Option(
        False, "--sameColumns", help="Merge files with common header"
    ),
    map_var: str = typer.Option(
        None, "--mapVar", help="Column name to join files column wise"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
    """Convert bioformat supported image datatypes conversion to ome.tif or ome.zarr file format."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"stripExtension = {strip_extension}")
    logger.info(f"fileExtension = {file_extension}")
    logger.info(f"dim= {dim}")
    logger.info(f"sameRows= {same_rows}")
    logger.info(f"sameColumns= {same_columns}")
    logger.info(f"mapVar= {map_var}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} doesnot exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} doesnot exists!! Please check output path again"

    # By default it ingests all input files if not file_pattern is defined
    file_pattern = ".*" + file_pattern

    fps = fp.FilePattern(inp_dir, file_pattern)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].name.split(".")[0]) + file_extension
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    inp_dir_files = [f[1][0] for f in fps]
    st_time = time.time()
    tm.merge_files(
        inp_dir_files,
        strip_extension,
        file_extension,
        dim,
        same_rows,
        same_columns,
        map_var,
        out_dir,
    )

    exec_time = time.time() - st_time
    logger.info("Execution time:", time.strftime("%H:%M:%S", time.gmtime(exec_time)))
    logger.info("Finished Merging of files!")


if __name__ == "__main__":
    app()
