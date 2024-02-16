"""Outlier Removal Plugin."""

import json
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Optional

import filepattern as fp
import polus.images.clustering.outlier_removal.outlier_removal as rm
import preadator
import typer

num_workers = max([cpu_count(), 2])

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.clustering.outlier_removal")


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to folder containing tabular files",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Pattern use to parse filenames",
    ),
    method: rm.Methods = typer.Option(
        rm.Methods.DEFAULT,
        "--method",
        "-m",
        help="Select methods for outlier detection",
    ),
    output_type: rm.Outputs = typer.Option(
        rm.Outputs.DEFAULT,
        "--outputType",
        "-ot",
        help="Select type of output file",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output directory",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Remove outliers from the data."""
    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--filePattern = {file_pattern}")
    logger.info(f"--method = {method}")
    logger.info(f"--outputType = {output_type}")
    logger.info(f"--outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    files = fp.FilePattern(inp_dir, file_pattern)

    if preview:
        with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in files():
                outname = file[1][0].name.replace(
                    "".join(file[1][0].suffixes),
                    f"_{output_type}{rm.POLUS_TAB_EXT}",
                )

                out_json["outDir"].append(outname)
            json.dump(out_json, jfile, indent=2)

    else:
        with preadator.ProcessManager(
            name="Cluster data using HDBSCAN",
            num_processes=num_workers,
            threads_per_process=2,
        ) as pm:
            for file in files():
                pm.submit_process(
                    rm.outlier_detection,
                    file[1][0],
                    method,
                    output_type,
                    out_dir,
                )
            pm.join_processes()


if __name__ == "__main__":
    app()
