"""Tabular Thresholding."""
import json
import logging
import multiprocessing
import os
import pathlib
import time
from functools import partial
from multiprocessing import cpu_count
from typing import Any, List, Optional, Union

import filepattern as fp
import typer

from polus.images.transforms.tabular.tabular_thresholding import (
    tabular_thresholding as tt,
)

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("polus.images.transforms.tabular.tabular_thresholding")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

app = typer.Typer()
# Set number of processors for scalability
max_workers = max(1, cpu_count() // 2)


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input data",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Patttern to parse file names",
    ),
    neg_control: str = typer.Option(
        ...,
        "--negControl",
        help="Column name containing information of the position of non treated wells",
    ),
    pos_control: str = typer.Option(
        ...,
        "--posControl",
        help="Column name containing information of the position of wells with known treatment outcome",
    ),
    var_name: str = typer.Option(
        tt.Methods.Default, "--varName", help="Column name for computing thresholds"
    ),
    threshold_type: tt.Methods = typer.Option(
        ..., "--thresholdType", help="Name of the threshold method"
    ),
    false_positive_rate: float = typer.Option(
        0.1, "--falsePositiverate", help="False positive rate threshold value"
    ),
    num_bins: int = typer.Option(
        512, "--numBins", help="Number of Bins for otsu threshold"
    ),
    n: int = typer.Option(4, "--n", help="Number of Standard deviation"),
    out_format: tt.Extensions = typer.Option(
        tt.Extensions.Default, "--outFormat", help="Output format"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this plugin",
    ),
) -> None:
    """Calculate binary thresholds for tabular data."""
    starttime = time.time()
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"negControl = {neg_control}")
    logger.info(f"posControl = {pos_control}")
    logger.info(f"varName = {var_name}")
    logger.info(f"thresholdType = {threshold_type}")
    logger.info(f"falsePositiverate = {false_positive_rate}")
    logger.info(f"numBins = {num_bins}")
    logger.info(f"n = {n}")
    logger.info(f"outFormat = {out_format}")

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
            out_json: dict[Union[str, List], Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].name.split(".")[0]) + "_binary" + out_format
                thr_json = str(file[1][0].name.split(".")[0]) + "_thresholds.json"
                out_json["outDir"].append(out_name)
                out_json["outDir"].append(thr_json)

            json.dump(out_json, jfile, indent=2)

    num_workers = max(multiprocessing.cpu_count() // 2, 2)

    flist = [f[1][0] for f in fps]
    logger.info(f"Number of tabular files detected: {len(flist)}, filenames: {flist}")
    assert len(flist) != 0, f"No tabular file is detected: {flist}"

    with multiprocessing.Pool(processes=num_workers) as executor:
        executor.map(
            partial(
                tt.thresholding_func,
                neg_control,
                pos_control,
                var_name,
                threshold_type,
                false_positive_rate,
                num_bins,
                n,
                out_format,
                out_dir,
            ),
            flist,
        )
        executor.close()
        executor.join()

    # Deleting intermediate files from input directory
    for f in inp_dir.iterdir():
        if f.is_file() and file_pattern != ".*.hdf5":
            if f.suffix in [".hdf5", ".yaml"]:
                os.remove(f)
        else:
            if ".hdf5.hdf5" in f.name or f.suffix == ".yaml":
                os.remove(f)

    endtime = round((time.time() - starttime) / 60, 3)
    logger.info(f"Time taken to process binary threhold CSVs: {endtime} minutes!!!")
    return


if __name__ == "__main__":
    app()
