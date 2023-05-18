"""K_means clustering."""
import json
import logging
import multiprocessing
import pathlib
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Optional

import filepattern as fp
import typer

from polus.plugins.clustering.k_means import k_means as km

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.clustering.k_means")
logger.setLevel(logging.INFO)


app = typer.Typer()


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ..., "--inpDir", help="Input collection-Data need to be clustered"
    ),
    file_pattern: km.Extensions = typer.Option(
        km.Extensions.Default, "--filePattern", help="pattern to parse tabular files"
    ),
    methods: km.Methods = typer.Option(
        km.Methods.Default,
        "--methods",
        help="Select Manual or Elbow or Calinski Harabasz or Davies Bouldin method",
    ),
    minimum_range: int = typer.Option(
        ..., "--minimumRange", help="Enter minimum k-value"
    ),
    maximum_range: int = typer.Option(
        ..., "--maximumRange", help="Enter maximum k-value"
    ),
    num_of_clus: int = typer.Option(..., "--numOfClus", help="Number of clusters"),
    file_extension: km.Extensions = typer.Option(
        km.Extensions.Default, "--fileExtension", help="Output file format"
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this plugin",
    ),
) -> None:
    """K-means clustering plugin."""
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"minimumRange = {minimum_range}")
    logger.info(f"maximumRange = {maximum_range}")
    logger.info(f"numOfClus = {num_of_clus}")
    logger.info(f"outDir = {out_dir}")

    assert inp_dir.exists(), f"{inp_dir} doesnot exist!! Please check input path again"
    assert out_dir.exists(), f"{out_dir} doesnot exist!! Please check output path again"

    num_threads = max([cpu_count(), 2])

    pattern = ".*" + file_pattern
    fps = fp.FilePattern(inp_dir, pattern)

    if not fps:
        raise ValueError(f"No {file_pattern} files found.")

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": pattern,
                "outDir": [],
            }
            for file in fps:
                out_name = str(file[1][0].stem) + pattern
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)

    flist = [f[1][0] for f in fps]

    with multiprocessing.Pool(processes=num_threads) as executor:
        executor.map(
            partial(
                km.clustering,
                file_pattern=pattern,
                methods=methods,
                minimum_range=minimum_range,
                maximum_range=maximum_range,
                num_of_clus=num_of_clus,
                file_extension=file_extension,
                out_dir=out_dir,
            ),
            flist,
        )
        executor.close()
        executor.join()


if __name__ == "__main__":
    app()
