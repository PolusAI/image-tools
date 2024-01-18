"""Hdbscan Clustering Plugin."""

import json
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Optional

import filepattern as fp
import polus.plugins.clustering.feature_subsetting.feature_subset as fs
import preadator
import typer
from tqdm import tqdm

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.clustering.feature_subsetting")
logger.setLevel(logging.INFO)


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to folder with tabular files",
    ),
    file_pattern: Optional[str] = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Pattern use to parse filenames",
    ),
    grouping_pattern: Optional[str] = typer.Option(
        None,
        "--groupingPattern",
        "-g",
        help="Regular expression to group rows to capture groups.",
    ),
    average_groups: Optional[bool] = typer.Option(
        False,
        "--averageGroups",
        "-a",
        help="Whether to average data across groups. Requires capture groups.",
    ),
    label_col: Optional[str] = typer.Option(
        None,
        "--labelCol",
        "-l",
        help="Name of column containing labels. Required only for grouping operations.",
    ),
    min_cluster_size: int = typer.Option(
        ...,
        "--minClusterSize",
        "-m",
        help="Minimum cluster size.",
    ),
    increment_outlier_id: Optional[bool] = typer.Option(
        False,
        "--incrementOutlierId",
        "-io",
        help="Increments outlier ID to 1.",
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
    """Cluster data using HDBSCAN."""
    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--filePattern = {file_pattern}")
    # Regular expression for grouping.
    logger.info(f"--groupingPattern = {grouping_pattern}")
    # Whether to average data for each group.
    logger.info(f"--averageGroups = {average_groups}")
    # Name of column to use for grouping.
    logger.info(f"--labelCol = {label_col}")
    # Minimum cluster size for clustering using HDBSCAN.
    logger.info(f"--minClusterSize = {min_cluster_size}")
    # Set outlier cluster id as 1.
    logger.info(f"--incrementOutlierId = {increment_outlier_id}")
    logger.info(f"--outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    num_workers = max([cpu_count(), 2])

    files = fp.FilePattern(inp_dir, file_pattern)

    if files is None:
        msg = f"No tabular files found. Please check {file_pattern} again"
        raise ValueError(msg)

    if preview:
        with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in files():
                out_name = file[1][0].name.replace(
                    "".join(file[1][0].suffixes),
                    f"_hdbscan{hd.POLUS_TAB_EXT}",
                )
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)
    else:
        with preadator.ProcessManager(
            name="Cluster data using HDBSCAN",
            num_processes=num_workers,
            threads_per_process=2,
        ) as pm:
            for file in tqdm(
                files(),
                total=len(files()),
                desc="Clustering data",
                mininterval=5,
                initial=0,
                unit_scale=True,
                colour="cyan",
            ):
                pm.submit_process(
                    hd.hdbscan_clustering,
                    file[1][0],
                    min_cluster_size,
                    out_dir,
                    grouping_pattern,
                    label_col,
                    average_groups,
                    increment_outlier_id,
                )
            pm.join_processes()


if __name__ == "__main__":
    app()