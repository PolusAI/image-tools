"""Feature Subsetting Plugin."""
import logging
import shutil
from pathlib import Path
from typing import Optional

import polus.plugins.clustering.feature_subsetting.feature_subset as fs
import typer

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.clustering.feature_subsetting")
logger.setLevel(logging.INFO)


def generate_preview(
    out_dir: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    shutil.copy(
        Path(__file__).parents[4].joinpath("example/summary.txt"),
        out_dir,
    )


@app.command()
def main(  # noqa: PLR0913
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to the collection of input images.",
    ),
    tabular_dir: Path = typer.Option(
        ...,
        "--tabularDir",
        "-t",
        help="Path to the collection of tabular files containing features.",
    ),
    file_pattern: Optional[str] = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Pattern use to parse filenames",
    ),
    image_feature: str = typer.Option(
        None,
        "--imageFeature",
        "-if",
        help="Image filenames feature in tabular data.",
    ),
    tabular_feature: str = typer.Option(
        None,
        "--tabularFeature",
        "-tf",
        help="Select tabular feature to subset data.",
    ),
    padding: Optional[int] = typer.Option(
        0,
        "--padding",
        "-p",
        help="Number of images to capture outside the cutoff.",
    ),
    group_var: str = typer.Option(
        ...,
        "--groupVar",
        "-g",
        help="variables to group by in a section.",
    ),
    percentile: float = typer.Option(
        None,
        "--percentile",
        "-pc",
        help="Percentile to remove.",
    ),
    remove_direction: Optional[str] = typer.Option(
        "Below",
        "--removeDirection",
        "-r",
        help="Remove direction above or below percentile.",
    ),
    section_var: Optional[str] = typer.Option(
        None,
        "--sectionVar",
        "-s",
        help="Variables to divide larger sections.",
    ),
    write_output: Optional[bool] = typer.Option(
        False,
        "--writeOutput",
        "-w",
        help="Write output image collection or not.",
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
    """Subset data using a given feature."""
    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--tabularDir = {tabular_dir}")
    logger.info(f"--imageFeature = {image_feature}")
    logger.info(f"--tabularFeature = {tabular_feature}")
    logger.info(f"--filePattern = {file_pattern}")
    logger.info(f"--padding = {padding}")
    logger.info(f"--groupVar = {group_var}")
    logger.info(f"--percentile = {percentile}")
    logger.info(f"--removeDirection = {remove_direction}")
    logger.info(f"--sectionVar = {section_var}")
    logger.info(f"--writeOutput = {write_output}")
    logger.info(f"--outDir = {out_dir}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert inp_dir.exists(), f"{inp_dir} does not exist!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exist!! Please check output path again"

    if preview:
        generate_preview(out_dir)

    else:
        fs.feature_subset(
            inp_dir,
            tabular_dir,
            out_dir,
            file_pattern,
            group_var,
            percentile,
            remove_direction,
            section_var,
            image_feature,
            tabular_feature,
            padding,
            write_output,
        )


if __name__ == "__main__":
    app()
