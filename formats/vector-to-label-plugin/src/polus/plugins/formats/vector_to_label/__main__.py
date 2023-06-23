"""CLI for the vector-to-label plugin."""

import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.plugins.formats.vector_to_label.dynamics.vector_to_label import convert
from polus.plugins.formats.vector_to_label.utils import helpers

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = helpers.make_logger("polus.plugins.formats.vector_to_label")

app = typer.Typer()


@app.command()
def _main(
    *,
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collection to be processed by this plugin.",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Image-name pattern to use when selecting images to process.",
    ),
    flow_magnitude_threshold: float = typer.Option(
        0.1,
        "--flowMagnitudeThreshold",
        help="Cell Probability Threshold.",
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output collection",
    ),
) -> None:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        logger.critical(f"Input directory does not exist: {input_dir}")
        raise FileNotFoundError(input_dir)

    if not input_dir.is_dir():
        logger.critical(f"Input path is not a directory: {input_dir}")
        raise NotADirectoryError(input_dir)

    if input_dir.joinpath("images").is_dir():
        # switch to images folder if present
        input_dir = input_dir.joinpath("images")
        logger.info(f"inpDir = {input_dir}")

    logger.info(f"filePattern = {file_pattern}")

    if not 0.0 <= flow_magnitude_threshold <= 1.0:  # noqa: PLR2004
        msg = (
            f"flowMagnitudeThreshold must be a float between 0 and 1. "
            f"Got {flow_magnitude_threshold} instead."
        )
        logger.critical(msg)
        raise ValueError(msg)
    logger.info(f"flowMagnitudeThreshold = {flow_magnitude_threshold}")

    output_dir = output_dir.resolve()
    if not output_dir.exists():
        logger.critical(f"Output directory does not exist: {output_dir}")
        raise FileNotFoundError(output_dir)

    if not output_dir.is_dir():
        logger.critical(f"Output path is not a directory: {output_dir}")
        raise NotADirectoryError(output_dir)

    logger.info(f"outDir = {output_dir}")

    fp = filepattern.FilePattern(input_dir, file_pattern)
    files = [pathlib.Path(file[1][0]) for file in fp()]
    files = list(
        filter(lambda file_path: file_path.name.endswith("_flow.ome.zarr"), files),
    )

    if len(files) == 0:
        logger.critical("No flow files detected.")
        return

    for in_path in tqdm.tqdm(files):
        convert(in_path, flow_magnitude_threshold, output_dir)


if __name__ == "__main__":
    app()
