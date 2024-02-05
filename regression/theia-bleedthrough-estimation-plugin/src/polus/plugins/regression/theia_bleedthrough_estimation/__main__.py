"""CLI for the Theia Bleedthrough Estimation plugin."""

import concurrent.futures
import json
import logging
import pathlib
import typing

import filepattern
import tqdm
import typer
from polus.plugins.regression.theia_bleedthrough_estimation import model
from polus.plugins.regression.theia_bleedthrough_estimation import tile_selectors
from polus.plugins.regression.theia_bleedthrough_estimation.utils import constants
from polus.plugins.regression.theia_bleedthrough_estimation.utils import helpers

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = helpers.make_logger("polus.plugins.regression.theia_bleedthrough_estimation")

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input directory.",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="File pattern to match files in the input directory.",
    ),
    group_by: str = typer.Option(
        "",
        "--groupBy",
        help="Group files by these file pattern keys.",
    ),
    channel_ordering: str = typer.Option(
        "",
        "--channelOrdering",
        help="Order of channels in the input images.",
    ),
    selection_criterion: tile_selectors.Selectors = typer.Option(
        "MeanIntensity",
        "--selectionCriterion",
        help="Criterion to select tiles for training.",
    ),
    channel_overlap: int = typer.Option(
        1,
        "--channelOverlap",
        help="Number of adjacent channels to consider.",
    ),
    kernel_size: int = typer.Option(
        3,
        "--kernelSize",
        help="Size of the kernel to use for the convolution.",
    ),
    remove_interactions: bool = typer.Option(
        True,
        "--removeInteractions",
        help="Whether to remove interactions between channels.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Path to the output directory.",
        exists=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the results without running any computation.",
    ),
) -> None:
    """CLI for estimating bleedthrough using Theia."""
    if inp_dir.joinpath("images").exists():
        inp_dir = inp_dir.joinpath("images")

    grouping_variables = list(group_by)

    channel_order: typing.Optional[list[int]] = None
    if channel_ordering:
        channel_order = list(map(int, channel_ordering.split(",")))

    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f'--filePattern = "{pattern}"')
    logger.info(f"--groupBy = \"{''.join(grouping_variables)}\"")
    logger.info(f'--channelOrdering = "{channel_ordering}"')
    logger.info(f'--selectionCriterion = "{selection_criterion.value}"')
    logger.info(f"--channelOverlap = {channel_overlap}")
    logger.info(f"--kernelSize = {kernel_size}")
    logger.info(f"--removeInteractions = {remove_interactions}")
    logger.info(f"--outDir = {out_dir}")
    logger.info(f"--preview = {preview}")

    fp = filepattern.FilePattern(str(inp_dir), pattern)
    groups = [
        [pathlib.Path(p) for _, [p] in files]
        for _, files in fp(group_by=grouping_variables)
    ]

    if preview:
        logger.info("Previewing results without running any computation ...")
        metadata: dict[str, list[str]] = {"files": []}
        for image_paths in groups:
            for path in image_paths:
                metadata["files"].append(str(path))
        with out_dir.joinpath("preview.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        return

    logger.info("Running Bleedthrough Estimation ...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=constants.NUM_THREADS,
    ) as executor:
        futures = []
        for image_paths in groups:
            futures.append(
                executor.submit(
                    model.estimate_bleedthrough,
                    image_paths,
                    channel_order,
                    selection_criterion,
                    channel_overlap,
                    kernel_size,
                    remove_interactions,
                    out_dir,
                ),
            )
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            future.result()


if __name__ == "__main__":
    app()
