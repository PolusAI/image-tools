"""Provides the CLI for the Image Calculator plugin."""

import json
import logging
import pathlib

import filepattern
import preadator
import typer
from polus.images.transforms.images import image_calculator

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.transforms.images.image_calculator")
logger.setLevel(image_calculator.POLUS_LOG)

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    primary_dir: pathlib.Path = typer.Option(
        ...,
        "--primaryDir",
        help="The first set of images",
    ),
    primary_pattern: str = typer.Option(
        ".*",
        "--primaryPattern",
        help="Filename pattern used to select images.",
    ),
    operation: image_calculator.Operation = typer.Option(
        ...,
        "--operator",
        help="The operation to perform",
    ),
    secondary_dir: pathlib.Path = typer.Option(
        ...,
        "--secondaryDir",
        help="The second set of images",
    ),
    secondary_pattern: str = typer.Option(
        ".*",
        "--secondaryPattern",
        help="Filename pattern used to select images.",
    ),
    out_dir: pathlib.Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the output without saving.",
    ),
) -> None:
    """Perform simple mathematical operations on images."""
    primary_dir = primary_dir.resolve()
    if primary_dir.joinpath("images").is_dir():
        # switch to images folder if present
        primary_dir = primary_dir.joinpath("images")

    secondary_dir = secondary_dir.resolve()
    if secondary_dir.joinpath("images").is_dir():
        # switch to images folder if present
        secondary_dir = secondary_dir.joinpath("images")

    out_dir = out_dir.resolve()

    logger.info(f"primaryDir = {primary_dir}")
    logger.info(f"primaryPattern = {primary_pattern}")
    logger.info(f"operator = {operation.value}")
    logger.info(f"secondaryDir = {secondary_dir}")
    logger.info(f"secondaryPattern = {secondary_pattern}")
    logger.info(f"outDir = {out_dir}")

    fp_primary = filepattern.FilePattern(primary_dir, primary_pattern)
    fp_secondary = filepattern.FilePattern(secondary_dir, secondary_pattern)

    with preadator.ProcessManager(
        name="Image Calculator",
        num_processes=image_calculator.MAX_WORKERS,
        threads_per_process=2,
    ) as manager:
        preview_files = []
        group: dict[str, int]
        files: list[pathlib.Path]
        for group, files in fp_primary():
            for file in files:
                logger.info(f"Processing {file.name} ...")

                matches: list[pathlib.Path] = fp_secondary.get_matching(**group)[0][1]

                if len(matches) == 0:
                    # TODO: Should this raise an error?
                    msg = "".join(
                        [
                            "No secondary images found to match the ",
                            f"primary image: {file.name}. Skipping ...",
                        ],
                    )
                    logger.error(msg)
                    continue

                if preview:
                    preview_files.append(file)
                else:
                    if len(matches) > 1:
                        msg = "".join(
                            [
                                "Found multiple secondary images to match the ",
                                f"primary image: {file.name}.\n",
                                f"Matches: {matches}.\n",
                                f"Using only the first match: {matches[0]}",
                            ],
                        )
                        logger.warning(msg)
                    match = matches.pop()

                    if not preview:
                        manager.submit_process(
                            image_calculator.process_image,
                            file,
                            match,
                            out_dir,
                            operation,
                        )

        if preview:
            with out_dir.joinpath("preview.json").open("w") as writer:
                json.dump({"files": preview_files}, writer)
        else:
            manager.join_processes()


if __name__ == "__main__":
    app()
