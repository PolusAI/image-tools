"""Provides the CLI for the Image Calculator plugin."""

import concurrent.futures
import json
import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.plugins.transforms.images import image_calculator

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.transforms.images.image_calculator")
logger.setLevel(image_calculator.POLUS_LOG)

app = typer.Typer()


@app.command()
def _main(  # noqa: PLR0913
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

    if preview:
        output = {"files": [files.pop()["file"].name for files in fp_primary()]}
        with out_dir.joinpath("preview.json").open("w") as writer:
            json.dump(output, writer)
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=image_calculator.MAX_WORKERS,
    ) as executor:
        futures = []

        for files in fp_primary():
            # get the first file
            file = files.pop()

            logger.info(f'Processing image: {file["file"]}')

            matches = fp_secondary.get_matching(
                **{k.upper(): v for k, v in file.items() if k != "file"},
            )
            if len(matches) > 1:
                msg = "".join(
                    [
                        "Found multiple secondary images to match the primary image: ",
                        f"{file['file'].name}. ",
                        f"Matches: {matches}",
                    ],
                )
                logger.warning(msg)
            sfile = matches.pop()

            futures.append(
                executor.submit(
                    image_calculator.process_image,
                    file["file"],
                    sfile["file"],
                    out_dir,
                    operation,
                ),
            )

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing images",
        ):
            future.result()


if __name__ == "__main__":
    app()
