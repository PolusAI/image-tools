"""CLI for the plugin."""

import concurrent.futures
import json
import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.images.transforms.images.lumos_bleedthrough_correction import lumos
from polus.images.transforms.images.lumos_bleedthrough_correction import utils

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(
    "polus.images.transforms.images.lumos_bleedthrough_correction",
)
logger.setLevel(utils.POLUS_LOG)

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to input images.",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="The output directory for the corrected images.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    file_pattern: str = typer.Option(
        ...,
        "--filePattern",
        help="Filepattern for the images.",
    ),
    group_by: str = typer.Option(
        ...,
        "--groupBy",
        help="Grouping variables for images.",
    ),
    num_fluorophores: int = typer.Option(
        ...,
        "--numFluorophores",
        help="Number of fluorophores in the images.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the results without running any computation",
    ),
) -> None:
    """Apply LUMoS bleedthrough correction to the input images and save the outputs."""
    # Checking if there is an `images` subdirectory
    if inp_dir.joinpath("images").is_dir():
        inp_dir = inp_dir.joinpath("images")

    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"groupBy = {group_by}")
    logger.info(f"numFluorophores = {num_fluorophores}")
    logger.info(f"preview = {preview}")

    fp = filepattern.FilePattern(str(inp_dir), file_pattern)
    groups: list[tuple[list[pathlib.Path], pathlib.Path]] = []
    for _, files in fp(group_by=list(group_by)):
        paths = [p for _, [p] in files]
        output_name = utils.get_output_name(paths, ".ome.zarr")
        output_path = out_dir.joinpath(output_name)
        groups.append((paths, output_path))

    if preview:
        files_dict = {"files": [o for _, o in groups]}
        with out_dir.joinpath("preview.json").open("w") as f:
            json.dump(files_dict, f, indent=2)
        return

    futures = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=utils.MAX_WORKERS,
    ) as executor:
        for input_paths, output_path in groups:
            futures.append(
                executor.submit(
                    lumos.correct,
                    input_paths,
                    num_fluorophores,
                    output_path,
                ),
            )

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
        ):
            future.result()


if __name__ == "__main__":
    app()
