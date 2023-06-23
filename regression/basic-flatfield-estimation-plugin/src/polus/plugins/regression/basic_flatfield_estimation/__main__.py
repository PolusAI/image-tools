"""CLI for the plugin."""
import concurrent.futures
import json
import logging
import pathlib

import filepattern
import tqdm
import typer
from polus.plugins.regression.basic_flatfield_estimation import basic
from polus.plugins.regression.basic_flatfield_estimation import utils

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.regression.basic_flatfield_estimation")
logger.setLevel(utils.POLUS_LOG)

app = typer.Typer()


@app.command()
def _main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input images.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="The output directory for the flatfield images.",
    ),
    file_pattern: str = typer.Option(
        ...,
        "--filePattern",
        "-f",
        help="Input file name pattern.",
    ),
    group_by: str = typer.Option(
        ...,
        "--groupBy",
        "-g",
        help="Grouping variables for filePattern.",
    ),
    get_darkfield: bool = typer.Option(
        False,
        "--getDarkfield",
        "-d",
        help="If true, calculate darkfield contribution.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="If true, show what files would be produced.",
    ),
) -> None:
    # Checking if there is images subdirectory
    inp_dir = inp_dir.resolve()
    if inp_dir.joinpath("images").is_dir():
        inp_dir = inp_dir.joinpath("images")

    out_dir = out_dir.resolve()

    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"groupBy = {group_by}")
    logger.info(f"getDarkfield = {get_darkfield}")
    logger.info(f"preview = {preview}")

    fp = filepattern.FilePattern(str(inp_dir), file_pattern)
    extension = utils.POLUS_IMG_EXT

    if preview:
        out_dict: dict[str, list[str]] = {"files": []}
        for _, files in fp(group_by=list(group_by)):
            paths = [p for _, [p] in files]
            image_paths = [pathlib.Path(p) for p in paths]
            base_output = utils.get_output_path(image_paths)
            suffix = utils.get_suffix(base_output)
            flatfield_out = base_output.replace(suffix, "_flatfield" + extension)
            out_dict["files"].append(flatfield_out)
            if get_darkfield:
                darkfield_out = base_output.replace(suffix, "_darkfield" + extension)
                out_dict["files"].append(darkfield_out)

        with out_dir.open("w") as writer:
            json.dump(out_dict, writer, indent=4)

    else:
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=utils.MAX_WORKERS,
        ) as executor:
            for _, files in fp(group_by=list(group_by)):
                paths = [p for _, [p] in files]
                futures.append(
                    executor.submit(
                        basic,
                        paths,
                        out_dir,
                        get_darkfield,
                        extension,
                    ),
                )

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
            ):
                future.result()


if __name__ == "__main__":
    app()
